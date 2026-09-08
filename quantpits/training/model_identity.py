"""Non-executing identity for copied pickle models and explicit recorder ancestry.

Raw hashes remain storage-integrity evidence. This stricter-than-object-equality
projection ignores only pickle framing and Torch legacy storage allocation IDs.
It never imports Torch or executes pickle GLOBAL/REDUCE instructions.
"""
import hashlib
import pickletools

from quantpits.evidence.contracts import canonical_json_bytes


class ModelIdentityError(ValueError):
    pass


def _ops(data):
    ops = list(pickletools.genops(data))
    if not ops or ops[-1][0].name != "STOP":
        raise ModelIdentityError("incomplete pickle")
    return ops


def model_content_digest(data):
    """Versioned complete-object fingerprint; unsupported encodings fail closed.

    Tensor shapes/strides, configuration, optimizer state and alias relationships
    remain in the projection. False negatives for other serialization formats are
    preferable to discarding unknown object state.
    """
    if not isinstance(data, bytes) or len(data) > 128 * 1024 * 1024:
        raise ModelIdentityError("model size/type invalid")
    allocation_ids = {}

    def storage(blob):
        offset, headers = 0, []
        for _ in range(5):
            ops = _ops(blob[offset:])
            offset += ops[-1][2] + 1
            headers.append(ops)
        values = [[arg for op, arg, _ in ops if op.name not in ("PROTO", "STOP")]
                  for ops in headers]
        if values[0] != [119547037146038801333356] or values[1] != [1001]:
            raise ModelIdentityError("unsupported Torch storage header")
        # A legacy storage pickle contains exactly one persistent storage tuple
        # followed by a one-element storage-key list. Keep all other instructions.
        strings = [arg for op, arg, _ in headers[3] if op.name == "BINUNICODE"]
        keys = [arg for op, arg, _ in headers[4] if op.name == "BINUNICODE"]
        if len(strings) != 3 or strings[0] != "storage" or keys != [strings[1]]:
            raise ModelIdentityError("unsupported storage keys")
        identifier = strings[1]
        if not identifier.isdigit() or not any(op.name == "BINPERSID" for op, _, _ in headers[3]):
            raise ModelIdentityError("unsupported storage identity")
        globals_ = [arg for op, arg, _ in headers[3] if op.name == "GLOBAL"]
        if len(globals_) != 1 or not globals_[0].startswith("torch ") or not globals_[0].endswith("Storage"):
            raise ModelIdentityError("unsupported storage type")
        body = blob[offset:]
        if len(body) < 8:
            raise ModelIdentityError("truncated storage")
        body_digest = hashlib.sha256(body).hexdigest()
        # IDs may recur: preserve alias identity and reject inconsistent content.
        if identifier not in allocation_ids:
            allocation_ids[identifier] = (len(allocation_ids), body_digest)
        index, previous = allocation_ids[identifier]
        if previous != body_digest:
            raise ModelIdentityError("conflicting storage identity")
        projected = []
        for i, ops in enumerate(headers):
            projected.append([(op.name, ["allocation", index] if i in (3, 4)
                               and op.name == "BINUNICODE" and arg == identifier else arg)
                              for op, arg, _ in ops])
        return ["torch_legacy_storage", projected, len(body), body_digest]

    try:
        ops = _ops(data)
        if ops[-1][2] + 1 != len(data):
            raise ModelIdentityError("trailing pickle bytes")
        rows = []
        for op, arg, _ in ops:
            if op.name == "FRAME":
                continue
            name = op.name
            if isinstance(arg, bytes):
                name = "BYTES" if name in ("SHORT_BINBYTES", "BINBYTES", "BINBYTES8") else name
                # Recognize only the Torch serialization magic, not arbitrary pickles.
                if arg.startswith(b'\x80\x02\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19'):
                    arg = storage(arg)
                else:
                    arg = ["raw", len(arg), hashlib.sha256(arg).hexdigest()]
            elif isinstance(arg, float):
                arg = ["float", arg.hex()]
            rows.append([name, arg])
        return hashlib.sha256(canonical_json_bytes({
            "protocol": "COPIED_MODEL_PICKLE_CONTENT_V1", "ops": rows,
        })).hexdigest()
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        raise ModelIdentityError("model content cannot be compared") from exc


def trace_training_origin(experiment, recorder_id, read_tags, model_name, max_depth=128):
    """Walk explicit direct-parent tags; never treat a cached root tag as proof.

    read_tags is a caller-owned reader. Legacy untagged terminal training runs
    are accepted only when neither parent tag is present. Missing/partial parents,
    mismatched models, cycles and cached-root contradictions are errors.
    """
    seen, chain = set(), []
    for _ in range(max_depth):
        identity = (experiment, recorder_id)
        if not all(isinstance(v, str) and v for v in identity) or identity in seen:
            raise ModelIdentityError("invalid or cyclic model lineage")
        seen.add(identity)
        tags = read_tags(experiment, recorder_id)
        if not isinstance(tags, dict) or tags.get("model") != model_name:
            raise ModelIdentityError("model lineage tags invalid")
        chain.append((identity, tags))
        parent, parent_exp = tags.get("source_record_id"), tags.get("source_experiment")
        if bool(parent) != bool(parent_exp):
            raise ModelIdentityError("partial model lineage")
        if not parent:
            if tags.get("mode") not in (None, "train", "static", "cpcv", "cpcv_train"):
                raise ModelIdentityError("prediction has no training parent")
            for _, entry in chain:
                root = (entry.get("training_origin_experiment"), entry.get("training_origin_record_id"))
                if root != (None, None) and root != identity:
                    raise ModelIdentityError("training origin tag conflicts with lineage")
            return {"training_origin_experiment": experiment,
                    "training_origin_record_id": recorder_id}
        experiment, recorder_id = parent_exp, parent
    raise ModelIdentityError("model lineage exceeds depth limit")


def prediction_origin_tags(experiment, recorder_id, read_tags, model_name):
    """Annotate a prediction without requiring retention of every legacy run.

    Missing historical ancestry does not invalidate an available prediction
    model. It does deny verified-origin claims and continuity admission.
    """
    try:
        origin = trace_training_origin(experiment, recorder_id, read_tags, model_name)
        return dict(origin, training_origin_status='VERIFIED')
    except Exception:
        return {'training_origin_status': 'UNRESOLVED'}
