"""Frozen V1 completion encoder for compatibility tests (not the V2 writer).

Only volatile values come from the test inputs. The V1 schema and physical hash
payload are deliberately independent of the engine's binding implementation.
"""
import hashlib
import json


def physical(root, slot, stores=None):
    info = root.stat()
    payload = dict(root=str(root), identity=[info.st_dev, info.st_ino, info.st_mode], epoch_id=slot)
    if stores is not None:
        payload['store_bindings'] = stores
    return hashlib.sha256((json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False) + '\n').encode()).hexdigest()


def convert_intent(root, epoch, success_path, index=None, stores=None):
    """Build a V1 fixture before testing readers; never used on real records."""
    target = root / epoch if index is None else root / epoch / str(index)
    path = target / 'completion.json'
    source = json.loads(path.read_bytes())
    keys = ('event', 'operation_id', 'epoch_id', 'current_cycle_id', 'request_digest',
            'manifest_digest', 'time_policy', 'bundle_verified_at_utc', 'preparation_started_at_utc')
    record = dict(schema_version=1, **{key: source[key] for key in keys})
    bindings = None
    if index is not None:
        bindings = {role: physical(value / epoch, 'CONTINUING_%s_ROOT' % role.upper()) for role, value in stores.items()}
        record['store_bindings'] = bindings
    record['target_binding_digest'] = physical(target.parent, target.name, bindings)
    path.write_text(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')
    success = json.loads(success_path.read_bytes())
    success['target_binding_digest'] = record['target_binding_digest']
    success_path.write_text(json.dumps(success, sort_keys=True, separators=(',', ':')))
