"""Inspector-owned construction and create-only publication of cycle bundles."""

from __future__ import annotations

import errno
import ctypes
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import (
    CaptureRequest,
    CaptureResult,
    ContractError,
    DecisionEvent,
    TypedDigest,
    _RESULT_AUTHORITY,
    canonical_json_bytes,
)
from quantpits.evidence.inspection import (
    FileSnapshot,
    PathBoundaryError,
    contained_path,
    inspect_file,
    inspect_git,
    inspect_many,
    inspect_tree,
    parse_json,
    root_identity,
    SourceMutationObserver,
)
from quantpits.evidence.ranking import RankingResult, canonical_full_ranking


SCHEMA_VERSION = 1
EMBED_LIMIT = 2 * 1024 * 1024
EXPECTED_COMMANDS = {
    "post_trade": {"post-trade", "prod_post_trade"},
    "prediction": {"static_train", "static-train"},
    "ensemble": {"ensemble_fusion", "ensemble-fusion"},
    "order": {"order_gen", "order-gen"},
}


def _result(
    cycle_id: str, status: str, did_write: bool, bundle_path: Optional[str],
    seal_digest: Optional[TypedDigest], problems: Tuple[dict, ...] = (),
    *, sealed_status: Optional[str] = None,
) -> CaptureResult:
    if sealed_status is None and status in {"sealed_complete", "sealed_partial"}:
        sealed_status = status
    return CaptureResult(
        cycle_id, status, did_write, bundle_path, seal_digest, problems,
        sealed_status, _authority=_RESULT_AUTHORITY,
    )


def _problem(code: str, evidence_class: str, detail: str, *, blocking: bool = False) -> dict:
    return {
        "code": code, "evidence_class": evidence_class,
        "detail": detail[:1000], "blocks_complete": bool(blocking),
    }


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()]
    directories.sort(key=lambda path: len(path.parts), reverse=True)
    for directory in directories:
        _fsync_dir(directory)
    _fsync_dir(root)


def _safe_mkdirs(root: Path, target: Path) -> None:
    canonical_root = root.resolve(strict=True)
    try:
        relative = target.absolute().relative_to(canonical_root)
    except ValueError as exc:
        raise PathBoundaryError("write parent is outside workspace") from exc
    current = canonical_root
    for part in relative.parts:
        current = current / part
        try:
            os.lstat(str(current))
            if not os.path.isdir(str(current)) or os.path.islink(str(current)):
                raise PathBoundaryError("write parent contains a non-directory alias")
        except FileNotFoundError:
            parent = current.parent
            os.mkdir(str(current), 0o700)
            os.lstat(str(current))
            _fsync_dir(parent)
        if current.resolve(strict=True).parent != current.parent.resolve(strict=True):
            raise PathBoundaryError("write parent escaped during creation")


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic create-only directory publish is unsupported")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(str(source)), -100, os.fsencode(str(target)), 1,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def _relative_existing(root: Path, value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise PathBoundaryError("referenced evidence escapes workspace") from exc
    return contained_path(root, value).relative_to(root.resolve()).as_posix()


def _extract_anchor(manifest: Mapping[str, Any]) -> Optional[str]:
    records = manifest.get("records", {})
    if not isinstance(records, Mapping):
        return None
    for name in ("expected_anchor", "anchor_date", "cycle_id"):
        value = records.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _manifest_refs(manifest: Mapping[str, Any]) -> Tuple[str, ...]:
    refs = []
    for collection in ("inputs", "outputs"):
        values = manifest.get(collection, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                path = item["path"]
                if "<" not in path and path not in refs:
                    refs.append(path)
    return tuple(refs)


def _universe_from_file(data: bytes, anchor: Optional[str]) -> Tuple[str, ...]:
    text = data.decode("utf-8")
    members = []
    for raw in text.splitlines():
        fields = raw.strip().replace(",", "\t").split()
        if not fields:
            continue
        if len(fields) >= 3 and anchor:
            if fields[1][:10] > anchor[:10] or fields[2][:10] < anchor[:10]:
                continue
        members.append(fields[0])
    return tuple(members)


def _selected_combo(records: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    combos = records.get("combos")
    if not isinstance(combos, list) or not combos:
        return None
    defaults = [item for item in combos if isinstance(item, Mapping) and item.get("is_default") is True]
    selected = defaults if defaults else [item for item in combos if isinstance(item, Mapping)]
    return selected[0] if len(selected) == 1 else None


def _selected_model_evidence(
    records: Mapping[str, Any], resolved_members: Sequence[str],
) -> Tuple[Mapping[str, Any], ...]:
    raw = records.get("input_models", [])
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise ContractError("M3 source model inventory is invalid")
    by_key = {}
    for item in raw:
        key = item.get("resolved_key")
        if key in by_key:
            raise ContractError("M3 source model inventory has duplicate identity")
        if isinstance(key, str):
            by_key[key] = item
    if set(by_key).intersection(resolved_members) != set(resolved_members):
        raise ContractError("M3 source model inventory does not cover exact combo members")
    return tuple(by_key[key] for key in resolved_members)


@dataclass
class _BundleDraft:
    manifest: dict
    named_files: Dict[str, bytes]
    objects: Dict[str, bytes]
    problems: list
    blocked: bool = False
    source_digests: Optional[Dict[str, Optional[TypedDigest]]] = None


class ProductionCycleEvidenceSealer:
    """The sole truth owner for Phase 37A evidence and seal capability."""

    def __init__(
        self, workspace_root: Path, *, engine_root: Optional[Path] = None,
        qlib_data_dir: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
        fault_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.root = Path(workspace_root).resolve(strict=True)
        self.engine_root = (engine_root or Path(__file__).resolve().parents[2]).resolve(strict=True)
        self.qlib_data_dir = Path(qlib_data_dir).expanduser().resolve() if qlib_data_dir else None
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.fault_hook = fault_hook or (lambda _point: None)

    def _embed(self, draft: _BundleDraft, snapshot: FileSnapshot) -> dict:
        if snapshot.status != "observed" or snapshot.digest is None:
            return snapshot.to_public_dict("missing" if snapshot.status == "missing" else "incomparable")
        digest = snapshot.digest.value
        if snapshot.data is not None and len(snapshot.data) <= EMBED_LIMIT:
            draft.objects.setdefault(digest, snapshot.data)
            preservation = "embedded"
        else:
            preservation = "workspace_file"
        return snapshot.to_public_dict(preservation)

    def _engine_surface(self, draft: _BundleDraft) -> Tuple[dict, Dict[str, TypedDigest]]:
        members = [
            "quantpits/evidence/__init__.py",
            "quantpits/evidence/contracts.py",
            "quantpits/evidence/inspection.py",
            "quantpits/evidence/ranking.py",
            "quantpits/evidence/sealing.py",
            "quantpits/scripts/cycle_evidence.py",
            "quantpits/config_contracts/normalizers.py",
            "quantpits/utils/workspace.py",
        ]
        snapshots = inspect_many(self.engine_root, tuple((path, path) for path in members))
        public = []
        digests = {}
        for path, snapshot in snapshots:
            public.append(self._embed(draft, snapshot))
            if snapshot.status != "observed" or snapshot.digest is None:
                draft.problems.append(_problem("engine_surface_incomparable", "engine", path, blocking=True))
            else:
                digests[path] = snapshot.digest
        return {
            "members": public,
            "surface_digest": TypedDigest.canonical([
                {"path": path, "digest": digest.to_dict()}
                for path, digest in sorted(digests.items())
            ], "file_inventory").to_dict(),
        }, digests

    def _observe_sources(self, request: CaptureRequest, draft: _BundleDraft) -> Tuple[dict, Dict[str, FileSnapshot]]:
        requested = [(name, path) for name, path, _required in request.source_paths() if path]
        observations = inspect_many(self.root, requested)
        by_name = {name: snapshot for name, snapshot in observations}
        manifests = {}
        public = []
        for name, path, _required in request.source_paths():
            if not path:
                if name == "deep_analysis":
                    draft.problems.append(_problem("deep_analysis_missing", name, "no Deep Analysis run was provided", blocking=True))
                continue
            snapshot = by_name[name]
            public.append({"class": name, **self._embed(draft, snapshot)})
            parsed = parse_json(snapshot) if name != "deep_analysis" else None
            if name == "deep_analysis":
                tree = inspect_tree(self.root, path)
                public[-1]["members"] = [self._embed(draft, item) for item in tree]
                if not tree or any(item.status != "observed" for item in tree):
                    draft.problems.append(_problem("deep_analysis_incomplete", name, "trace tree is absent or incomparable", blocking=True))
                else:
                    inventory = [
                        {"path": item.logical_path, "digest": item.digest.to_dict()}
                        for item in tree if item.digest is not None
                    ]
                    inventory_data = canonical_json_bytes(inventory)
                    aggregate = FileSnapshot(
                        path, "observed", inventory_data,
                        TypedDigest.canonical(inventory, "file_inventory"), None,
                    )
                    by_name[name] = aggregate
                    members = public[-1]["members"]
                    public[-1] = {"class": name, **self._embed(draft, aggregate), "members": members}
            elif name == "decision":
                # Decision is validated separately; malformed input remains visible.
                pass
            elif parsed is None:
                draft.problems.append(_problem("manifest_invalid", name, "manifest is missing or invalid JSON", blocking=True))
            else:
                manifests[name] = parsed
                if parsed.get("status") != "success":
                    draft.problems.append(_problem("run_not_success", name, "manifest status is not success", blocking=True))
                if parsed.get("command") not in EXPECTED_COMMANDS.get(name, set()):
                    draft.problems.append(_problem("manifest_command_mismatch", name, "manifest command does not match its evidence class", blocking=True))
                if not isinstance(parsed.get("run_id"), str) or not parsed.get("run_id"):
                    draft.problems.append(_problem("manifest_run_id_missing", name, "manifest has no exact run ID", blocking=True))
        draft.manifest["run_evidence"] = public
        draft.source_digests = {name: snapshot.digest for name, snapshot in by_name.items()}
        return manifests, by_name

    def _decision(self, request: CaptureRequest, snapshot: Optional[FileSnapshot], draft: _BundleDraft) -> dict:
        if not request.decision_event:
            return {"status": "not_recorded", "as_of_capture": True}
        raw = parse_json(snapshot) if snapshot else None
        try:
            event = DecisionEvent.from_mapping(raw)  # type: ignore[arg-type]
            if event.evidence_cycle_id != request.cycle_id:
                raise ContractError("decision event belongs to a foreign cycle")
            canonical = event.to_dict()
            return {
                "status": "recorded", "event": canonical,
                "raw_digest": snapshot.digest.to_dict() if snapshot and snapshot.digest else None,
                "canonical_digest": TypedDigest.canonical(canonical).to_dict(),
            }
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            draft.problems.append(_problem("decision_invalid", "decision", str(exc), blocking=True))
            return {"status": "invalid", "detail": str(exc)[:1000]}

    def _portfolio(self, draft: _BundleDraft) -> dict:
        snapshot = inspect_file(self.root, "config/prod_config.json")
        public = self._embed(draft, snapshot)
        if snapshot.status != "observed" or snapshot.data is None:
            draft.problems.append(_problem("portfolio_missing", "portfolio", snapshot.detail, blocking=True))
            return public
        try:
            raw = json.loads(snapshot.data.decode("utf-8"))
            cash = raw.get("current_cash")
            holdings = raw.get("current_holding")
            if not isinstance(holdings, list) or isinstance(cash, bool) or not isinstance(cash, (int, float)):
                raise ContractError("prod_config lacks canonical cash/holding state")
            canonical = {"current_cash": cash, "current_holding": holdings}
            data = canonical_json_bytes(canonical)
            draft.named_files["portfolio_state.json"] = data
            public["canonical_digest"] = TypedDigest.canonical(canonical, "semantic_config").to_dict()
            public["holding_count"] = len(holdings)
            return public
        except Exception as exc:
            draft.problems.append(_problem("portfolio_invalid", "portfolio", str(exc), blocking=True))
            return public

    def _frozen_market(self, ensemble: Optional[Mapping[str, Any]], draft: _BundleDraft) -> Optional[str]:
        if not ensemble:
            return None
        expected = None
        fingerprints = ensemble.get("config_fingerprints", {})
        if isinstance(fingerprints, Mapping):
            expected = fingerprints.get("model_config")
        if expected is None:
            for item in ensemble.get("inputs", []) if isinstance(ensemble.get("inputs"), list) else []:
                if isinstance(item, Mapping) and item.get("path") == "config/model_config.json":
                    expected = item.get("fingerprint")
                    break
        if not isinstance(expected, str):
            return None
        snapshot = inspect_file(self.root, "config/model_config.json")
        if snapshot.data is None:
            draft.problems.append(_problem("market_config_missing", "data", "frozen model config is unavailable", blocking=True))
            return None
        try:
            from quantpits.config_contracts.normalizers import normalize_model_config
            from quantpits.utils.workspace import fingerprint_value

            raw = json.loads(snapshot.data.decode("utf-8"))
            normalized = normalize_model_config(raw)
            if fingerprint_value(normalized) != expected:
                raise ContractError("model config differs from M3 frozen fingerprint")
            market = normalized.get("market")
            if not isinstance(market, str) or not market:
                raise ContractError("frozen model config has no market")
            return market.lower()
        except Exception as exc:
            draft.problems.append(_problem("market_config_incomparable", "data", str(exc), blocking=True))
            return None

    def _data_identity(
        self, anchor: Optional[str], market: Optional[str], draft: _BundleDraft,
    ) -> Tuple[dict, Optional[FileSnapshot]]:
        qlib = self.qlib_data_dir
        if qlib is None:
            configured = os.environ.get("QLIB_DATA_DIR")
            qlib = Path(configured).expanduser().resolve() if configured else None
        result = {
            "source_dolt_identity": {"status": "missing"},
            "source_to_materialization_relation": "unverified",
        }
        if qlib is None or not qlib.is_dir():
            result["qlib_materialization_identity"] = {"status": "missing"}
            draft.problems.append(_problem("qlib_identity_missing", "data", "Qlib materialization was not configured", blocking=True))
            return result, None
        # Qlib is a separate read-only authority and is intentionally not forced
        # under the private workspace. Only logical component names are emitted.
        calendar_path = qlib / "calendars" / "day.txt"
        instrument_files = sorted((qlib / "instruments").glob("*.txt")) if (qlib / "instruments").is_dir() else []
        try:
            if calendar_path.is_symlink():
                raise RuntimeError("calendar public name is a symlink")
            before = os.lstat(str(calendar_path))
            with calendar_path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                calendar_data = handle.read()
            after = os.lstat(str(calendar_path))
            identities = [
                (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
                for item in (before, opened, after)
            ]
            if identities[0] != identities[1] or identities[0] != identities[2] or before.st_nlink != 1:
                raise RuntimeError("calendar changed while reading")
            calendars = [line.strip() for line in calendar_data.decode().splitlines() if line.strip()]
            matching = [path for path in instrument_files if market and path.stem.lower() == market]
            universe_path = matching[0] if len(matching) == 1 else (instrument_files[0] if len(instrument_files) == 1 else None)
            universe_snapshot = None
            if universe_path:
                if universe_path.is_symlink():
                    raise RuntimeError("universe public name is a symlink")
                universe_before = os.lstat(str(universe_path))
                with universe_path.open("rb") as handle:
                    universe_opened = os.fstat(handle.fileno())
                    universe_data = handle.read()
                universe_after = os.lstat(str(universe_path))
                universe_ids = [
                    (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
                    for item in (universe_before, universe_opened, universe_after)
                ]
                if universe_ids[0] != universe_ids[1] or universe_ids[0] != universe_ids[2] or universe_before.st_nlink != 1:
                    raise RuntimeError("universe changed while reading")
                universe_snapshot = FileSnapshot(
                    "qlib/instruments/%s" % universe_path.name, "observed", universe_data,
                    TypedDigest.raw(universe_data), None,
                )
            result["qlib_materialization_identity"] = {
                "status": "observed",
                "calendar_cutoff": calendars[-1][:10] if calendars else None,
                "calendar_digest": TypedDigest.raw(calendar_data).to_dict(),
                "universe_digest": universe_snapshot.digest.to_dict() if universe_snapshot else None,
                "universe_name": universe_path.stem if universe_path else None,
            }
            if not calendars or (anchor and anchor[:10] not in {item[:10] for item in calendars}):
                draft.problems.append(_problem("calendar_anchor_missing", "data", "cycle anchor is absent from Qlib calendar", blocking=True))
            if universe_snapshot is None:
                draft.problems.append(_problem("universe_ambiguous", "data", "exact eligible universe file is not unique", blocking=True))
            return result, universe_snapshot
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            result["qlib_materialization_identity"] = {"status": "incomparable", "detail": str(exc)[:1000]}
            draft.problems.append(_problem("qlib_identity_incomparable", "data", str(exc), blocking=True))
            return result, None

    def _ranking(
        self, ensemble: Optional[Mapping[str, Any]], anchor: Optional[str],
        universe_snapshot: Optional[FileSnapshot], draft: _BundleDraft,
    ) -> dict:
        if not ensemble:
            draft.problems.append(_problem("ranking_source_missing", "ranking", "M3 manifest is invalid", blocking=True))
            return {"status": "unavailable"}
        records = ensemble.get("records", {})
        if not isinstance(records, Mapping):
            records = {}
        combo = _selected_combo(records)
        if combo is None:
            draft.problems.append(_problem("ranking_combo_ambiguous", "ranking", "M3 does not select one exact combo", blocking=True))
            return {"status": "unavailable"}
        resolved_members = combo.get("resolved_models", combo.get("models", []))
        if (
            not isinstance(resolved_members, list)
            or not resolved_members
            or len(set(resolved_members)) != len(resolved_members)
            or any(not isinstance(item, str) or not item for item in resolved_members)
            or not isinstance(combo.get("method"), str)
            or not combo.get("method")
        ):
            draft.problems.append(_problem("ranking_combo_invalid", "ranking", "M3 combo definition is not canonical", blocking=True))
            return {"status": "unavailable"}
        pred_ref = combo.get("recorder_id")
        scores = None
        prediction_digest = None
        try:
            source_models = _selected_model_evidence(records, resolved_members)
            if isinstance(pred_ref, str) and pred_ref:
                import pandas as pd

                output_evidence = combo.get("output_evidence", {})
                if not isinstance(output_evidence, Mapping):
                    raise ContractError("M3 output recorder evidence is absent")
                if (
                    output_evidence.get("contained") is not True
                    or output_evidence.get("recorder_id") != pred_ref
                ):
                    raise ContractError("M3 output recorder identity is inconsistent")
                artifact_path = output_evidence.get("artifact_path")
                if not isinstance(artifact_path, str):
                    raise ContractError("M3 output artifact path is absent")
                relative = _relative_existing(self.root, artifact_path)
                pred_relative = (Path(relative) / "pred.pkl").as_posix()
                snapshot = inspect_file(self.root, pred_relative)
                if snapshot.status != "observed":
                    raise ContractError("M3 output pred.pkl is incomparable")
                prediction = pd.read_pickle(contained_path(self.root, pred_relative))
                if getattr(prediction, "name", None) == "score":
                    prediction = prediction.to_frame("score")
                if not hasattr(prediction, "index") or "score" not in prediction:
                    raise ContractError("M3 output pred.pkl schema is invalid")
                if not anchor or "datetime" not in prediction.index.names or "instrument" not in prediction.index.names:
                    raise ContractError("M3 output pred.pkl lacks exact datetime/instrument identity")
                dates = prediction.index.get_level_values("datetime")
                selected = [str(value)[:10] == anchor[:10] for value in dates]
                prediction = prediction[selected]
                if prediction.empty:
                    raise ContractError("M3 output pred.pkl has no exact anchor rows")
                prediction = prediction.droplevel([name for name in prediction.index.names if name != "instrument"])
                if prediction.index.has_duplicates:
                    raise ContractError("M3 output pred.pkl has duplicate instrument rows")
                scores = {str(key): value for key, value in prediction["score"].items()}
                prediction_digest = snapshot.digest.to_dict() if snapshot.digest else None
            else:
                raise ContractError("M3 combo lacks an exact recorder reference")
            if universe_snapshot is None or universe_snapshot.data is None:
                raise ContractError("eligible universe is not provable")
            universe = _universe_from_file(universe_snapshot.data, anchor)
            result: RankingResult = canonical_full_ranking(universe, scores)
            ranking_bytes = result.to_csv_bytes()
            draft.named_files["ranking.csv"] = ranking_bytes
            ranking_digest = TypedDigest.raw(ranking_bytes)
            if not result.complete:
                draft.problems.append(_problem("ranking_coverage_partial", "ranking", "eligible members lack finite predictions", blocking=True))
            return {
                "status": "complete" if result.complete else "partial",
                "eligible_count": result.eligible_count,
                "scored_count": result.scored_count,
                "missing_count": result.missing_count,
                "prediction_digest": prediction_digest,
                "ranking_digest": ranking_digest.to_dict(),
                "combo": {
                    "name": combo.get("name"), "method": combo.get("method"),
                    "resolved_members": resolved_members,
                    "ensemble_recorder_id": pred_ref,
                    "output_evidence": combo.get("output_evidence", {}),
                },
                "source_models": list(source_models),
            }
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            draft.problems.append(_problem("ranking_unavailable", "ranking", str(exc), blocking=True))
            return {"status": "unavailable", "detail": str(exc)[:1000]}

    def _referenced_evidence(self, manifests: Mapping[str, Mapping[str, Any]], draft: _BundleDraft) -> list:
        observed = []
        seen = set()
        for evidence_class, manifest in manifests.items():
            for ref in _manifest_refs(manifest):
                try:
                    relative = _relative_existing(self.root, ref)
                except FileNotFoundError:
                    relative = ref if not Path(ref).is_absolute() else "<unavailable>"
                    draft.problems.append(_problem("referenced_file_missing", evidence_class, "referenced output is missing", blocking=True))
                    continue
                except PathBoundaryError as exc:
                    draft.problems.append(_problem("referenced_path_escape", evidence_class, str(exc), blocking=True))
                    draft.blocked = True
                    continue
                if relative in seen:
                    continue
                seen.add(relative)
                snapshot = inspect_file(self.root, relative)
                observed.append(self._embed(draft, snapshot))
                if snapshot.status != "observed":
                    draft.problems.append(_problem("referenced_file_incomparable", evidence_class, relative, blocking=True))
        return observed

    def _lineage_artifacts(
        self, ensemble: Optional[Mapping[str, Any]], anchor: Optional[str],
        draft: _BundleDraft,
    ) -> list:
        if not ensemble or not isinstance(ensemble.get("records"), Mapping):
            return []
        records = ensemble["records"]
        observed = []
        combo = _selected_combo(records)
        resolved = combo.get("resolved_models", combo.get("models", [])) if combo else []
        try:
            models = _selected_model_evidence(records, resolved)
        except ContractError as exc:
            draft.problems.append(_problem("model_lineage_missing", "model", str(exc), blocking=True))
            return observed
        if not models:
            draft.problems.append(_problem("model_lineage_missing", "model", "M3 has no source model evidence", blocking=True))
            return observed
        for position, model in enumerate(models):
            if not isinstance(model, Mapping):
                draft.problems.append(_problem("model_lineage_invalid", "model", "source model evidence is not an object", blocking=True))
                continue
            artifact_path = model.get("artifact_path")
            recorder_id = model.get("recorder_id")
            identity_valid = (
                isinstance(recorder_id, str) and bool(recorder_id)
                and isinstance(artifact_path, str)
                and model.get("status") == "ready"
                and isinstance(model.get("experiment_name"), str)
                and bool(model.get("experiment_name"))
                and isinstance(model.get("experiment_id"), str)
                and bool(model.get("experiment_id"))
                and model.get("prediction_end") == anchor
            )
            if not identity_valid:
                draft.problems.append(_problem("model_lineage_invalid", "model", "source recorder/artifact identity is absent", blocking=True))
                continue
            try:
                relative = _relative_existing(self.root, artifact_path)
                members = inspect_tree(self.root, relative)
                public_members = [self._embed(draft, item) for item in members]
                if not members or any(item.status != "observed" for item in members):
                    raise ContractError("model artifact tree is incomparable")
                observed.append({
                    "position": position, "recorder_id": recorder_id,
                    "source_recorder_id": model.get("source_recorder_id"),
                    "artifact_locator": relative, "members": public_members,
                    "artifact_tree_digest": TypedDigest.canonical([
                        {"path": item.logical_path, "digest": item.digest.to_dict()}
                        for item in members if item.digest is not None
                    ], "file_inventory").to_dict(),
                })
                source_recorder_id = model.get("source_recorder_id")
                if source_recorder_id:
                    if not isinstance(source_recorder_id, str) or not model.get("source_experiment_name"):
                        raise ContractError("source training recorder identity is invalid")
                    source_path = model.get("source_artifact_path")
                    if source_recorder_id == recorder_id and not isinstance(source_path, str):
                        source_path = artifact_path
                    if not isinstance(source_path, str):
                        candidates = []
                        for base_name in ("mlruns", "mlartifacts"):
                            base = self.root / base_name
                            if base.is_dir() and not base.is_symlink():
                                candidates.extend(base.glob("*/%s/artifacts" % source_recorder_id))
                        contained = [
                            path for path in candidates
                            if path.is_dir() and not path.is_symlink()
                        ]
                        if len(contained) != 1:
                            raise ContractError("source training artifact path is not uniquely observable")
                        source_path = contained[0].relative_to(self.root).as_posix()
                    source_relative = _relative_existing(self.root, source_path)
                    source_members = inspect_tree(self.root, source_relative)
                    if not source_members or any(item.status != "observed" for item in source_members):
                        raise ContractError("source training artifact tree is incomparable")
                    observed.append({
                        "position": position, "role": "source_training",
                        "recorder_id": source_recorder_id,
                        "experiment_name": model.get("source_experiment_name"),
                        "artifact_locator": source_relative,
                        "members": [self._embed(draft, item) for item in source_members],
                        "artifact_tree_digest": TypedDigest.canonical([
                            {"path": item.logical_path, "digest": item.digest.to_dict()}
                            for item in source_members if item.digest is not None
                        ], "file_inventory").to_dict(),
                    })
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as exc:
                draft.problems.append(_problem("model_artifact_incomparable", "model", str(exc), blocking=True))
        output = combo.get("output_evidence", {}) if combo else {}
        if isinstance(output, Mapping) and isinstance(output.get("artifact_path"), str):
            try:
                relative = _relative_existing(self.root, output["artifact_path"])
                members = inspect_tree(self.root, relative)
                if not members or any(item.status != "observed" for item in members):
                    raise ContractError("ensemble artifact tree is incomparable")
                observed.append({
                    "position": "ensemble", "recorder_id": combo.get("recorder_id"),
                    "artifact_locator": relative,
                    "members": [self._embed(draft, item) for item in members],
                    "artifact_tree_digest": TypedDigest.canonical([
                        {"path": item.logical_path, "digest": item.digest.to_dict()}
                        for item in members if item.digest is not None
                    ], "file_inventory").to_dict(),
                })
            except Exception as exc:
                draft.problems.append(_problem("ensemble_artifact_incomparable", "ensemble", str(exc), blocking=True))
        return observed

    def _build(
        self, request: CaptureRequest, observer: SourceMutationObserver,
        data_observer: Optional[SourceMutationObserver],
    ) -> _BundleDraft:
        draft = _BundleDraft({}, {}, {}, [], source_digests={})
        original_root = root_identity(self.root)
        manifests, snapshots = self._observe_sources(request, draft)
        anchors = {name: _extract_anchor(value) for name, value in manifests.items()}
        for name in ("post_trade", "prediction", "ensemble", "order"):
            if name in manifests and anchors.get(name) is None:
                draft.problems.append(_problem("manifest_anchor_missing", name, "manifest has no exact cycle anchor", blocking=True))
        ensemble_records = manifests.get("ensemble", {}).get("records", {})
        if isinstance(ensemble_records, Mapping):
            expected_anchor = ensemble_records.get("expected_anchor")
            actual_anchor = ensemble_records.get("anchor_date")
            if expected_anchor != actual_anchor:
                draft.problems.append(_problem("ensemble_anchor_mismatch", "ensemble", "expected and actual ensemble anchors differ", blocking=True))
        comparable = {value for value in anchors.values() if value}
        if len(comparable) > 1:
            draft.problems.append(_problem("anchor_mismatch", "cycle", "run manifests disagree on cycle anchor", blocking=True))
        anchor = _extract_anchor(manifests.get("ensemble", {})) or (next(iter(comparable)) if comparable else None)
        engine_git = inspect_git(self.engine_root)
        workspace_git = inspect_git(self.root)
        engine_surface, engine_surface_digests = self._engine_surface(draft)
        if engine_git.get("status") == "dirty_unresolved":
            draft.problems.append(_problem("engine_git_unresolved", "engine", "engine Git identity is unresolved", blocking=True))
        if workspace_git.get("status") == "dirty_unresolved":
            draft.problems.append(_problem("workspace_git_unresolved", "workspace", "workspace Git identity is unresolved", blocking=True))
        market = self._frozen_market(manifests.get("ensemble"), draft)
        data_identity, universe = self._data_identity(anchor, market, draft)
        ranking = self._ranking(manifests.get("ensemble"), anchor, universe, draft)
        lineage_artifacts = self._lineage_artifacts(manifests.get("ensemble"), anchor, draft)
        portfolio = self._portfolio(draft)
        decision = self._decision(request, snapshots.get("decision"), draft)
        referenced = self._referenced_evidence(manifests, draft)
        engine_git_after = inspect_git(self.engine_root)
        workspace_git_after = inspect_git(self.root)
        git_keys = ("commit", "tree", "status", "status_inventory_digest", "tracked_diff_digest")
        if any(engine_git.get(key) != engine_git_after.get(key) for key in git_keys):
            draft.problems.append(_problem("engine_git_mutated", "engine", "engine Git facts changed during capture", blocking=True))
        if any(workspace_git.get(key) != workspace_git_after.get(key) for key in git_keys):
            draft.problems.append(_problem("workspace_git_mutated", "workspace", "workspace Git facts changed during capture", blocking=True))
        for path, expected_digest in engine_surface_digests.items():
            current = inspect_file(self.engine_root, path)
            if current.status != "observed" or current.digest != expected_digest:
                draft.problems.append(_problem("engine_surface_mutated", "engine", path, blocking=True))
        if root_identity(self.root) != original_root:
            draft.blocked = True
            draft.problems.append(_problem("workspace_root_drift", "workspace", "workspace root identity changed", blocking=True))
        if not observer.supported:
            draft.problems.append(_problem("source_observer_unavailable", "capture", "transient source mutation observer is unavailable", blocking=True))
        if observer.mutated():
            draft.problems.append(_problem("source_mutation_observed", "capture", "source namespace changed during observation", blocking=True))
        if data_observer is not None and (not data_observer.supported or data_observer.mutated()):
            draft.problems.append(_problem("data_mutation_observed", "data", "Qlib source continuity is unavailable or changed", blocking=True))
        # Re-observe every explicit source. Digest disagreement is fail-closed.
        for name, path, _required in request.source_paths():
            if not path:
                continue
            previous = snapshots.get(name)
            if name == "deep_analysis" and contained_path(self.root, path).is_dir():
                current_tree = inspect_tree(self.root, path)
                current_inventory = [
                    {"path": item.logical_path, "digest": item.digest.to_dict()}
                    for item in current_tree if item.digest is not None
                ]
                current = FileSnapshot(
                    path, "observed" if current_tree and all(item.status == "observed" for item in current_tree) else "incomparable",
                    None, TypedDigest.canonical(current_inventory, "file_inventory"), None,
                )
            else:
                current = inspect_file(self.root, path)
            if previous is None or previous.digest != current.digest or current.status != "observed":
                draft.problems.append(_problem("source_continuity_lost", name, "source changed during capture", blocking=True))
        core = {
            "schema_version": SCHEMA_VERSION,
            "cycle_identity": {
                "cycle_id": request.cycle_id,
                "research_epoch_id": request.research_epoch_id,
                "evidence_as_of": anchor,
            },
            "engine_identity": {**engine_git, "executable_surface": engine_surface},
            "workspace_identity": workspace_git,
            "data_identity": data_identity,
            "run_evidence": draft.manifest.get("run_evidence", []),
            "model_and_ensemble_lineage": {
                **ranking, "source_artifacts": lineage_artifacts,
            },
            "ranking": ranking,
            "portfolio_state": portfolio,
            "decision_state": decision,
            "referenced_evidence": referenced,
            "preservation": {
                "embedded_object_count": len(draft.objects),
                "named_file_count": len(draft.named_files),
            },
            "problems": draft.problems,
        }
        # Publication itself changes the workspace Git inventory.  Replay
        # identity therefore joins every observed cycle fact except that
        # self-referential repository inventory; its original observation is
        # still sealed in the manifest.
        replay_core = {key: value for key, value in core.items() if key != "workspace_identity"}
        content_digest = TypedDigest.canonical(replay_core)
        complete = not any(item["blocks_complete"] for item in draft.problems)
        core["capture_time"] = self.clock().isoformat()
        core["status"] = "sealed_complete" if complete else "sealed_partial"
        core["request_content_digest"] = content_digest.to_dict()
        draft.manifest = core
        return draft

    def _sources_continuous(self, request: CaptureRequest, draft: _BundleDraft) -> bool:
        initial = draft.source_digests or {}
        for name, path, _required in request.source_paths():
            if not path:
                continue
            if name == "deep_analysis" and contained_path(self.root, path).is_dir():
                tree = inspect_tree(self.root, path)
                inventory = [
                    {"path": item.logical_path, "digest": item.digest.to_dict()}
                    for item in tree if item.digest is not None
                ]
                current = TypedDigest.canonical(inventory, "file_inventory")
                if not tree or any(item.status != "observed" for item in tree):
                    return False
            else:
                snapshot = inspect_file(self.root, path)
                if snapshot.status != "observed":
                    return False
                current = snapshot.digest
            if current != initial.get(name):
                return False
        return True

    def _existing(self, final: Path, cycle_id: str, request_digest: TypedDigest) -> Optional[CaptureResult]:
        if not final.exists():
            return None
        try:
            final_before = os.lstat(str(final))
            if os.path.islink(str(final)) or not final.is_dir():
                raise ContractError("existing final public name is not a canonical directory")
            manifest_path = final / "manifest.json"
            seal_path = final / "seal.json"
            for public_file in (manifest_path, seal_path):
                info = os.lstat(str(public_file))
                if public_file.is_symlink() or not public_file.is_file() or info.st_nlink != 1:
                    raise ContractError("existing seal member is not a canonical regular file")
            manifest_data = manifest_path.read_bytes()
            seal_data = seal_path.read_bytes()
            manifest = json.loads(manifest_data.decode("utf-8"))
            seal = json.loads(seal_data.decode("utf-8"))
            expected_seal_fields = {
                "schema_version", "cycle_id", "status", "manifest_digest",
                "artifact_root_digest", "object_digests", "named_file_digests",
            }
            if not isinstance(manifest, dict) or not isinstance(seal, dict) or set(seal) != expected_seal_fields:
                raise ContractError("existing seal representation is invalid")
            if (
                seal.get("schema_version") != SCHEMA_VERSION
                or seal.get("cycle_id") != cycle_id
                or manifest.get("cycle_identity", {}).get("cycle_id") != cycle_id
                or seal.get("status") != manifest.get("status")
            ):
                raise ContractError("existing cycle/seal identity is inconsistent")
            if manifest.get("request_content_digest") != request_digest.to_dict():
                return _result(cycle_id, "conflict", False, None, None)
            manifest_digest = TypedDigest(**seal["manifest_digest"])
            if manifest_digest != TypedDigest.raw(manifest_data):
                raise ContractError("existing manifest digest is invalid")
            object_digests = seal["object_digests"]
            named_digests = seal["named_file_digests"]
            if (
                not isinstance(object_digests, list)
                or object_digests != sorted(set(object_digests))
                or not isinstance(named_digests, dict)
                or any("/" in name or name in {"manifest.json", "seal.json", "objects"} for name in named_digests)
            ):
                raise ContractError("existing artifact inventory is invalid")
            objects_root = final / "objects"
            actual_objects = []
            if objects_root.exists():
                if objects_root.is_symlink() or not objects_root.is_dir():
                    raise ContractError("existing object root is not canonical")
                for path in objects_root.rglob("*"):
                    if path.is_symlink():
                        raise ContractError("existing object inventory contains a symlink")
                    if path.is_file():
                        if path.stat().st_nlink != 1 or path.parent.name != path.name[:2]:
                            raise ContractError("existing object public name is invalid")
                        actual_objects.append(path.name)
            if sorted(actual_objects) != object_digests:
                raise ContractError("existing object inventory cardinality differs")
            for digest in object_digests:
                path = final / "objects" / digest[:2] / digest
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ContractError("existing embedded object is invalid")
            expected_top = {"manifest.json", "seal.json", *named_digests}
            if object_digests:
                expected_top.add("objects")
            actual_top = {path.name for path in final.iterdir()}
            if actual_top != expected_top or any(path.is_symlink() for path in final.iterdir()):
                raise ContractError("existing bundle has an unsealed top-level member")
            for name, digest in named_digests.items():
                typed = TypedDigest(**digest)
                path = final / name
                if not path.is_file() or path.stat().st_nlink != 1 or TypedDigest.raw(path.read_bytes()) != typed:
                    raise ContractError("existing named evidence is invalid")
            artifact_root = TypedDigest.canonical({
                "objects": object_digests, "named_files": named_digests,
            })
            if TypedDigest(**seal["artifact_root_digest"]) != artifact_root:
                raise ContractError("existing artifact root digest is invalid")
            final_after = os.lstat(str(final))
            if (final_before.st_dev, final_before.st_ino) != (final_after.st_dev, final_after.st_ino):
                raise ContractError("existing final identity changed during verification")
            status = manifest.get("status")
            if status not in {"sealed_complete", "sealed_partial"}:
                raise ContractError("existing bundle status is invalid")
            return _result(
                manifest["cycle_identity"]["cycle_id"], "adopted", False,
                final.relative_to(self.root).as_posix(), TypedDigest.raw(seal_data),
                tuple(manifest.get("problems", [])),
                sealed_status=status,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _result(cycle_id, "conflict", False, None, None)

    def capture(self, request: CaptureRequest, *, dry_run: bool = False) -> CaptureResult:
        paths = [path for _name, path, _required in request.source_paths() if path]
        try:
            configured = self.qlib_data_dir
            if configured is None and os.environ.get("QLIB_DATA_DIR"):
                configured = Path(os.environ["QLIB_DATA_DIR"]).expanduser().resolve()
            with ExitStack() as stack:
                observer = stack.enter_context(SourceMutationObserver(self.root, paths))
                data_observer = None
                if configured is not None and configured.is_dir():
                    data_observer = stack.enter_context(SourceMutationObserver(
                        configured, ("calendars/day.txt", "instruments"),
                    ))
                return self._capture(
                    request, dry_run=dry_run, observer=observer,
                    data_observer=data_observer,
                )
        except PathBoundaryError as exc:
            return _result(request.cycle_id, "blocked", False, None, None, (_problem("path_boundary", "workspace", str(exc), blocking=True),))

    def _capture(
        self, request: CaptureRequest, *, dry_run: bool,
        observer: SourceMutationObserver,
        data_observer: Optional[SourceMutationObserver],
    ) -> CaptureResult:
        root_before = root_identity(self.root)
        try:
            draft = self._build(request, observer, data_observer)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError as exc:
            return _result(request.cycle_id, "blocked", False, None, None, (_problem("path_boundary", "workspace", str(exc), blocking=True),))
        except Exception as exc:
            return _result(request.cycle_id, "blocked", False, None, None, (_problem("inspection_failed", "capture", "%s: %s" % (type(exc).__name__, exc), blocking=True),))
        if draft.blocked:
            return _result(request.cycle_id, "blocked", False, None, None, tuple(draft.problems))
        request_digest = TypedDigest.canonical({
            key: value for key, value in draft.manifest.items()
            if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
        })
        # _build computed this before adding time/status; enforce the internal join.
        if draft.manifest["request_content_digest"] != request_digest.to_dict():
            raise ContractError("inspector content digest join failed")
        evidence_root = self.root / "data" / "evidence" / "v1"
        final = evidence_root / "cycles" / request.cycle_id
        if final.exists():
            existing = self._existing(final, request.cycle_id, request_digest)
            if existing and existing.cycle_id == request.cycle_id:
                return existing
            return _result(request.cycle_id, "conflict", False, None, None, tuple(draft.problems))
        object_digests = sorted(draft.objects)
        named_digests = {
            name: TypedDigest.raw(data).to_dict()
            for name, data in sorted(draft.named_files.items())
        }
        artifact_root = TypedDigest.canonical({
            "objects": object_digests, "named_files": named_digests,
        })
        manifest_data = canonical_json_bytes(draft.manifest)
        manifest_digest = TypedDigest.raw(manifest_data)
        seal = {
            "schema_version": SCHEMA_VERSION,
            "cycle_id": request.cycle_id,
            "status": draft.manifest["status"],
            "manifest_digest": manifest_digest.to_dict(),
            "artifact_root_digest": artifact_root.to_dict(),
            "object_digests": object_digests,
            "named_file_digests": named_digests,
        }
        seal_data = canonical_json_bytes(seal)
        if dry_run:
            return _result(
                request.cycle_id, draft.manifest["status"], False,
                final.relative_to(self.root).as_posix(), TypedDigest.raw(seal_data),
                tuple(draft.problems),
            )
        lock = evidence_root / ".locks" / (request.cycle_id + ".lock")
        stage = None
        stage_identity = None
        lock_fd = None
        lock_owned = False
        wrote_staging = False
        published = False
        try:
            _safe_mkdirs(self.root, lock.parent)
            _safe_mkdirs(self.root, evidence_root / "cycles")
            lock_parent_fd = os.open(str(lock.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
            lock_fd = os.open(lock.name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=lock_parent_fd)
            lock_owned = True
            os.write(lock_fd, request_digest.value.encode("ascii"))
            os.fsync(lock_fd)
            os.fsync(lock_parent_fd)
            self.fault_hook("after_lock")
            if final.exists():
                existing = self._existing(final, request.cycle_id, request_digest)
                if existing and existing.cycle_id == request.cycle_id:
                    return existing
                return _result(request.cycle_id, "conflict", False, None, None, tuple(draft.problems))
            if root_identity(self.root) != root_before:
                return _result(request.cycle_id, "blocked", False, None, None, tuple(draft.problems))
            parent_identity = root_identity(final.parent)
            _safe_mkdirs(self.root, evidence_root / ".staging")
            stage = Path(tempfile.mkdtemp(prefix=request.cycle_id + ".", dir=str(evidence_root / ".staging")))
            stage_info = os.lstat(str(stage))
            stage_identity = (stage_info.st_dev, stage_info.st_ino)
            wrote_staging = True
            for digest, data in draft.objects.items():
                _atomic_bytes(stage / "objects" / digest[:2] / digest, data)
            for name, data in draft.named_files.items():
                _atomic_bytes(stage / name, data)
            _atomic_bytes(stage / "manifest.json", manifest_data)
            _atomic_bytes(stage / "seal.json", seal_data)
            _fsync_tree_directories(stage)
            self.fault_hook("before_publish")
            if (
                observer.mutated()
                or (data_observer is not None and data_observer.mutated())
                or not self._sources_continuous(request, draft)
            ):
                return _result(
                    request.cycle_id, "failed_no_final", True, None, None,
                    tuple(draft.problems + [_problem(
                        "source_continuity_lost", "capture",
                        "source changed before namespace publication", blocking=True,
                    )]),
                )
            if root_identity(self.root) != root_before or root_identity(final.parent) != parent_identity:
                return _result(request.cycle_id, "blocked", False, None, None, tuple(draft.problems))
            _rename_noreplace(stage, final)
            published = True
            stage = None
            _fsync_dir(final.parent)
            self.fault_hook("after_publish")
            try:
                final_info = os.lstat(str(final))
                final_continuous = (
                    not os.path.islink(str(final))
                    and os.path.isdir(str(final))
                    and stage_identity == (final_info.st_dev, final_info.st_ino)
                )
            except OSError:
                final_continuous = False
            if (
                not final_continuous
                or root_identity(self.root) != root_before
                or root_identity(final.parent) != parent_identity
            ):
                return _result(request.cycle_id, "uncertain", True, None, None, tuple(draft.problems))
            adopted = self._existing(final, request.cycle_id, request_digest)
            if adopted is None or adopted.cycle_id != request.cycle_id:
                return _result(request.cycle_id, "uncertain", True, None, None, tuple(draft.problems))
            final_info_after = os.lstat(str(final))
            if stage_identity != (final_info_after.st_dev, final_info_after.st_ino):
                return _result(request.cycle_id, "uncertain", True, None, None, tuple(draft.problems))
            return _result(
                request.cycle_id, draft.manifest["status"], True,
                final.relative_to(self.root).as_posix(), TypedDigest.raw(seal_data),
                tuple(draft.problems),
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError as exc:
            status = "failed_no_final" if wrote_staging else "blocked"
            return _result(
                request.cycle_id, status, wrote_staging, None, None,
                tuple(draft.problems + [_problem("write_boundary", "publication", str(exc), blocking=True)]),
            )
        except FileExistsError:
            status = "conflict" if lock_owned else "blocked"
            return _result(request.cycle_id, status, False, None, None, tuple(draft.problems))
        except Exception as exc:
            status = "uncertain" if published else "failed_no_final"
            return _result(
                request.cycle_id, status, published or wrote_staging, None, None,
                tuple(draft.problems + [_problem("publication_failed", "publication", "%s: %s" % (type(exc).__name__, exc), blocking=True)]),
            )
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            if stage is not None:
                try:
                    current = os.lstat(str(stage))
                    stage.resolve(strict=True).relative_to(self.root)
                    if stage_identity == (current.st_dev, current.st_ino):
                        stage_parent = stage.parent
                        shutil.rmtree(str(stage))
                        _fsync_dir(stage_parent)
                except (FileNotFoundError, ValueError, OSError):
                    pass
            try:
                if 'lock_parent_fd' in locals():
                    try:
                        if lock_owned:
                            os.unlink(lock.name, dir_fd=lock_parent_fd)
                            os.fsync(lock_parent_fd)
                    except FileNotFoundError:
                        pass
                    finally:
                        os.close(lock_parent_fd)
            except OSError:
                pass
