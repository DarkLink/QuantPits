"""Pure Champion--Challenger ranking replay and strict read-only adapters.

Stage A deliberately has no portfolio or publication capability.  The only
durable adapter writes a complete, disposable result tree below ``/tmp``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
import struct
from dataclasses import InitVar, dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.evidence.ranking import RankingResult, canonical_full_ranking


WARNING_LINES = (
    "RETROSPECTIVE TECHNICAL REPLAY",
    "Not prospective evidence.",
    "Not an automatic promotion basis.",
    "Current-rule replay may differ from historical Production behavior.",
)
FUSION_DEFINITION = "cross_sectional_percentile_rank_average_ties_then_equal_mean_v1"
PARITY_SCORE_TOLERANCE = 1e-12
_OBSERVATION_AUTHORITY = object()
_RESULT_AUTHORITY = object()


class ReplayContractError(ValueError):
    """A caller-supplied replay representation is invalid."""


class ReplayInputError(RuntimeError):
    """A sealed or provider input cannot be observed exactly."""

    def __init__(self, message: str, evidence: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(message)
        self.evidence = tuple(MappingProxyType(dict(item)) for item in evidence)


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReplayContractError("%s must be non-empty canonical text" % name)
    return value


def _strict_date(value: Any, name: str) -> str:
    value = _strict_text(value, name)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ReplayContractError("%s must be YYYY-MM-DD" % name) from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise ReplayContractError("%s must be a canonical date" % name)
    return value


def _raw_digest(data: bytes) -> dict:
    return {
        "algorithm": "sha256",
        "domain": "raw_bytes",
        "value": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _canonical_digest(value: Any) -> dict:
    data = canonical_json_bytes(value)
    return {
        "algorithm": "sha256",
        "domain": "canonical_json",
        "value": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _validate_digest(value: Any, *, domain: str, name: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "algorithm", "domain", "value", "size_bytes",
    }:
        raise ReplayInputError("%s digest fields are not exact" % name)
    if (
        value.get("algorithm") != "sha256" or value.get("domain") != domain
        or not isinstance(value.get("value"), str)
        or len(value["value"]) != 64
        or any(char not in "0123456789abcdef" for char in value["value"])
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] < 0
    ):
        raise ReplayInputError("%s digest is invalid" % name)
    return dict(value)


def _relative_file(root: Path, value: Any, name: str) -> Tuple[str, Path]:
    logical = _strict_text(value, name)
    if "\\" in logical or "\0" in logical:
        raise ReplayInputError("%s is not a canonical relative path" % name)
    relative = Path(logical)
    if relative.is_absolute() or relative.as_posix() != logical or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ReplayInputError("%s is not a canonical relative path" % name)
    canonical_root = root.resolve(strict=True)
    candidate = canonical_root.joinpath(*relative.parts)
    current = canonical_root
    for part in relative.parts:
        current = current / part
        info = os.lstat(str(current))
        if stat.S_ISLNK(info.st_mode):
            raise ReplayInputError("%s contains a symlink alias" % name)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(canonical_root)
    except ValueError as exc:
        raise ReplayInputError("%s escapes its authority root" % name) from exc
    return logical, resolved


def _stable_read(root: Path, logical: str, expected: Optional[Mapping[str, Any]] = None) -> bytes:
    logical, path = _relative_file(root, logical, "source path")
    before = os.lstat(str(path))
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1:
        raise ReplayInputError("source path is not an exclusive regular file")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        data = handle.read()
        after_open = os.fstat(handle.fileno())
    after = os.lstat(str(path))
    identities = [
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after_open, after)
    ]
    if len(set(identities)) != 1:
        raise ReplayInputError("source identity changed during read")
    actual = _raw_digest(data)
    if expected is not None:
        expected = _validate_digest(expected, domain="raw_bytes", name=logical)
        if actual != expected:
            raise ReplayInputError("source bytes differ from frozen digest")
    return data


def _json_object(data: bytes, name: str) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite")),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReplayInputError("%s is not strict JSON" % name) from exc
    if not isinstance(value, dict):
        raise ReplayInputError("%s must be a JSON object" % name)
    return value


def _prediction_frame(data: bytes, model_name: str) -> Mapping[str, Tuple[Tuple[str, Any], ...]]:
    try:
        value = pd.read_pickle(io.BytesIO(data))
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        raise ReplayInputError("prediction safe decode failed: %s" % type(exc).__name__) from exc
    if isinstance(value, pd.Series):
        value = value.to_frame("score")
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ReplayInputError("prediction must be a non-empty frame")
    if list(value.columns) == ["score"]:
        result = value.copy()
    elif len(value.columns) > 1 and list(value.columns).count("score") == 1:
        result = value[["score"]].copy()
    else:
        raise ReplayInputError("prediction must contain one unique score column")
    if not isinstance(result.index, pd.MultiIndex) or set(result.index.names) != {
        "datetime", "instrument",
    } or len(result.index.names) != 2:
        raise ReplayInputError("prediction index must be exact datetime/instrument MultiIndex")
    if result.index.has_duplicates:
        raise ReplayInputError("prediction contains duplicate datetime/instrument rows")
    dates = pd.to_datetime(result.index.get_level_values("datetime"), errors="raise")
    if getattr(dates, "tz", None) is not None or any(item != item.normalize() for item in dates):
        raise ReplayInputError("prediction dates must be timezone-naive midnight sessions")
    instruments = result.index.get_level_values("instrument")
    if any(not isinstance(item, str) or not item or item.strip() != item for item in instruments):
        raise ReplayInputError("prediction instruments must be canonical text")
    result.index = pd.MultiIndex.from_arrays(
        [dates, instruments], names=["datetime", "instrument"],
    )
    result.columns = [model_name]
    result = result.sort_index()
    by_date = {}
    for date, group in result.groupby(level="datetime", sort=True):
        selected = group.droplevel("datetime")
        by_date[pd.Timestamp(date).strftime("%Y-%m-%d")] = tuple(
            (instrument, value) for instrument, value in selected[model_name].items()
        )
    return MappingProxyType(by_date)


def _universe_intervals(data: bytes) -> Tuple[Tuple[str, str, str], ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayInputError("universe is not UTF-8") from exc
    rows = []
    for raw in text.splitlines():
        fields = raw.strip().replace(",", "\t").split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ReplayInputError("universe row must contain instrument/start/end")
        instrument = _strict_text(fields[0], "universe instrument")
        start = _strict_date(fields[1], "universe start")
        end = _strict_date(fields[2], "universe end")
        if start > end:
            raise ReplayInputError("universe interval is inverted")
        rows.append((instrument, start, end))
    if not rows:
        raise ReplayInputError("universe inventory is empty")
    return tuple(rows)


def _universe_at(intervals: Sequence[Tuple[str, str, str]], anchor: str) -> Tuple[str, ...]:
    members = [instrument for instrument, start, end in intervals if start <= anchor <= end]
    if not members or len(members) != len(set(members)):
        raise ReplayInputError("eligible universe is empty or has overlapping identities")
    return tuple(members)


@dataclass(frozen=True)
class ReplaySource:
    model_name: str
    recorder_id: str
    experiment_name: str
    artifact_path: str
    prediction_digest: Mapping[str, Any]
    prediction: Mapping[str, Tuple[Tuple[str, Any], ...]]
    _authority: InitVar[object] = None

    def __post_init__(self, _authority: object) -> None:
        if _authority is not _OBSERVATION_AUTHORITY:
            raise ReplayContractError("source observations are inspector-owned")
        for name in ("model_name", "recorder_id", "experiment_name", "artifact_path"):
            _strict_text(getattr(self, name), name)
        _validate_digest(self.prediction_digest, domain="raw_bytes", name="prediction")
        if not isinstance(self.prediction, Mapping) or not self.prediction:
            raise ReplayContractError("prediction observation must be a non-empty mapping")
        for anchor, rows in self.prediction.items():
            _strict_date(anchor, "prediction anchor")
            if not isinstance(rows, tuple) or any(
                not isinstance(row, tuple) or len(row) != 2 for row in rows
            ):
                raise ReplayContractError("prediction rows must be exact tuples")
        object.__setattr__(self, "prediction_digest", MappingProxyType(dict(self.prediction_digest)))

    def public_identity(self) -> dict:
        return {
            "model_name": self.model_name,
            "recorder_id": self.recorder_id,
            "experiment_name": self.experiment_name,
            "artifact_path": self.artifact_path,
            "prediction_digest": dict(self.prediction_digest),
        }


@dataclass(frozen=True)
class SealedReplayInputs:
    parity_anchor: str
    sources: Tuple[ReplaySource, ...]
    universe_name: str
    universe_digest: Mapping[str, Any]
    calendar_digest: Mapping[str, Any]
    universe_intervals: Tuple[Tuple[str, str, str], ...]
    calendar_dates: Tuple[str, ...]
    sealed_ranking: RankingResult
    sealed_ranking_digest: Mapping[str, Any]
    source_to_materialization_relation: str
    price_coverage: Mapping[str, Mapping[str, Any]]
    price_source_inventory: Tuple[Mapping[str, Any], ...]
    price_source_inventory_digest: Optional[Mapping[str, Any]]
    _authority: InitVar[object] = None

    def __post_init__(self, _authority: object) -> None:
        if _authority is not _OBSERVATION_AUTHORITY:
            raise ReplayContractError("sealed replay inputs are inspector-owned")
        _strict_date(self.parity_anchor, "parity_anchor")
        if not isinstance(self.sources, tuple) or len(self.sources) != 4:
            raise ReplayContractError("Stage A requires exactly four Champion sources")
        if len({item.model_name for item in self.sources}) != len(self.sources):
            raise ReplayContractError("source model identities must be unique")
        _strict_text(self.universe_name, "universe_name")
        _validate_digest(self.universe_digest, domain="raw_bytes", name="universe")
        _validate_digest(self.calendar_digest, domain="raw_bytes", name="calendar")
        _validate_digest(self.sealed_ranking_digest, domain="raw_bytes", name="sealed ranking")
        if self.source_to_materialization_relation != "unverified":
            raise ReplayContractError("source-to-materialization relation must remain unverified")
        if not isinstance(self.price_coverage, Mapping):
            raise ReplayContractError("price coverage must be an observed mapping")
        if not isinstance(self.price_source_inventory, tuple):
            raise ReplayContractError("price source inventory must be an exact tuple")
        canonical_price_inventory = []
        identities = []
        for item in self.price_source_inventory:
            if not isinstance(item, Mapping) or set(item) != {
                "instrument", "field", "logical_path", "status", "digest",
            }:
                raise ReplayContractError("price source inventory member fields are not exact")
            instrument = _strict_text(item["instrument"], "price instrument")
            field = item["field"]
            if field not in {"close", "open"}:
                raise ReplayContractError("price field must be close or open")
            expected_path = "features/%s/%s.day.bin" % (instrument.lower(), field)
            if item["logical_path"] != expected_path:
                raise ReplayContractError("price source logical path is not canonical")
            if item["status"] == "observed":
                digest = _validate_digest(
                    item["digest"], domain="raw_bytes", name="price source",
                )
            elif item["status"] == "missing" and item["digest"] is None:
                digest = None
            else:
                raise ReplayContractError("price source status and digest are inconsistent")
            identities.append((instrument, field))
            canonical_price_inventory.append({
                "instrument": instrument,
                "field": field,
                "logical_path": expected_path,
                "status": item["status"],
                "digest": digest,
            })
        if len(identities) != len(set(identities)) or identities != sorted(identities):
            raise ReplayContractError("price source inventory identities must be unique and ordered")
        if canonical_price_inventory:
            observed_digest = _validate_digest(
                self.price_source_inventory_digest,
                domain="canonical_json", name="price source inventory",
            )
            if observed_digest != _canonical_digest(canonical_price_inventory):
                raise ReplayContractError("price source inventory digest is inconsistent")
        elif self.price_source_inventory_digest is not None:
            raise ReplayContractError("empty price source inventory cannot carry a digest")
        if not isinstance(self.universe_intervals, tuple) or not isinstance(self.calendar_dates, tuple):
            raise ReplayContractError("provider inventories must be exact tuples")
        if not self.calendar_dates or tuple(sorted(set(self.calendar_dates))) != self.calendar_dates:
            raise ReplayContractError("calendar dates must be non-empty, unique, and increasing")
        if self.parity_anchor not in self.calendar_dates:
            raise ReplayContractError("parity anchor is absent from calendar")
        object.__setattr__(self, "universe_digest", MappingProxyType(dict(self.universe_digest)))
        object.__setattr__(self, "calendar_digest", MappingProxyType(dict(self.calendar_digest)))
        object.__setattr__(self, "sealed_ranking_digest", MappingProxyType(dict(self.sealed_ranking_digest)))
        object.__setattr__(self, "price_coverage", MappingProxyType({
            key: MappingProxyType(dict(value)) for key, value in self.price_coverage.items()
        }))
        object.__setattr__(self, "price_source_inventory", tuple(
            MappingProxyType(item) for item in canonical_price_inventory
        ))
        if self.price_source_inventory_digest is not None:
            object.__setattr__(
                self, "price_source_inventory_digest",
                MappingProxyType(dict(self.price_source_inventory_digest)),
            )


def _ranking_from_csv(data: bytes) -> RankingResult:
    try:
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ReplayInputError("sealed ranking CSV is invalid") from exc
    expected = [
        "instrument", "eligible", "scored", "coverage_status",
        "raw_score", "rank", "normalized_score",
    ]
    if not rows or list(rows[0]) != expected:
        raise ReplayInputError("sealed ranking fields are not exact")
    rebuilt = []
    for row in rows:
        if set(row) != set(expected) or row["eligible"] not in {"True", "False"} or row["scored"] not in {"True", "False"}:
            raise ReplayInputError("sealed ranking row is invalid")
        scored = row["scored"] == "True"
        rebuilt.append({
            **row,
            "eligible": row["eligible"] == "True",
            "scored": scored,
            "rank": int(row["rank"]) if scored else "",
        })
    scored_count = sum(item["scored"] for item in rebuilt)
    return RankingResult(
        tuple(rebuilt), len(rebuilt), scored_count, len(rebuilt) - scored_count,
        scored_count == len(rebuilt),
    )


def _feature_value(data: Optional[bytes], calendar_position: int) -> Optional[float]:
    if data is None or len(data) < 8 or len(data) % 4:
        return None
    start = struct.unpack_from("<f", data, 0)[0]
    if not math.isfinite(start) or int(start) != start:
        return None
    offset = (calendar_position - int(start) + 1) * 4
    if offset < 4 or offset + 4 > len(data):
        return None
    value = struct.unpack_from("<f", data, offset)[0]
    return float(value) if math.isfinite(value) else None


def _observe_price_coverage(
    provider: Path,
    intervals: Sequence[Tuple[str, str, str]],
    calendar_dates: Sequence[str],
    start: str,
    end: str,
) -> Tuple[Mapping[str, Mapping[str, Any]], Tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    selected = [date for date in calendar_dates if start <= date <= end]
    calendar_position = {date: position for position, date in enumerate(calendar_dates)}
    instruments = sorted({
        instrument for date in selected for instrument in _universe_at(intervals, date)
    })
    feature_data = {}
    inventory = []
    for instrument in instruments:
        for field in ("close", "open"):
            logical = "features/%s/%s.day.bin" % (instrument.lower(), field)
            try:
                data = _stable_read(provider, logical)
                status = "observed"
                digest = _raw_digest(data)
            except FileNotFoundError:
                data = None
                status = "missing"
                digest = None
            feature_data[(instrument, field)] = data
            inventory.append({
                "instrument": instrument, "field": field,
                "logical_path": logical,
                "status": status, "digest": digest,
            })
    coverage = {}
    for anchor in selected:
        position = calendar_position[anchor]
        next_session = calendar_dates[position + 1] if position + 1 < len(calendar_dates) else None
        universe = _universe_at(intervals, anchor)
        missing_close = [
            instrument for instrument in universe
            if _feature_value(feature_data[(instrument, "close")], position) is None
        ]
        missing_open = [
            instrument for instrument in universe
            if next_session is None
            or _feature_value(feature_data[(instrument, "open")], position + 1) is None
        ]
        coverage[anchor] = {
            "anchor_close_status": "complete" if not missing_close else "partial",
            "anchor_close_finite_count": len(universe) - len(missing_close),
            "anchor_close_missing": missing_close,
            "next_session": next_session,
            "next_open_status": "complete" if not missing_open else "partial",
            "next_open_finite_count": len(universe) - len(missing_open),
            "next_open_missing": missing_open,
        }
    return MappingProxyType({
        key: MappingProxyType(value) for key, value in coverage.items()
    }), tuple(inventory), _canonical_digest(inventory)


def load_sealed_replay_inputs(
    workspace_root: Path, sealed_cycle: str, *, qlib_data_dir: Optional[Path] = None,
    coverage_start: Optional[str] = None, coverage_end: Optional[str] = None,
) -> SealedReplayInputs:
    """Observe exact Stage-A inputs without initializing Qlib or MLflow."""

    workspace = Path(workspace_root).resolve(strict=True)
    cycle = _strict_date(sealed_cycle, "sealed_cycle")
    cycle_root = "data/evidence/v1/cycles/%s" % cycle
    manifest_bytes = _stable_read(workspace, cycle_root + "/manifest.json")
    seal_bytes = _stable_read(workspace, cycle_root + "/seal.json")
    manifest = _json_object(manifest_bytes, "manifest.json")
    seal = _json_object(seal_bytes, "seal.json")
    if seal.get("cycle_id") != cycle or manifest.get("cycle_identity", {}).get("cycle_id") != cycle:
        raise ReplayInputError("sealed cycle identity mismatch")
    if _raw_digest(manifest_bytes) != _validate_digest(
        seal.get("manifest_digest"), domain="raw_bytes", name="manifest",
    ):
        raise ReplayInputError("manifest bytes differ from seal")
    named = seal.get("named_file_digests")
    if not isinstance(named, Mapping) or "ranking.csv" not in named:
        raise ReplayInputError("seal lacks ranking.csv digest")
    ranking_bytes = _stable_read(
        workspace, cycle_root + "/ranking.csv",
        _validate_digest(named["ranking.csv"], domain="raw_bytes", name="ranking.csv"),
    )
    lineage = manifest.get("model_and_ensemble_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("status") != "complete":
        raise ReplayInputError("sealed Champion lineage is not complete")
    models = lineage.get("source_models")
    artifacts = lineage.get("source_artifacts")
    combo = lineage.get("combo")
    if not isinstance(models, list) or not isinstance(artifacts, list) or not isinstance(combo, Mapping):
        raise ReplayInputError("sealed source inventory is invalid")
    resolved = combo.get("resolved_members")
    if not isinstance(resolved, list) or len(models) != len(resolved):
        raise ReplayInputError("sealed source identity/cardinality is inconsistent")
    by_recorder = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or artifact.get("recorder_id") in by_recorder:
            raise ReplayInputError("source artifact inventory is malformed")
        by_recorder[artifact.get("recorder_id")] = artifact
    sources = []
    source_evidence = []
    for position, model_name in enumerate(resolved):
        evidence = {"position": position, "model_name": model_name, "status": "failed"}
        try:
            model = models[position]
            if not isinstance(model, Mapping) or model.get("resolved_key") != model_name or model.get("status") != "ready":
                raise ReplayInputError("source model ordering or status is invalid")
            recorder_id = _strict_text(model.get("recorder_id"), "recorder_id")
            evidence["recorder_id"] = recorder_id
            artifact = by_recorder.get(recorder_id)
            if artifact is None:
                raise ReplayInputError("source model lacks matching artifact observation")
            pred_members = [
                item for item in artifact.get("members", [])
                if isinstance(item, Mapping) and str(item.get("path", "")).endswith("/pred.pkl")
            ]
            if len(pred_members) != 1:
                raise ReplayInputError("source artifact must contain one pred.pkl observation")
            member = pred_members[0]
            artifact_path = _strict_text(model.get("artifact_path"), "artifact_path")
            if Path(member.get("path", "")).parent.as_posix() != artifact_path:
                raise ReplayInputError("prediction public path differs from frozen artifact root")
            prediction_bytes = _stable_read(
                workspace, member.get("path"),
                _validate_digest(member.get("digest"), domain="raw_bytes", name="pred.pkl"),
            )
            sources.append(ReplaySource(
                model_name=_strict_text(model_name, "model_name"),
                recorder_id=recorder_id,
                experiment_name=_strict_text(model.get("experiment_name"), "experiment_name"),
                artifact_path=artifact_path,
                prediction_digest=_raw_digest(prediction_bytes),
                prediction=_prediction_frame(prediction_bytes, model_name),
                _authority=_OBSERVATION_AUTHORITY,
            ))
            evidence["status"] = "ready"
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            evidence["error_type"] = type(exc).__name__
            evidence["detail"] = str(exc)
        source_evidence.append(evidence)
    if any(item["status"] != "ready" for item in source_evidence):
        raise ReplayInputError("one or more requested source observations failed", source_evidence)
    data_root = manifest.get("data_identity")
    if not isinstance(data_root, Mapping):
        raise ReplayInputError("sealed data identity is invalid")
    source_relation = data_root.get("source_to_materialization_relation")
    if source_relation != "unverified":
        raise ReplayInputError("sealed source-to-materialization relation must be unverified")
    data_identity = data_root.get("qlib_materialization_identity", {})
    if data_identity.get("status") != "observed" or data_identity.get("calendar_cutoff") != cycle:
        raise ReplayInputError("sealed Qlib materialization is not comparable")
    universe_name = _strict_text(data_identity.get("universe_name"), "universe_name")
    provider = Path(qlib_data_dir or os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data")).expanduser().resolve(strict=True)
    universe_logical = "instruments/%s.txt" % universe_name
    calendar_logical = "calendars/day.txt"
    universe_digest = _validate_digest(data_identity.get("universe_digest"), domain="raw_bytes", name="universe")
    calendar_digest = _validate_digest(data_identity.get("calendar_digest"), domain="raw_bytes", name="calendar")
    universe_bytes = _stable_read(provider, universe_logical, universe_digest)
    calendar_bytes = _stable_read(provider, calendar_logical, calendar_digest)
    try:
        calendar_dates = tuple(
            _strict_date(row.strip()[:10], "calendar date")
            for row in calendar_bytes.decode("utf-8").splitlines() if row.strip()
        )
    except UnicodeDecodeError as exc:
        raise ReplayInputError("calendar is not UTF-8") from exc
    sealed_ranking = _ranking_from_csv(ranking_bytes)
    intervals = _universe_intervals(universe_bytes)
    exact_universe = _universe_at(intervals, cycle)
    if {item["instrument"] for item in sealed_ranking.rows} != set(exact_universe):
        raise ReplayInputError("sealed ranking does not match exact eligible universe")
    if (coverage_start is None) != (coverage_end is None):
        raise ReplayContractError("price coverage start and end must be supplied together")
    price_coverage = {}
    price_inventory = ()
    price_inventory_digest = None
    if coverage_start is not None and coverage_end is not None:
        observed_start = _strict_date(coverage_start, "coverage_start")
        observed_end = _strict_date(coverage_end, "coverage_end")
        if observed_start > observed_end:
            raise ReplayContractError("price coverage window is inverted")
        price_coverage, price_inventory, price_inventory_digest = _observe_price_coverage(
            provider, intervals, calendar_dates, observed_start, observed_end,
        )
    return SealedReplayInputs(
        parity_anchor=cycle,
        sources=tuple(sources),
        universe_name=universe_name,
        universe_digest=universe_digest,
        calendar_digest=calendar_digest,
        universe_intervals=intervals,
        calendar_dates=calendar_dates,
        sealed_ranking=sealed_ranking,
        sealed_ranking_digest=_raw_digest(ranking_bytes),
        source_to_materialization_relation=source_relation,
        price_coverage=price_coverage,
        price_source_inventory=price_inventory,
        price_source_inventory_digest=price_inventory_digest,
        _authority=_OBSERVATION_AUTHORITY,
    )


def _date_scores(
    prediction: Mapping[str, Tuple[Tuple[str, Any], ...]], model_name: str, anchor: str,
) -> Tuple[Dict[str, Any], int]:
    del model_name  # identity is already bound by ReplaySource
    rows = prediction.get(anchor, ())
    return dict(rows), len(rows)


def _rank_percentiles(scores: Mapping[str, float]) -> Dict[str, float]:
    series = pd.Series(scores, dtype=float)
    if len(series) == 1:
        return {str(series.index[0]): 0.5}
    ranks = series.rank(method="average")
    normalized = (ranks - 1.0) / (len(series) - 1.0)
    return {str(key): float(value) for key, value in normalized.items()}


def _spearman(left: RankingResult, right: RankingResult) -> float:
    left_ranks = {row["instrument"]: row["rank"] for row in left.rows if row["scored"]}
    right_ranks = {row["instrument"]: row["rank"] for row in right.rows if row["scored"]}
    if set(left_ranks) != set(right_ranks) or len(left_ranks) < 2:
        raise ReplayContractError("Spearman requires two complete identical member sets")
    keys = sorted(left_ranks)
    x = [float(left_ranks[key]) for key in keys]
    y = [float(right_ranks[key]) for key in keys]
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 1.0


def _top(result: RankingResult, count: int) -> Tuple[str, ...]:
    return tuple(row["instrument"] for row in result.rows if row["scored"] and row["rank"] <= count)


def _compare(champion: RankingResult, challenger: RankingResult, top_k: int) -> dict:
    champ_ranks = {row["instrument"]: row["rank"] for row in champion.rows if row["scored"]}
    arm_ranks = {row["instrument"]: row["rank"] for row in challenger.rows if row["scored"]}
    champ_top = set(_top(champion, top_k))
    arm_top = set(_top(challenger, top_k))
    total = len(champ_ranks)
    champ_bottom = {key for key, rank in champ_ranks.items() if rank > total - top_k}
    arm_bottom = {key for key, rank in arm_ranks.items() if rank > total - top_k}
    migrations = sorted(
        (
            {
                "instrument": key,
                "champion_rank": champ_ranks[key],
                "arm_rank": arm_ranks[key],
                "rank_change": champ_ranks[key] - arm_ranks[key],
                "absolute_rank_change": abs(champ_ranks[key] - arm_ranks[key]),
            }
            for key in champ_ranks
        ),
        key=lambda item: (-item["absolute_rank_change"], item["instrument"]),
    )
    return {
        "full_ranking_spearman": format(_spearman(champion, challenger), ".17g"),
        "top_k": top_k,
        "top_k_overlap_count": len(champ_top & arm_top),
        "top_k_overlap_ratio": format(len(champ_top & arm_top) / float(top_k), ".17g"),
        "top_k_entered": sorted(arm_top - champ_top),
        "top_k_exited": sorted(champ_top - arm_top),
        "bottom_k_overlap_count": len(champ_bottom & arm_bottom),
        "bottom_k_overlap_ratio": format(len(champ_bottom & arm_bottom) / float(top_k), ".17g"),
        "bottom_k_entered": sorted(arm_bottom - champ_bottom),
        "bottom_k_exited": sorted(champ_bottom - arm_bottom),
        "top_migrations": migrations[:top_k],
    }


class ResearchRankingReplay:
    """Sole truth owner for Stage-A inventory, fusion, parity, and comparison."""

    def __init__(self, inputs: SealedReplayInputs, *, top_k: int = 22) -> None:
        if not isinstance(inputs, SealedReplayInputs):
            raise ReplayContractError("inputs must be SealedReplayInputs")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ReplayContractError("top_k must be a positive integer")
        self.inputs = inputs
        self.top_k = top_k

    def _inventory_date(self, anchor: str) -> dict:
        universe = _universe_at(self.inputs.universe_intervals, anchor)
        eligible = set(universe)
        members = []
        complete = True
        for source in self.inputs.sources:
            scores, observed_rows = _date_scores(source.prediction, source.model_name, anchor)
            score_ids = set(scores)
            foreign = sorted(score_ids - eligible)
            missing = sorted(eligible - score_ids)
            non_finite = sorted(
                key for key, value in scores.items() if key in eligible
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            )
            finite_eligible = eligible & score_ids - set(non_finite)
            ready = not foreign and not missing and not non_finite and observed_rows == len(universe)
            complete = complete and ready
            members.append({
                **source.public_identity(),
                "status": "complete" if ready else "incomplete",
                "requested_eligible_count": len(universe),
                "observed_row_count": observed_rows,
                "finite_scored_count": len(finite_eligible),
                "missing_eligible": missing,
                "foreign_instruments": foreign,
                "non_finite_instruments": non_finite,
            })
        return {
            "anchor": anchor,
            "status": "complete" if complete else "incomplete",
            "universe_name": self.inputs.universe_name,
            "universe_digest": dict(self.inputs.universe_digest),
            "eligible_count": len(universe),
            "price_availability": dict(self.inputs.price_coverage.get(anchor, {
                "status": "not_observed",
            })),
            "members": members,
        }

    def _weekly_candidates(self, start: str, end: str) -> Tuple[str, ...]:
        sessions = [date for date in self.inputs.calendar_dates if start <= date <= end]
        by_week = {}
        for date in sessions:
            key = datetime.strptime(date, "%Y-%m-%d").isocalendar()[:2]
            by_week[key] = date
        return tuple(by_week[key] for key in sorted(by_week))

    def _ranking(self, anchor: str, member_names: Sequence[str]) -> RankingResult:
        universe = _universe_at(self.inputs.universe_intervals, anchor)
        source_by_name = {source.model_name: source for source in self.inputs.sources}
        if len(member_names) != len(set(member_names)) or any(name not in source_by_name for name in member_names):
            raise ReplayContractError("arm membership is not an exact source subset")
        normalized_columns = {}
        for name in member_names:
            scores, observed_rows = _date_scores(source_by_name[name].prediction, name, anchor)
            if observed_rows != len(universe) or set(scores) != set(universe) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in scores.values()
            ):
                raise ReplayInputError("cannot rank an incomplete model/date observation")
            normalized_columns[name] = pd.Series(
                _rank_percentiles({key: float(value) for key, value in scores.items()}),
                index=list(universe), dtype=float,
            )
        # Match the Production equal-fusion primitive exactly: construct the
        # normalized wide frame in declared member order, then use pandas'
        # row-wise mean.  A Python scalar sum can differ by one ULP and change
        # deterministic tie ordering even when scores are tolerance-equal.
        normalized_frame = pd.DataFrame(normalized_columns, index=list(universe))
        fused_series = normalized_frame[list(member_names)].mean(axis=1)
        fused = {instrument: float(value) for instrument, value in fused_series.items()}
        return canonical_full_ranking(universe, fused)

    def run(self, *, preferred_start: str, preferred_end: str, window_size: int = 6) -> dict:
        start = _strict_date(preferred_start, "preferred_start")
        end = _strict_date(preferred_end, "preferred_end")
        if start > end:
            raise ReplayContractError("preferred window is inverted")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or not 4 <= window_size <= 6:
            raise ReplayContractError("window_size must be an integer from 4 through 6")
        weekly = self._weekly_candidates(start, end)
        inventory_rows = [self._inventory_date(date) for date in weekly]
        complete_dates = [row["anchor"] for row in inventory_rows if row["status"] == "complete"]
        runs = []
        current_run = []
        for row in inventory_rows:
            if row["status"] == "complete":
                current_run.append(row["anchor"])
            elif current_run:
                runs.append(current_run)
                current_run = []
        if current_run:
            runs.append(current_run)
        eligible_runs = [run for run in runs if len(run) >= 4]
        selected = eligible_runs[-1][-window_size:] if eligible_runs else []
        inventory = {
            "status": "complete" if len(selected) >= 4 else "blocked_inventory",
            "preferred_start": start,
            "preferred_end": end,
            "weekly_candidates": list(weekly),
            "selected_anchors": selected,
            "excluded_anchors": [
                {
                    "anchor": row["anchor"],
                    "reason": (
                        "incomplete_model_or_universe_coverage"
                        if row["status"] != "complete" else "outside_selected_contiguous_window"
                    ),
                }
                for row in inventory_rows if row["anchor"] not in selected
            ],
            "dates": inventory_rows,
        }
        base = {
            "schema_version": 2,
            "evidence_class": "RETROSPECTIVE_TECHNICAL_REPLAY",
            "prospective_claim": False,
            "promotion_capability": False,
            "warnings": list(WARNING_LINES),
            "fusion_definition": FUSION_DEFINITION,
            "source_models": [source.public_identity() for source in self.inputs.sources],
            "universe_identity": {
                "name": self.inputs.universe_name,
                "digest": dict(self.inputs.universe_digest),
            },
            "calendar_identity": {
                "logical_path": "calendars/day.txt",
                "digest": dict(self.inputs.calendar_digest),
            },
            "source_to_materialization_relation": self.inputs.source_to_materialization_relation,
            "price_source_inventory": [
                dict(item) for item in self.inputs.price_source_inventory
            ],
            "price_source_inventory_digest": (
                dict(self.inputs.price_source_inventory_digest)
                if self.inputs.price_source_inventory_digest is not None else None
            ),
            "inventory": inventory,
            "rankings": {},
            "comparisons": [],
            "parity": {"status": "not_run"},
        }
        if inventory["status"] != "complete":
            base["status"] = "blocked_inventory"
            base["result_digest"] = _canonical_digest({key: value for key, value in base.items() if key != "result_digest"})
            return ReplayResult(base, _RESULT_AUTHORITY)
        if self.inputs.parity_anchor not in complete_dates:
            base["status"] = "blocked_parity"
            base["parity"] = {"status": "failed", "reason": "sealed_anchor_is_not_complete"}
            base["result_digest"] = _canonical_digest({key: value for key, value in base.items() if key != "result_digest"})
            return ReplayResult(base, _RESULT_AUTHORITY)
        model_names = tuple(source.model_name for source in self.inputs.sources)
        champion = self._ranking(self.inputs.parity_anchor, model_names)
        sealed = self.inputs.sealed_ranking
        champion_rows = {row["instrument"]: row for row in champion.rows}
        sealed_rows = {row["instrument"]: row for row in sealed.rows}
        score_mismatches = []
        for instrument in sorted(set(champion_rows) & set(sealed_rows)):
            left, right = champion_rows[instrument], sealed_rows[instrument]
            if left["scored"] and right["scored"] and abs(float(left["raw_score"]) - float(right["raw_score"])) > PARITY_SCORE_TOLERANCE:
                score_mismatches.append(instrument)
        parity_facts = {
            "eligible_identity_equal": set(champion_rows) == set(sealed_rows),
            "scored_identity_equal": {
                key for key, row in champion_rows.items() if row["scored"]
            } == {key for key, row in sealed_rows.items() if row["scored"]},
            "rank_order_equal": [row["instrument"] for row in champion.rows] == [row["instrument"] for row in sealed.rows],
            "score_tolerance": format(PARITY_SCORE_TOLERANCE, ".17g"),
            "score_mismatches": score_mismatches,
            "universe_digest_equal": True,
            "sealed_ranking_digest": dict(self.inputs.sealed_ranking_digest),
        }
        parity_ok = all(
            parity_facts[key] for key in (
                "eligible_identity_equal", "scored_identity_equal", "rank_order_equal", "universe_digest_equal",
            )
        ) and not score_mismatches
        base["parity"] = {"status": "passed" if parity_ok else "failed", **parity_facts}
        base["rankings"][self.inputs.parity_anchor] = {"CHAMPION_4": champion}
        if not parity_ok:
            base["status"] = "blocked_parity"
            base["result_digest"] = _canonical_digest(_public_result(base, include_digest=False, include_csv=False))
            return ReplayResult(base, _RESULT_AUTHORITY)
        arms = [("CHAMPION_4", model_names)] + [
            ("DROP_%s_3" % (index + 1), tuple(name for pos, name in enumerate(model_names) if pos != index))
            for index in range(len(model_names))
        ]
        previous_top = {}
        replay_anchors = sorted(set(selected) | {self.inputs.parity_anchor})
        for anchor in replay_anchors:
            rankings = {}
            for arm_name, members in arms:
                ranking = champion if anchor == self.inputs.parity_anchor and arm_name == "CHAMPION_4" else self._ranking(anchor, members)
                rankings[arm_name] = ranking
                if arm_name != "CHAMPION_4":
                    comparison = {
                        "anchor": anchor,
                        "arm": arm_name,
                        "members": list(members),
                        **_compare(rankings["CHAMPION_4"], ranking, min(self.top_k, ranking.scored_count)),
                    }
                    current_top = set(_top(ranking, min(self.top_k, ranking.scored_count)))
                    prior = previous_top.get(arm_name)
                    comparison["weekly_top_k_entered"] = sorted(current_top - prior) if prior is not None else []
                    comparison["weekly_top_k_exited"] = sorted(prior - current_top) if prior is not None else []
                    previous_top[arm_name] = current_top
                    base["comparisons"].append(comparison)
            base["rankings"][anchor] = rankings
        base["status"] = "complete"
        base["result_digest"] = _canonical_digest(_public_result(base, include_digest=False, include_csv=False))
        return ReplayResult(base, _RESULT_AUTHORITY)


def _copy_result_value(value: Any) -> Any:
    if isinstance(value, RankingResult):
        return RankingResult(
            tuple(dict(row) for row in value.rows),
            value.eligible_count,
            value.scored_count,
            value.missing_count,
            value.complete,
        )
    if isinstance(value, Mapping):
        return {key: _copy_result_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_result_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_result_value(item) for item in value)
    return value


class ReplayResult(Mapping[str, Any]):
    """Inspector-owned terminal result; JSON replay cannot manufacture status."""

    def __init__(self, payload: Mapping[str, Any], authority: object = None) -> None:
        if authority is not _RESULT_AUTHORITY:
            raise ReplayContractError("replay results are truth-owner constructed")
        copied = _copy_result_value(payload)
        digest = copied.get("result_digest")
        if not isinstance(digest, Mapping):
            raise ReplayContractError("terminal replay result lacks its digest")
        expected = _canonical_digest(_public_result(copied, include_digest=False, include_csv=False))
        if dict(digest) != expected:
            raise ReplayContractError("terminal replay result digest is inconsistent")
        self._payload = copied

    def __getitem__(self, key: str) -> Any:
        return _copy_result_value(self._payload[key])

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def _trusted_payload(self) -> Mapping[str, Any]:
        return self._payload


def revalidate_replay_result(result: ReplayResult) -> ReplayResult:
    """Return a fresh truth-owner snapshot after rechecking the current payload."""
    if not isinstance(result, ReplayResult):
        raise ReplayContractError("revalidation requires a replay-owned result")
    try:
        return ReplayResult(result._trusted_payload(), _RESULT_AUTHORITY)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        raise ReplayContractError("current replay result failed revalidation") from exc


def _public_result(result: Mapping[str, Any], *, include_digest: bool = True, include_csv: bool = False) -> dict:
    rendered = {}
    for key, value in result.items():
        if key == "result_digest" and not include_digest:
            continue
        if key == "rankings":
            rendered_rankings = {}
            for anchor, arms in value.items():
                rendered_rankings[anchor] = {}
                for arm, ranking in arms.items():
                    item = {
                        "eligible_count": ranking.eligible_count,
                        "scored_count": ranking.scored_count,
                        "missing_count": ranking.missing_count,
                        "complete": ranking.complete,
                        "ranking_digest": _raw_digest(ranking.to_csv_bytes()),
                    }
                    if include_csv:
                        item["csv"] = ranking.to_csv_bytes().decode("utf-8")
                    rendered_rankings[anchor][arm] = item
            rendered[key] = rendered_rankings
        else:
            rendered[key] = value
    return rendered


def _summary_csv(result: Mapping[str, Any]) -> bytes:
    stream = io.StringIO(newline="")
    fields = (
        "anchor", "arm", "members", "full_ranking_spearman", "top_k",
        "top_k_overlap_count", "top_k_overlap_ratio", "top_k_entered", "top_k_exited",
        "bottom_k_overlap_count", "bottom_k_overlap_ratio", "bottom_k_entered", "bottom_k_exited",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in result.get("comparisons", []):
        writer.writerow({
            "anchor": item["anchor"],
            "arm": item["arm"],
            "members": "|".join(item["members"]),
            "full_ranking_spearman": item["full_ranking_spearman"],
            "top_k": item["top_k"],
            "top_k_overlap_count": item["top_k_overlap_count"],
            "top_k_overlap_ratio": item["top_k_overlap_ratio"],
            "top_k_entered": "|".join(item["top_k_entered"]),
            "top_k_exited": "|".join(item["top_k_exited"]),
            "bottom_k_overlap_count": item["bottom_k_overlap_count"],
            "bottom_k_overlap_ratio": item["bottom_k_overlap_ratio"],
            "bottom_k_entered": "|".join(item["bottom_k_entered"]),
            "bottom_k_exited": "|".join(item["bottom_k_exited"]),
        })
    return stream.getvalue().encode("utf-8")


def _report_markdown(result: Mapping[str, Any]) -> bytes:
    lines = ["# Champion–Challenger Ranking Replay", ""]
    lines.extend(["> %s" % line for line in WARNING_LINES])
    lines.extend([
        "", "- Status: `%s`" % result["status"],
        "- Fusion: `%s`" % result["fusion_definition"],
        "- Parity: `%s`" % result["parity"]["status"],
        "- Calendar SHA-256: `%s`" % result["calendar_identity"]["digest"]["value"],
        "- Source-to-materialization relation: `%s`" % result["source_to_materialization_relation"],
        "- Price source members: `%s`" % len(result["price_source_inventory"]),
        "- Selected anchors: `%s`" % ", ".join(result["inventory"]["selected_anchors"]),
        "", "## Arms", "",
    ])
    arm_members = {}
    for item in result.get("comparisons", []):
        arm_members.setdefault(item["arm"], item["members"])
    for arm, members in sorted(arm_members.items()):
        lines.append("- `%s`: %s" % (arm, ", ".join(members)))
    lines.extend([
        "", "## Ranking comparison", "",
        "| Anchor | Arm | Spearman | Top-K overlap | Bottom-K overlap |",
        "|---|---|---:|---:|---:|",
    ])
    for item in result.get("comparisons", []):
        lines.append("| %s | %s | %s | %s/%s | %s/%s |" % (
            item["anchor"], item["arm"], item["full_ranking_spearman"],
            item["top_k_overlap_count"], item["top_k"],
            item["bottom_k_overlap_count"], item["top_k"],
        ))
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_replay_output(result: Mapping[str, Any], output_dir: Path) -> dict:
    """Create one deterministic, disposable replay tree below physical /tmp."""

    if not isinstance(result, ReplayResult):
        raise ReplayContractError("only a truth-owner replay result can be written")
    result = result._trusted_payload()
    output = Path(output_dir)
    tmp_root = Path("/tmp").resolve(strict=True)
    if output.exists():
        raise ReplayContractError("output directory must not already exist")
    raw_parent = output.absolute().parent
    current = Path("/")
    for part in raw_parent.parts[1:]:
        current = current / part
        info = os.lstat(str(current))
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ReplayContractError("output parent must be a canonical directory")
    parent = raw_parent.resolve(strict=True)
    try:
        parent.relative_to(tmp_root)
    except ValueError as exc:
        raise ReplayContractError("output directory must resolve below /tmp") from exc
    parent_identity = os.lstat(str(parent))
    output.mkdir(mode=0o700)
    parent_after = os.lstat(str(parent))
    output_info = os.lstat(str(output))
    if (
        (parent_identity.st_dev, parent_identity.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
        or not stat.S_ISDIR(output_info.st_mode) or stat.S_ISLNK(output_info.st_mode)
        or output.resolve(strict=True).parent != parent
    ):
        raise ReplayContractError("output namespace identity changed during create")
    output_identity = (output_info.st_dev, output_info.st_ino)
    files = {
        "inventory.json": canonical_json_bytes(result["inventory"]),
        "summary.csv": _summary_csv(result),
        "report.md": _report_markdown(result),
        "result.json": canonical_json_bytes(_public_result(result, include_csv=False)),
    }
    for anchor, arms in result.get("rankings", {}).items():
        for arm, ranking in arms.items():
            files["rankings/%s/%s.csv" % (anchor, arm)] = ranking.to_csv_bytes()
    for logical, data in sorted(files.items()):
        target = output / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        target_parent = os.lstat(str(target.parent))
        if not stat.S_ISDIR(target_parent.st_mode) or stat.S_ISLNK(target_parent.st_mode):
            raise ReplayContractError("output member parent is not canonical")
        target_parent_identity = (target_parent.st_dev, target_parent.st_ino)
        temporary = target.with_name(".%s.tmp" % target.name)
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        try:
            current_root = os.lstat(str(output))
            current_parent = os.lstat(str(target.parent))
            current_target = os.lstat(str(target))
        except OSError as exc:
            raise ReplayContractError(
                "output member namespace identity changed during commit"
            ) from exc
        if (
            (current_root.st_dev, current_root.st_ino) != output_identity
            or (current_parent.st_dev, current_parent.st_ino) != target_parent_identity
            or not stat.S_ISREG(current_target.st_mode)
            or stat.S_ISLNK(current_target.st_mode) or current_target.st_nlink != 1
            or target.resolve(strict=True).parent != target.parent.resolve(strict=True)
        ):
            raise ReplayContractError("output member namespace identity changed during commit")
    manifest = {
        "status": result["status"],
        "output_files": [
            {"path": logical, "digest": _raw_digest(data)}
            for logical, data in sorted(files.items())
        ],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    with (output / "output_manifest.json").open("xb") as handle:
        handle.write(manifest_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    final_root = os.lstat(str(output))
    final_manifest = os.lstat(str(output / "output_manifest.json"))
    if (
        (final_root.st_dev, final_root.st_ino) != output_identity
        or not stat.S_ISREG(final_manifest.st_mode)
        or stat.S_ISLNK(final_manifest.st_mode) or final_manifest.st_nlink != 1
    ):
        raise ReplayContractError("output manifest namespace identity changed during commit")
    return {**manifest, "output_manifest_digest": _raw_digest(manifest_bytes)}
