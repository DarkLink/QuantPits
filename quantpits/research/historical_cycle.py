"""One-cycle, five-arm retrospective shadow replay.

The module joins the already-public Stage A ranking, B1 intent-planning and
B0 accounting contracts.  Market data is read directly from Qlib's immutable
binary files; Qlib and MLflow are deliberately never initialized here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import stat
import struct
import zlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.ranking import RankingResult
from quantpits.research.accounting import (
    ExecutionAssumption,
    ShadowPortfolioState,
    ShadowPortfolioTransition,
    ShadowQuoteSnapshot,
)
from quantpits.research.intents import (
    AnchorPriceSnapshot,
    CurrentRuleIntentDefinition,
    CurrentRuleShadowIntentPlanner,
)
from quantpits.research.replay import ReplayResult, SealedReplayInputs


ARM_IDS = ("CHAMPION_4", "DROP_1_3", "DROP_2_3", "DROP_3_3", "DROP_4_3")
EVIDENCE_CLASS = "RETROSPECTIVE_TECHNICAL_REPLAY"
DERIVATION_RULE = "IEEE754_FLOAT64_DIVISION_THEN_17G_V1"
WARNING_LINES = (
    "RETROSPECTIVE TECHNICAL REPLAY",
    "Not prospective evidence.",
    "Not an automatic promotion basis.",
    "Current-rule replay may differ from historical Production behavior.",
    "Dolt-to-Qlib materialization relation remains unverified.",
    "Corporate actions are unmodeled.",
)
_AUTHORITY = object()
_INSTRUMENT_PREFIXES = ("SH", "SZ")


class HistoricalCycleContractError(ValueError):
    """A caller representation violates the B2 contract."""


class HistoricalCycleInputError(RuntimeError):
    """A private or provider input could not be observed exactly."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest_bytes(data: bytes, domain: str = "raw_bytes") -> Dict[str, Any]:
    return {
        "algorithm": "sha256", "domain": domain,
        "value": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }


def _digest_payload(value: Any) -> Dict[str, Any]:
    return _digest_bytes(_canonical_json(value), "canonical_json")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HistoricalCycleContractError("%s must be non-empty exact text" % field)
    return value


def _date(value: Any, field: str) -> str:
    raw = _text(value, field)
    try:
        from datetime import datetime
        parsed = datetime.strptime(raw, "%Y-%m-%d")
    except ValueError as exc:
        raise HistoricalCycleContractError("%s must be YYYY-MM-DD" % field) from exc
    if parsed.strftime("%Y-%m-%d") != raw:
        raise HistoricalCycleContractError("%s must be YYYY-MM-DD" % field)
    return raw


def _instrument(value: Any, field: str) -> str:
    raw = _text(value, field)
    if len(raw) != 8 or raw[:2] not in _INSTRUMENT_PREFIXES or not raw[2:].isdigit():
        raise HistoricalCycleContractError("%s must be a canonical SH/SZ instrument" % field)
    return raw


def _strict_decimal(value: Any, field: str, *, positive: bool = False) -> Decimal:
    raw = _text(value, field)
    try:
        result = Decimal(raw)
    except InvalidOperation as exc:
        raise HistoricalCycleContractError("%s must be a decimal string" % field) from exc
    if not result.is_finite() or (positive and result <= 0):
        raise HistoricalCycleContractError("%s must be finite%s" % (field, " and positive" if positive else ""))
    return result


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _strict_json(data: bytes, name: str) -> Mapping[str, Any]:
    def pairs(items: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise HistoricalCycleInputError("%s is not strict JSON" % name) from exc
    if not isinstance(value, Mapping):
        raise HistoricalCycleInputError("%s must be a JSON object" % name)
    return value


def _exact_keys(value: Any, fields: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalCycleContractError("%s must be an object" % name)
    if set(value) != set(fields):
        raise HistoricalCycleContractError(
            "%s fields mismatch (missing=%r, extra=%r)" % (
                name, sorted(set(fields) - set(value)), sorted(set(value) - set(fields)),
            )
        )
    return value


def _path_identity(info: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _root_identity(root: Path) -> Tuple[int, int, int, int, int]:
    info = os.lstat(str(root))
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise HistoricalCycleInputError("authority root must be a physical directory")
    # Directory size/mtime necessarily change when a child is created.  The
    # namespace continuity fact is the physical directory object itself.
    return (info.st_dev, info.st_ino, info.st_mode, 0, 0)


def _try_root_identity(root: Path) -> Optional[Tuple[int, int, int, int, int]]:
    try:
        return _root_identity(root)
    except (OSError, HistoricalCycleInputError):
        return None


def _physical_file(root: Path, logical: str) -> Path:
    if not isinstance(logical, str) or not logical or "\\" in logical or "\0" in logical:
        raise HistoricalCycleInputError("source logical path is invalid")
    relative = Path(logical)
    if relative.is_absolute() or relative.as_posix() != logical or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise HistoricalCycleInputError("source logical path is not canonical")
    canonical_root = root.resolve(strict=True)
    current = canonical_root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            info = os.lstat(str(current))
        except FileNotFoundError:
            return current
        if stat.S_ISLNK(info.st_mode):
            raise HistoricalCycleInputError("source path contains a symlink alias")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise HistoricalCycleInputError("source path contains a non-directory ancestor")
    try:
        current.resolve(strict=False).relative_to(canonical_root)
    except ValueError as exc:
        raise HistoricalCycleInputError("source path escapes its authority root") from exc
    return current


def _stable_read(root: Path, logical: str, *, allow_missing: bool = False) -> Optional[bytes]:
    root_before = _root_identity(root)
    path = _physical_file(root, logical)
    try:
        before = os.lstat(str(path))
    except FileNotFoundError:
        if allow_missing and _root_identity(root) == root_before:
            return None
        raise HistoricalCycleInputError("source file is missing: %s" % logical)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1:
        raise HistoricalCycleInputError("source is not an exclusive regular file: %s" % logical)
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        data = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = os.lstat(str(path))
    identities = tuple(_path_identity(item) for item in (before, opened, opened_after, after))
    if len(set(identities)) != 1 or _root_identity(root) != root_before:
        raise HistoricalCycleInputError("source identity changed during read: %s" % logical)
    return data


@dataclass(frozen=True, init=False)
class ReplayProfileReceipt:
    schema_version: int
    profile_id: str
    research_protocol_id: str
    bootstrap: Mapping[str, Any]
    definition: CurrentRuleIntentDefinition
    assumption: ExecutionAssumption
    raw_digest: Mapping[str, Any]
    canonical_digest: Mapping[str, Any]
    _private_path: Path
    _file_identity: Tuple[int, int, int, int, int]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _AUTHORITY or args:
            raise HistoricalCycleContractError("profile receipts are inspector-owned")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def public_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "research_protocol_id": self.research_protocol_id,
            "bootstrap": {
                "as_of_date": self.bootstrap["as_of_date"],
                "cash": self.bootstrap["cash"],
                "positions": [dict(item) for item in self.bootstrap["positions"]],
            },
            "definition": self.definition.to_dict(),
            "execution_assumption": self.assumption.to_dict(),
            "raw_digest": dict(self.raw_digest),
            "canonical_digest": dict(self.canonical_digest),
        }

    def revalidate(self) -> "ReplayProfileReceipt":
        current = load_replay_profile(self._private_path)
        if current.raw_digest != self.raw_digest or current._file_identity != self._file_identity:
            raise HistoricalCycleInputError("private profile identity changed")
        return current


def load_replay_profile(path: Path) -> ReplayProfileReceipt:
    """Strictly observe one workspace-private replay profile."""
    source = Path(path)
    if not source.is_absolute() or source != source.resolve(strict=True):
        raise HistoricalCycleInputError("profile path must be an absolute physical path without aliases")
    parent = source.parent.resolve(strict=True)
    logical = source.name
    if source != source.parent / logical or logical in ("", ".", ".."):
        raise HistoricalCycleInputError("profile path must name one direct file")
    data = _stable_read(parent, logical)
    assert data is not None
    info = os.lstat(str(source.resolve(strict=True)))
    raw = _strict_json(data, "profile")
    raw = _exact_keys(
        raw, ("schema_version", "profile_id", "research_protocol_id", "bootstrap", "intent_definition", "execution_assumption"),
        "profile",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise HistoricalCycleContractError("profile.schema_version must be integer 1")
    if raw["research_protocol_id"] != "RETROSPECTIVE_SINGLE_CYCLE_V1":
        raise HistoricalCycleContractError("profile.research_protocol_id is unsupported")
    bootstrap = _exact_keys(raw["bootstrap"], ("as_of_date", "cash", "positions"), "profile.bootstrap")
    as_of = _date(bootstrap["as_of_date"], "profile.bootstrap.as_of_date")
    cash = _strict_decimal(bootstrap["cash"], "profile.bootstrap.cash")
    if cash < 0:
        raise HistoricalCycleContractError("profile.bootstrap.cash must be non-negative")
    rows = bootstrap["positions"]
    if not isinstance(rows, list):
        raise HistoricalCycleContractError("profile.bootstrap.positions must be a list")
    canonical_rows = []
    for index, item in enumerate(rows):
        row = _exact_keys(item, ("instrument", "quantity", "book_cost"), "profile.bootstrap.positions[%d]" % index)
        instrument = _instrument(row["instrument"], "profile.bootstrap.positions[%d].instrument" % index)
        if isinstance(row["quantity"], bool) or not isinstance(row["quantity"], int) or row["quantity"] <= 0:
            raise HistoricalCycleContractError("profile position quantity must be a positive integer")
        book_cost = _strict_decimal(row["book_cost"], "profile position book_cost")
        if book_cost < 0:
            raise HistoricalCycleContractError("profile position book_cost must be non-negative")
        canonical_rows.append({"instrument": instrument, "quantity": row["quantity"], "book_cost": _decimal_text(book_cost)})
    if tuple(item["instrument"] for item in canonical_rows) != tuple(sorted(item["instrument"] for item in canonical_rows)) or len({item["instrument"] for item in canonical_rows}) != len(canonical_rows):
        raise HistoricalCycleContractError("profile positions must be unique and sorted")
    try:
        definition = CurrentRuleIntentDefinition.from_dict(raw["intent_definition"])
        assumption = ExecutionAssumption.from_dict(raw["execution_assumption"])
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except ValueError as exc:
        raise HistoricalCycleContractError(
            "profile definition or execution assumption is invalid (%s)" % type(exc).__name__
        ) from exc
    canonical = {
        "schema_version": 1,
        "profile_id": _text(raw["profile_id"], "profile.profile_id"),
        "research_protocol_id": "RETROSPECTIVE_SINGLE_CYCLE_V1",
        "bootstrap": {"as_of_date": as_of, "cash": _decimal_text(cash), "positions": canonical_rows},
        "intent_definition": definition.to_dict(),
        "execution_assumption": assumption.to_dict(),
    }
    return ReplayProfileReceipt(
        _authority=_AUTHORITY, schema_version=1, profile_id=canonical["profile_id"],
        research_protocol_id=canonical["research_protocol_id"],
        bootstrap=MappingProxyType({
            "as_of_date": as_of, "cash": canonical["bootstrap"]["cash"],
            "positions": tuple(MappingProxyType(dict(item)) for item in canonical_rows),
        }),
        definition=definition, assumption=assumption, raw_digest=MappingProxyType(_digest_bytes(data)),
        canonical_digest=MappingProxyType(_digest_payload(canonical)), _private_path=source.resolve(strict=True),
        _file_identity=_path_identity(info),
    )


@dataclass(frozen=True, init=False)
class CashPriceReceipt:
    observation_kind: str
    observation_date: str
    calendar_position: int
    requested_instruments: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]
    observed_instruments: Tuple[str, ...]
    missing_instruments: Tuple[str, ...]
    invalid_instruments: Tuple[str, ...]
    digest: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _AUTHORITY or args:
            raise HistoricalCycleContractError("cash-price receipts are observer-owned")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def _payload(self) -> Dict[str, Any]:
        return {
            "observation_kind": self.observation_kind,
            "observation_date": self.observation_date,
            "calendar_position": self.calendar_position,
            "requested_instruments": list(self.requested_instruments),
            "rows": [dict(item) for item in self.rows],
            "observed_instruments": list(self.observed_instruments),
            "missing_instruments": list(self.missing_instruments),
            "invalid_instruments": list(self.invalid_instruments),
            "counts": {
                "requested": len(self.requested_instruments),
                "raw_sources": len(self.rows) * 2,
                "observed": len(self.observed_instruments),
                "missing": len(self.missing_instruments),
                "invalid": len(self.invalid_instruments),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        result = self._payload()
        result["digest"] = dict(self.digest)
        return result

    def _validated_copy(self) -> "CashPriceReceipt":
        if self.observation_kind not in ("CASH_CLOSE", "NEXT_OPEN"):
            raise HistoricalCycleContractError("cash-price receipt kind is invalid")
        _date(self.observation_date, "price receipt date")
        if isinstance(self.calendar_position, bool) or not isinstance(self.calendar_position, int) or self.calendar_position < 0:
            raise HistoricalCycleContractError("cash-price calendar position is invalid")
        if self.requested_instruments != tuple(sorted(set(self.requested_instruments))) or not self.requested_instruments:
            raise HistoricalCycleContractError("cash-price requested set is invalid")
        if tuple(row.get("instrument") for row in self.rows) != self.requested_instruments:
            raise HistoricalCycleContractError("cash-price rows do not exactly cover the requested set")
        for row in self.rows:
            if set(row) != {
                "instrument", "observation_date", "calendar_position", "derived_field",
                "status", "reason_code", "numerator", "denominator", "derivation_rule", "cash_price",
            }:
                raise HistoricalCycleContractError("cash-price row fields are not exact")
            if (
                row["observation_date"] != self.observation_date
                or row["calendar_position"] != self.calendar_position
                or row["derived_field"] != self.observation_kind
                or row["derivation_rule"] != DERIVATION_RULE
                or row["status"] not in ("OBSERVED", "MISSING", "INVALID")
            ):
                raise HistoricalCycleContractError("cash-price row authority fields are inconsistent")
            if (row["status"] == "OBSERVED") != (isinstance(row["cash_price"], str)):
                raise HistoricalCycleContractError("cash-price status/value fields are inconsistent")
            if row["status"] == "OBSERVED":
                _strict_decimal(row["cash_price"], "cash price", positive=True)
        observed = tuple(row["instrument"] for row in self.rows if row["status"] == "OBSERVED")
        missing = tuple(row["instrument"] for row in self.rows if row["status"] == "MISSING")
        invalid = tuple(row["instrument"] for row in self.rows if row["status"] == "INVALID")
        if (observed, missing, invalid) != (
            self.observed_instruments, self.missing_instruments, self.invalid_instruments,
        ):
            raise HistoricalCycleContractError("cash-price partitions are inconsistent")
        expected = _digest_payload(self._payload())
        if dict(self.digest) != expected:
            raise HistoricalCycleContractError("cash-price receipt digest is inconsistent")
        return self

    def to_anchor_snapshot(self) -> AnchorPriceSnapshot:
        self._validated_copy()
        if self.observation_kind != "CASH_CLOSE":
            raise HistoricalCycleContractError("only a CASH_CLOSE receipt can enter planning")
        return AnchorPriceSnapshot.from_iterable(
            anchor_date=self.observation_date, requested_instruments=self.requested_instruments,
            rows=[{
                "instrument": row["instrument"], "anchor_date": self.observation_date,
                "status": "OBSERVED" if row["status"] == "OBSERVED" else "MISSING",
                "cash_close": row["cash_price"] if row["status"] == "OBSERVED" else None,
            } for row in self.rows],
        )

    def to_quote_snapshot(self, *, portfolio_id: str, cycle_id: str) -> ShadowQuoteSnapshot:
        self._validated_copy()
        if self.observation_kind != "NEXT_OPEN":
            raise HistoricalCycleContractError("only a NEXT_OPEN receipt can enter settlement")
        return ShadowQuoteSnapshot.from_iterable(
            portfolio_id=portfolio_id, cycle_id=cycle_id, trade_date=self.observation_date,
            requested_instruments=self.requested_instruments,
            rows=[{
                "instrument": row["instrument"], "trade_date": self.observation_date,
                "field": "NEXT_OPEN", "status": "OBSERVED" if row["status"] == "OBSERVED" else "MISSING",
                "price": row["cash_price"] if row["status"] == "OBSERVED" else None,
            } for row in self.rows],
        )


class QlibCashPriceObserver:
    """Direct, stable reader for Qlib float32 price/factor words."""

    def __init__(
        self, provider_root: Path, calendar_dates: Sequence[str],
        calendar_digest: Optional[Mapping[str, Any]] = None,
    ) -> None:
        supplied_root = Path(provider_root)
        resolved_root = supplied_root.resolve(strict=True)
        if not supplied_root.is_absolute() or supplied_root != resolved_root:
            raise HistoricalCycleInputError("provider root must be an absolute physical path without aliases")
        self._root = resolved_root
        self._root_identity = _root_identity(self._root)
        dates = tuple(_date(item, "calendar_dates") for item in calendar_dates)
        if not dates or dates != tuple(sorted(set(dates))):
            raise HistoricalCycleContractError("calendar_dates must be non-empty, unique and increasing")
        calendar = _stable_read(self._root, "calendars/day.txt")
        assert calendar is not None
        if calendar_digest is not None and _digest_bytes(calendar) != dict(calendar_digest):
            raise HistoricalCycleInputError("provider calendar bytes differ from Stage A")
        try:
            observed_dates = tuple(row.strip()[:10] for row in calendar.decode("utf-8").splitlines() if row.strip())
        except UnicodeDecodeError as exc:
            raise HistoricalCycleInputError("provider calendar is not UTF-8") from exc
        if observed_dates != dates:
            raise HistoricalCycleInputError("provider calendar inventory differs from Stage A")
        self._dates = dates
        self._calendar_digest = _digest_bytes(calendar)
        self._calendar_identity = _path_identity(os.lstat(str(self._root / "calendars/day.txt")))
        self._source_identities: Dict[str, Optional[Tuple[int, int, int, int, int]]] = {}

    def _word(self, instrument: str, field: str, position: int) -> Mapping[str, Any]:
        logical = "features/%s/%s.day.bin" % (instrument.lower(), field)
        path = _physical_file(self._root, logical)
        try:
            current_identity: Optional[Tuple[int, int, int, int, int]] = _path_identity(os.lstat(str(path)))
        except FileNotFoundError:
            current_identity = None
        if logical in self._source_identities and self._source_identities[logical] != current_identity:
            raise HistoricalCycleInputError("price source identity changed within logical replay: %s" % logical)
        self._source_identities.setdefault(logical, current_identity)
        data = _stable_read(self._root, logical, allow_missing=True)
        if data is None:
            return {"field": field, "logical_path": logical, "status": "MISSING", "reason_code": "SOURCE_FILE_MISSING", "raw_file_digest": None, "feature_float32_bits": None, "feature_text": None}
        base = {"field": field, "logical_path": logical, "raw_file_digest": _digest_bytes(data)}
        if len(data) < 8 or len(data) % 4:
            return {**base, "status": "INVALID", "reason_code": "MALFORMED_BIN", "feature_float32_bits": None, "feature_text": None}
        start = struct.unpack_from("<f", data, 0)[0]
        if not math.isfinite(start) or int(start) != start:
            return {**base, "status": "INVALID", "reason_code": "INVALID_START_POSITION", "feature_float32_bits": data[:4].hex(), "feature_text": format(float(start), ".17g")}
        offset = (position - int(start) + 1) * 4
        if offset < 4:
            return {**base, "status": "MISSING", "reason_code": "POSITION_BEFORE_START", "feature_float32_bits": None, "feature_text": None}
        if offset + 4 > len(data):
            return {**base, "status": "MISSING", "reason_code": "MISSING_TAIL", "feature_float32_bits": None, "feature_text": None}
        word = data[offset:offset + 4]
        value = struct.unpack("<f", word)[0]
        if not math.isfinite(value):
            return {**base, "status": "INVALID", "reason_code": "NONFINITE_WORD", "feature_float32_bits": word.hex(), "feature_text": format(float(value), ".17g")}
        return {**base, "status": "OBSERVED", "reason_code": "OBSERVED", "feature_float32_bits": word.hex(), "feature_text": format(float(value), ".17g")}

    def _observe(self, kind: str, date: str, requested: Sequence[str]) -> CashPriceReceipt:
        if _root_identity(self._root) != self._root_identity:
            raise HistoricalCycleInputError("provider root identity changed")
        calendar = _stable_read(self._root, "calendars/day.txt")
        assert calendar is not None
        if (
            _digest_bytes(calendar) != self._calendar_digest
            or _path_identity(os.lstat(str(self._root / "calendars/day.txt"))) != self._calendar_identity
        ):
            raise HistoricalCycleInputError("provider calendar identity changed")
        date = _date(date, "observation_date")
        if date not in self._dates:
            raise HistoricalCycleContractError("observation date is absent from calendar")
        instruments = tuple(_instrument(item, "requested_instruments") for item in requested)
        if not instruments or instruments != tuple(sorted(set(instruments))):
            raise HistoricalCycleContractError("requested instruments must be non-empty, unique and sorted")
        position = self._dates.index(date)
        field = "close" if kind == "CASH_CLOSE" else "open"
        rows = []
        for instrument in instruments:
            numerator = self._word(instrument, field, position)
            denominator = self._word(instrument, "factor", position)
            status = "OBSERVED"
            reason = "OBSERVED"
            cash = None
            if numerator["status"] == "MISSING" or denominator["status"] == "MISSING":
                status, reason = "MISSING", "NUMERATOR_OR_FACTOR_MISSING"
            elif numerator["status"] != "OBSERVED" or denominator["status"] != "OBSERVED":
                status, reason = "INVALID", "NUMERATOR_OR_FACTOR_INVALID"
            else:
                factor = float(denominator["feature_text"])
                value = float(numerator["feature_text"])
                derived = value / factor if factor > 0 else float("nan")
                if factor <= 0 or not math.isfinite(derived) or derived <= 0:
                    status, reason = "INVALID", "INVALID_FACTOR_OR_DERIVED_PRICE"
                else:
                    cash = format(derived, ".17g")
            rows.append(MappingProxyType({
                "instrument": instrument, "observation_date": date,
                "calendar_position": position, "derived_field": kind,
                "status": status, "reason_code": reason,
                "numerator": numerator, "denominator": denominator,
                "derivation_rule": DERIVATION_RULE, "cash_price": cash,
            }))
        observed = tuple(row["instrument"] for row in rows if row["status"] == "OBSERVED")
        missing = tuple(row["instrument"] for row in rows if row["status"] == "MISSING")
        invalid = tuple(row["instrument"] for row in rows if row["status"] == "INVALID")
        payload = {
            "observation_kind": kind, "observation_date": date, "calendar_position": position,
            "requested_instruments": list(instruments), "rows": [dict(row) for row in rows],
            "observed_instruments": list(observed), "missing_instruments": list(missing),
            "invalid_instruments": list(invalid),
            "counts": {
                "requested": len(instruments), "raw_sources": len(rows) * 2,
                "observed": len(observed), "missing": len(missing), "invalid": len(invalid),
            },
        }
        if _root_identity(self._root) != self._root_identity:
            raise HistoricalCycleInputError("provider root identity changed")
        return CashPriceReceipt(
            _authority=_AUTHORITY, observation_kind=kind, observation_date=date,
            calendar_position=position, requested_instruments=instruments, rows=tuple(rows),
            observed_instruments=observed, missing_instruments=missing, invalid_instruments=invalid,
            digest=MappingProxyType(_digest_payload(payload)),
        )

    def observe_anchor_close(self, *, anchor_date: str, requested_instruments: Sequence[str]) -> CashPriceReceipt:
        return self._observe("CASH_CLOSE", anchor_date, requested_instruments)

    def observe_next_open(self, *, trade_date: str, requested_instruments: Sequence[str]) -> CashPriceReceipt:
        return self._observe("NEXT_OPEN", trade_date, requested_instruments)


def _revalidate_ranking(value: Any) -> RankingResult:
    if not isinstance(value, RankingResult):
        raise HistoricalCycleContractError("Stage A ranking is not canonical")
    try:
        return RankingResult(
            tuple(dict(row) for row in value.rows), value.eligible_count, value.scored_count,
            value.missing_count, value.complete,
        )
    except Exception as exc:
        raise HistoricalCycleContractError("Stage A ranking failed revalidation") from exc


def _ranking_digest(value: RankingResult) -> str:
    return hashlib.sha256(value.to_csv_bytes()).hexdigest()


def _deterministic_id(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json({"domain": domain, **dict(payload)})).hexdigest()


def _source_fingerprint(engine_root: Path) -> Mapping[str, Any]:
    files = (
        "quantpits/research/replay.py", "quantpits/research/intents.py",
        "quantpits/research/accounting.py", "quantpits/utils/strategy.py",
        "quantpits/research/historical_cycle.py",
    )
    members = []
    for logical in files:
        data = _stable_read(engine_root, logical)
        assert data is not None
        members.append({"logical_path": logical, "digest": _digest_bytes(data)})
    commit = "unavailable"
    try:
        head = _stable_read(engine_root / ".git", "HEAD")
        assert head is not None
        head_text = head.decode("ascii").strip()
        if head_text.startswith("ref: "):
            ref = head_text[5:]
            ref_data = _stable_read(engine_root / ".git", ref, allow_missing=True)
            if ref_data is not None:
                candidate = ref_data.decode("ascii").strip()
                if len(candidate) == 40 and all(char in "0123456789abcdef" for char in candidate):
                    commit = candidate
        elif len(head_text) == 40 and all(char in "0123456789abcdef" for char in head_text):
            commit = head_text
    except (OSError, UnicodeError, HistoricalCycleInputError):
        commit = "unavailable"
    tree = "unavailable"
    if commit != "unavailable":
        loose = engine_root / ".git" / "objects" / commit[:2] / commit[2:]
        try:
            compressed = _stable_read(loose.parent, loose.name, allow_missing=True)
            if compressed is not None:
                decoded = zlib.decompress(compressed)
                body = decoded.split(b"\0", 1)[1]
                first = body.splitlines()[0].decode("ascii")
                candidate = first[5:] if first.startswith("tree ") else ""
                if len(candidate) == 40 and all(char in "0123456789abcdef" for char in candidate):
                    tree = candidate
        except (OSError, UnicodeError, ValueError, HistoricalCycleInputError):
            tree = "unavailable"
    payload = {"engine_commit": commit, "engine_tree": tree, "source_files": members}
    return MappingProxyType({**payload, "digest": _digest_payload(payload)})


def _environment_fingerprint() -> Mapping[str, Any]:
    versions: Dict[str, str] = {"python": platform.python_version(), "implementation": platform.python_implementation()}
    for module_name in ("pandas", "numpy"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(module.__version__)
        except ImportError:
            versions[module_name] = "unavailable"
    return MappingProxyType({**versions, "digest": _digest_payload(versions)})


def rounding_adjustment_within_bound(
    adjustment: Optional[Decimal], filled_count: int, money_quantum: Decimal,
) -> bool:
    """Apply the frozen ``(filled_count + 3) * quantum / 2`` B2 bound."""
    if adjustment is not None and (not isinstance(adjustment, Decimal) or not adjustment.is_finite()):
        raise HistoricalCycleContractError("rounding adjustment must be a finite Decimal or null")
    if isinstance(filled_count, bool) or not isinstance(filled_count, int) or filled_count < 0:
        raise HistoricalCycleContractError("filled_count must be a non-negative integer")
    if not isinstance(money_quantum, Decimal) or not money_quantum.is_finite() or money_quantum <= 0:
        raise HistoricalCycleContractError("money_quantum must be a positive finite Decimal")
    if adjustment is None:
        return True
    bound = Decimal(filled_count + 3) * money_quantum / Decimal(2)
    return abs(adjustment) <= bound


@dataclass(frozen=True, init=False)
class PreparedCycle:
    requested_arm_ids: Tuple[str, ...]
    terminal_arms: Tuple[Mapping[str, Any], ...]
    anchor_prices: CashPriceReceipt
    status: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _AUTHORITY or args:
            raise HistoricalCycleContractError("prepared cycles are replay-owned")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)


class HistoricalCycleResult(Mapping[str, Any]):
    """Truth-owner result; JSON cannot mint completion capability."""

    def __init__(self, payload: Mapping[str, Any], authority: object = None) -> None:
        if authority is not _AUTHORITY:
            raise HistoricalCycleContractError("historical-cycle results are replay-owned")
        expected_fields = {
            "schema_version", "status", "evidence_class", "prospective_claim",
            "promotion_capability", "warnings", "operation_identity", "anchor_date",
            "trade_date", "stage_a_result_digest", "source_models", "universe_identity",
            "calendar_identity", "source_to_materialization_relation",
            "engine_source_fingerprint", "environment_fingerprint", "profile",
            "intent_definition_digest", "execution_assumption_digest", "anchor_price_receipt",
            "next_open_receipts", "requested_arm_ids", "terminal_arms", "arm_counts",
            "state_independence_checked", "lot_compatibility_checked", "comparisons",
            "comparison_capability", "result_digest",
        }
        if set(payload) != expected_fields:
            raise HistoricalCycleContractError("historical-cycle result fields are not exact")
        arms = payload["terminal_arms"]
        complete = sum(item.get("terminal_status") == "COMPLETE" for item in arms)
        if (
            payload["schema_version"] != 1 or payload["evidence_class"] != EVIDENCE_CLASS
            or payload["prospective_claim"] is not False or payload["promotion_capability"] is not False
            or tuple(payload["warnings"]) != WARNING_LINES
            or tuple(payload["requested_arm_ids"]) != ARM_IDS
            or tuple(item.get("arm_id") for item in arms) != ARM_IDS
            or payload["arm_counts"] != {
                "requested": 5, "terminal": 5, "complete": complete, "blocked": 5 - complete,
            }
        ):
            raise HistoricalCycleContractError("historical-cycle result authority fields are inconsistent")
        should_complete = (
            complete == 5 and len(payload["comparisons"]) == 4
            and payload["state_independence_checked"] is True
            and payload["lot_compatibility_checked"] is True
        )
        if (
            payload["status"] != ("COMPLETE" if should_complete else "BLOCKED")
            or payload["comparison_capability"] is not should_complete
        ):
            raise HistoricalCycleContractError("historical-cycle result capability is inconsistent")
        without_digest = {key: value for key, value in payload.items() if key != "result_digest"}
        if payload["result_digest"] != _digest_payload(without_digest):
            raise HistoricalCycleContractError("historical-cycle result digest is inconsistent")
        raw = _canonical_json(payload)
        self._bytes = raw
        self._payload = json.loads(raw.decode("utf-8"))

    def __getitem__(self, key: str) -> Any:
        return json.loads(json.dumps(self._payload[key]))

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self._bytes.decode("utf-8"))

    def to_canonical_json_bytes(self) -> bytes:
        return self._bytes


class HistoricalShadowCycleReplay:
    """Truth owner for one exact historical anchor and five independent arms."""

    def __init__(
        self, *, stage_a_result: ReplayResult, stage_a_inputs: SealedReplayInputs,
        profile: ReplayProfileReceipt, provider_root: Path, anchor_date: str,
        market: str, engine_root: Optional[Path] = None,
    ) -> None:
        if not isinstance(stage_a_result, ReplayResult) or not isinstance(stage_a_inputs, SealedReplayInputs):
            raise HistoricalCycleContractError("Stage A inputs must be truth-owner contracts")
        if not isinstance(profile, ReplayProfileReceipt):
            raise HistoricalCycleContractError("profile must be an inspector-owned receipt")
        if stage_a_result["status"] != "complete" or stage_a_result["parity"].get("status") != "passed":
            raise HistoricalCycleContractError("Stage A result is not complete with parity")
        anchor = _date(anchor_date, "anchor_date")
        if (
            stage_a_result["calendar_identity"]["digest"] != dict(stage_a_inputs.calendar_digest)
            or stage_a_result["universe_identity"]["digest"] != dict(stage_a_inputs.universe_digest)
            or stage_a_result["source_to_materialization_relation"] != stage_a_inputs.source_to_materialization_relation
            or stage_a_result["source_models"] != [item.public_identity() for item in stage_a_inputs.sources]
        ):
            raise HistoricalCycleContractError("Stage A result and sealed inputs have foreign identity")
        if anchor not in tuple(stage_a_result["inventory"]["selected_anchors"]):
            raise HistoricalCycleContractError("anchor is not a Stage A selected anchor")
        rankings = stage_a_result["rankings"].get(anchor)
        if not isinstance(rankings, Mapping) or tuple(rankings) != ARM_IDS:
            raise HistoricalCycleContractError("Stage A anchor does not carry exact ordered five arms")
        self._rankings = MappingProxyType({arm: _revalidate_ranking(rankings[arm]) for arm in ARM_IDS})
        self._stage_a_result_digest = stage_a_result["result_digest"]
        self._stage_a_public = {
            "source_models": stage_a_result["source_models"],
            "universe_identity": stage_a_result["universe_identity"],
            "calendar_identity": stage_a_result["calendar_identity"],
            "source_to_materialization_relation": stage_a_result["source_to_materialization_relation"],
        }
        self._inputs = stage_a_inputs
        self._profile = profile
        self._anchor = anchor
        self._market = _text(market, "market")
        position = stage_a_inputs.calendar_dates.index(anchor)
        if position + 1 >= len(stage_a_inputs.calendar_dates):
            raise HistoricalCycleContractError("anchor has no next trading session")
        self._trade_date = stage_a_inputs.calendar_dates[position + 1]
        if profile.bootstrap["as_of_date"] > anchor:
            raise HistoricalCycleContractError("bootstrap state is later than anchor")
        if profile.definition.production_buy_lot_size != profile.assumption.lot_size:
            raise HistoricalCycleContractError("intent and execution lot sizes differ")
        self._observer = QlibCashPriceObserver(
            provider_root, stage_a_inputs.calendar_dates, stage_a_inputs.calendar_digest,
        )
        root = Path(engine_root) if engine_root is not None else Path(__file__).resolve().parents[2]
        self._engine_root = root.resolve(strict=True)
        self._source = _source_fingerprint(self._engine_root)
        self._environment = _environment_fingerprint()

    @property
    def requested_anchor_instruments(self) -> Tuple[str, ...]:
        ranked = {
            row["instrument"] for ranking in self._rankings.values()
            for row in ranking.rows if row["scored"]
        }
        held = {item["instrument"] for item in self._profile.bootstrap["positions"]}
        return tuple(sorted(ranked | held))

    def _state(self, arm: str) -> ShadowPortfolioState:
        portfolio_id = _deterministic_id("B2_PORTFOLIO_V1", {
            "profile_digest": self._profile.canonical_digest["value"],
            "stage_a_result_digest": self._stage_a_result_digest["value"],
            "anchor": self._anchor, "arm": arm,
        })
        return ShadowPortfolioState.from_dict({
            "portfolio_id": portfolio_id, "as_of_date": self._profile.bootstrap["as_of_date"],
            "cash": self._profile.bootstrap["cash"],
            "positions": [dict(item) for item in self._profile.bootstrap["positions"]],
        })

    def _cycle_id(self, arm: str) -> str:
        return _deterministic_id("B2_CYCLE_V1", {
            "profile_digest": self._profile.canonical_digest["value"],
            "stage_a_result_digest": self._stage_a_result_digest["value"],
            "anchor": self._anchor, "trade_date": self._trade_date, "arm": arm,
        })

    def prepare_arms(self, anchor_prices: CashPriceReceipt) -> PreparedCycle:
        if not isinstance(anchor_prices, CashPriceReceipt) or anchor_prices.observation_kind != "CASH_CLOSE":
            raise HistoricalCycleContractError("prepare_arms requires an anchor-close receipt")
        anchor_prices._validated_copy()
        if anchor_prices.observation_date != self._anchor or anchor_prices.requested_instruments != self.requested_anchor_instruments:
            raise HistoricalCycleContractError("anchor receipt identity/requested set mismatch")
        terminal = []
        for arm in ARM_IDS:
            ranking = self._rankings[arm]
            prior = self._state(arm)
            cycle_id = self._cycle_id(arm)
            requested = tuple(sorted(
                {row["instrument"] for row in ranking.rows if row["scored"]}
                | {item.instrument for item in prior.positions}
            ))
            arm_receipt_rows = [row for row in anchor_prices.rows if row["instrument"] in set(requested)]
            arm_receipt = CashPriceReceipt(
                _authority=_AUTHORITY, observation_kind="CASH_CLOSE", observation_date=self._anchor,
                calendar_position=anchor_prices.calendar_position, requested_instruments=requested,
                rows=tuple(arm_receipt_rows),
                observed_instruments=tuple(row["instrument"] for row in arm_receipt_rows if row["status"] == "OBSERVED"),
                missing_instruments=tuple(row["instrument"] for row in arm_receipt_rows if row["status"] == "MISSING"),
                invalid_instruments=tuple(row["instrument"] for row in arm_receipt_rows if row["status"] == "INVALID"),
                digest=MappingProxyType(_digest_payload({
                    "observation_kind": "CASH_CLOSE", "observation_date": self._anchor,
                    "calendar_position": anchor_prices.calendar_position,
                    "requested_instruments": list(requested), "rows": [dict(row) for row in arm_receipt_rows],
                    "observed_instruments": [row["instrument"] for row in arm_receipt_rows if row["status"] == "OBSERVED"],
                    "missing_instruments": [row["instrument"] for row in arm_receipt_rows if row["status"] == "MISSING"],
                    "invalid_instruments": [row["instrument"] for row in arm_receipt_rows if row["status"] == "INVALID"],
                    "counts": {
                        "requested": len(requested), "raw_sources": len(arm_receipt_rows) * 2,
                        "observed": sum(row["status"] == "OBSERVED" for row in arm_receipt_rows),
                        "missing": sum(row["status"] == "MISSING" for row in arm_receipt_rows),
                        "invalid": sum(row["status"] == "INVALID" for row in arm_receipt_rows),
                    },
                })),
            )
            try:
                planning = CurrentRuleShadowIntentPlanner.plan(
                    portfolio_id=prior.portfolio_id, cycle_id=cycle_id, market=self._market,
                    anchor_date=self._anchor, trade_date=self._trade_date,
                    definition=self._profile.definition, ranking=ranking, prior=prior,
                    prices=arm_receipt.to_anchor_snapshot(),
                )
                if planning.status != "COMPLETE" or planning.prior_state_digest != prior.digest:
                    raise HistoricalCycleContractError("B1 result did not preserve B2 identity")
                terminal.append(MappingProxyType({
                    "arm_id": arm, "terminal_status": "PREPARED", "ranking": ranking,
                    "ranking_digest": _ranking_digest(ranking), "prior_state": prior,
                    "anchor_price_receipt": arm_receipt, "intent_planning_result": planning,
                    "error": None,
                }))
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as exc:
                terminal.append(MappingProxyType({
                    "arm_id": arm, "terminal_status": "BLOCKED_PLANNING", "ranking": ranking,
                    "ranking_digest": _ranking_digest(ranking), "prior_state": prior,
                    "anchor_price_receipt": arm_receipt, "intent_planning_result": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }))
        status = "COMPLETE" if all(item["terminal_status"] == "PREPARED" for item in terminal) else "BLOCKED"
        return PreparedCycle(
            _authority=_AUTHORITY, requested_arm_ids=ARM_IDS, terminal_arms=tuple(terminal),
            anchor_prices=anchor_prices, status=status,
        )

    def settle_arms(
        self, prepared: PreparedCycle, next_open_receipts: Mapping[str, CashPriceReceipt],
    ) -> Tuple[Mapping[str, Any], ...]:
        if not isinstance(prepared, PreparedCycle) or prepared.status != "COMPLETE":
            raise HistoricalCycleContractError("settlement requires all five arms prepared")
        if not isinstance(next_open_receipts, Mapping) or tuple(next_open_receipts) != ARM_IDS:
            raise HistoricalCycleContractError("settlement requires exact ordered five next-open receipts")
        terminal = []
        for prepared_arm in prepared.terminal_arms:
            arm = prepared_arm["arm_id"]
            receipt = next_open_receipts[arm]
            prior = prepared_arm["prior_state"]
            planning = prepared_arm["intent_planning_result"]
            requested = tuple(sorted(
                {item.instrument for item in prior.positions}
                | {item.instrument for item in planning.intents.intents}
            ))
            try:
                if isinstance(receipt, CashPriceReceipt):
                    receipt._validated_copy()
                if (
                    not isinstance(receipt, CashPriceReceipt)
                    or receipt.observation_kind != "NEXT_OPEN"
                    or receipt.observation_date != self._trade_date
                    or receipt.requested_instruments != requested
                ):
                    raise HistoricalCycleContractError("arm next-open receipt identity mismatch")
                quotes = receipt.to_quote_snapshot(portfolio_id=prior.portfolio_id, cycle_id=self._cycle_id(arm))
                transition = ShadowPortfolioTransition.apply(
                    prior=ShadowPortfolioState.from_dict(prior._payload()),
                    intents=planning.intents._validated_copy(), quotes=quotes,
                    assumption=ExecutionAssumption.from_dict(self._profile.assumption._payload()),
                )
                filled = tuple(item for item in transition.terminal_results if item.status == "FILLED")
                rounding_bound = Decimal(transition.filled_count + 3) * self._profile.assumption.money_quantum / Decimal(2)
                rounding_ok = rounding_adjustment_within_bound(
                    transition.rounding_adjustment, transition.filled_count,
                    self._profile.assumption.money_quantum,
                )
                sell_net_ok = all(
                    item.side != "SELL" or item.status != "FILLED" or item.fee < item.gross
                    for item in transition.terminal_results
                )
                status = "COMPLETE"
                if not rounding_ok:
                    status = "BLOCKED_ROUNDING_ADJUSTMENT_OUT_OF_BOUND"
                elif not sell_net_ok:
                    status = "BLOCKED_EXECUTION_PROFILE_NONPOSITIVE_SELL_PROCEEDS"
                terminal.append(MappingProxyType({
                    **dict(prepared_arm), "terminal_status": status,
                    "next_open_receipt": receipt, "quotes": quotes, "transition_result": transition,
                    "after_state": transition.after_state if status == "COMPLETE" else None,
                    "rounding_bound": _decimal_text(rounding_bound),
                    "rounding_adjustment_checked": rounding_ok,
                    "positive_sell_net_proceeds_checked": sell_net_ok,
                    "filled_count_for_gate": len(filled), "error": None,
                }))
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as exc:
                terminal.append(MappingProxyType({
                    **dict(prepared_arm), "terminal_status": "BLOCKED_SETTLEMENT",
                    "next_open_receipt": receipt if isinstance(receipt, CashPriceReceipt) else None,
                    "quotes": None, "transition_result": None, "after_state": None,
                    "rounding_bound": None, "rounding_adjustment_checked": False,
                    "positive_sell_net_proceeds_checked": False,
                    "filled_count_for_gate": 0,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }))
        return tuple(terminal)

    @staticmethod
    def _arm_payload(item: Mapping[str, Any]) -> Dict[str, Any]:
        result = {
            "arm_id": item["arm_id"], "terminal_status": item["terminal_status"],
            "ranking_digest": item["ranking_digest"], "prior_state": item["prior_state"].to_dict(),
            "anchor_price_receipt": item["anchor_price_receipt"].to_dict(),
            "intent_planning_result": None if item["intent_planning_result"] is None else item["intent_planning_result"].to_dict(),
            "error": item["error"],
        }
        if "next_open_receipt" in item:
            result.update({
                "next_open_receipt": None if item["next_open_receipt"] is None else item["next_open_receipt"].to_dict(),
                "quotes": None if item["quotes"] is None else item["quotes"].to_dict(),
                "transition_result": None if item["transition_result"] is None else item["transition_result"].to_dict(),
                "after_state": None if item["after_state"] is None else item["after_state"].to_dict(),
                "rounding_bound": item["rounding_bound"],
                "rounding_adjustment_checked": item["rounding_adjustment_checked"],
                "positive_sell_net_proceeds_checked": item["positive_sell_net_proceeds_checked"],
            })
        return result

    @staticmethod
    def _comparison(champion: Mapping[str, Any], challenger: Mapping[str, Any]) -> Mapping[str, Any]:
        champ_plan = champion["intent_planning_result"]
        arm_plan = challenger["intent_planning_result"]
        champ_transition = champion["transition_result"]
        arm_transition = challenger["transition_result"]
        def ids(plan: Any, side: str) -> set:
            return {item.instrument for item in plan.intents.intents if item.side == side}
        def holding_ids(transition: Any) -> set:
            return {(item.instrument, item.quantity) for item in transition.after_state.positions}
        champ_sell, arm_sell = ids(champ_plan, "SELL"), ids(arm_plan, "SELL")
        champ_buy, arm_buy = ids(champ_plan, "BUY"), ids(arm_plan, "BUY")
        champ_hold, arm_hold = holding_ids(champ_transition), holding_ids(arm_transition)
        valuation_comparable = champ_transition.valuation_status == arm_transition.valuation_status == "COMPLETE"
        champ_status = {item.instrument: item.status for item in champ_transition.terminal_results}
        arm_status = {item.instrument: item.status for item in arm_transition.terminal_results}
        status_differences = [
            {"instrument": instrument, "champion": champ_status.get(instrument, "ABSENT"),
             "challenger": arm_status.get(instrument, "ABSENT")}
            for instrument in sorted(set(champ_status) | set(arm_status))
            if champ_status.get(instrument, "ABSENT") != arm_status.get(instrument, "ABSENT")
        ]
        return MappingProxyType({
            "champion_arm_id": "CHAMPION_4", "challenger_arm_id": challenger["arm_id"],
            "sell_overlap": sorted(champ_sell & arm_sell), "sell_entered": sorted(arm_sell - champ_sell),
            "sell_exited": sorted(champ_sell - arm_sell),
            "selected_buy_overlap": sorted(champ_buy & arm_buy), "selected_buy_entered": sorted(arm_buy - champ_buy),
            "selected_buy_exited": sorted(champ_buy - arm_buy),
            "filled_status_differences": status_differences,
            "after_holding_overlap": [list(item) for item in sorted(champ_hold & arm_hold)],
            "after_holding_entered": [list(item) for item in sorted(arm_hold - champ_hold)],
            "after_holding_exited": [list(item) for item in sorted(champ_hold - arm_hold)],
            "after_cash_difference": _decimal_text(arm_transition.after_state.cash - champ_transition.after_state.cash),
            "total_fee_difference": _decimal_text(arm_transition.total_fee - champ_transition.total_fee),
            "slippage_cost_difference": _decimal_text(arm_transition.slippage_cost - champ_transition.slippage_cost),
            "valuation_comparison_status": "COMPARABLE" if valuation_comparable else "NOT_COMPARABLE_MISSING_VALUATION",
            "pre_trade_next_open_nav_difference": _decimal_text(arm_transition.nav_before - champ_transition.nav_before) if valuation_comparable else None,
            "post_trade_next_open_nav_difference": _decimal_text(arm_transition.nav_after - champ_transition.nav_after) if valuation_comparable else None,
            "champion_transition_digest": champ_transition.result_digest,
            "challenger_transition_digest": arm_transition.result_digest,
        })

    def run(self) -> HistoricalCycleResult:
        if _source_fingerprint(self._engine_root) != self._source:
            raise HistoricalCycleInputError("engine source identity changed during logical replay")
        if _environment_fingerprint() != self._environment:
            raise HistoricalCycleInputError("runtime environment changed during logical replay")
        profile = self._profile.revalidate()
        if profile.canonical_digest != self._profile.canonical_digest:
            raise HistoricalCycleInputError("private profile content changed")
        anchor_receipt = self._observer.observe_anchor_close(
            anchor_date=self._anchor, requested_instruments=self.requested_anchor_instruments,
        )
        prepared = self.prepare_arms(anchor_receipt)
        if prepared.status != "COMPLETE":
            arms = prepared.terminal_arms
            comparisons: Tuple[Mapping[str, Any], ...] = ()
            receipts = {}
        else:
            receipts: Dict[str, CashPriceReceipt] = {}
            for item in prepared.terminal_arms:
                requested = tuple(sorted(
                    {position.instrument for position in item["prior_state"].positions}
                    | {intent.instrument for intent in item["intent_planning_result"].intents.intents}
                ))
                receipts[item["arm_id"]] = self._observer.observe_next_open(
                    trade_date=self._trade_date, requested_instruments=requested,
                )
            arms = self.settle_arms(prepared, receipts)
            comparisons = tuple(
                self._comparison(arms[0], item) for item in arms[1:]
                if arms[0]["terminal_status"] == item["terminal_status"] == "COMPLETE"
            )
        complete_count = sum(item["terminal_status"] == "COMPLETE" for item in arms)
        economic_states = {
            _canonical_json({
                "as_of_date": item["prior_state"].as_of_date,
                "cash": _decimal_text(item["prior_state"].cash),
                "positions": [position._raw() for position in item["prior_state"].positions],
            })
            for item in arms
        }
        portfolio_ids = {item["prior_state"].portfolio_id for item in arms}
        independence = len(economic_states) == 1 and len(portfolio_ids) == len(ARM_IDS)
        lot_compatible = profile.definition.production_buy_lot_size == profile.assumption.lot_size
        status = "COMPLETE" if (
            complete_count == len(ARM_IDS) and len(comparisons) == 4
            and independence and lot_compatible
        ) else "BLOCKED"
        payload: Dict[str, Any] = {
            "schema_version": 1, "status": status, "evidence_class": EVIDENCE_CLASS,
            "prospective_claim": False, "promotion_capability": False,
            "warnings": list(WARNING_LINES),
            "operation_identity": {
                "profile_id": profile.profile_id, "anchor_date": self._anchor,
                "trade_date": self._trade_date, "market": self._market,
            },
            "anchor_date": self._anchor, "trade_date": self._trade_date,
            "stage_a_result_digest": self._stage_a_result_digest,
            **self._stage_a_public,
            "engine_source_fingerprint": dict(self._source),
            "environment_fingerprint": dict(self._environment),
            "profile": profile.public_payload(),
            "intent_definition_digest": profile.definition.digest,
            "execution_assumption_digest": profile.assumption.digest,
            "anchor_price_receipt": anchor_receipt.to_dict(),
            "next_open_receipts": {
                arm: receipts[arm].to_dict() for arm in ARM_IDS if arm in receipts
            },
            "requested_arm_ids": list(ARM_IDS),
            "terminal_arms": [self._arm_payload(item) for item in arms],
            "arm_counts": {
                "requested": len(ARM_IDS), "terminal": len(arms),
                "complete": complete_count, "blocked": len(arms) - complete_count,
            },
            "state_independence_checked": independence,
            "lot_compatibility_checked": lot_compatible,
            "comparisons": [dict(item) for item in comparisons],
            "comparison_capability": status == "COMPLETE",
        }
        payload["result_digest"] = _digest_payload(payload)
        return HistoricalCycleResult(payload, _AUTHORITY)


@dataclass(frozen=True, init=False)
class PublicationReceipt:
    operation_id: str
    status: str
    did_write: bool
    result_digest: Optional[Mapping[str, Any]]
    manifest_digest: Optional[Mapping[str, Any]]
    member_count: int
    output_root_identity: Optional[Tuple[int, int, int, int, int]]
    root_parent_identity_before: Tuple[int, int, int, int, int]
    root_parent_identity_after: Optional[Tuple[int, int, int, int, int]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _AUTHORITY or args:
            raise HistoricalCycleContractError("publication receipts are writer-owned")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        if self.status not in ("COMMITTED", "CONFLICT", "UNCERTAIN"):
            raise HistoricalCycleContractError("publication receipt status is invalid")
        if self.status == "COMMITTED" and (
            self.did_write is not True or self.result_digest is None
            or self.manifest_digest is None or self.member_count <= 0
            or self.output_root_identity is None
            or self.root_parent_identity_after != self.root_parent_identity_before
        ):
            raise HistoricalCycleContractError("committed publication receipt is inconsistent")
        if self.status == "CONFLICT" and (
            self.did_write is not False or self.manifest_digest is not None
            or self.member_count != 0 or self.output_root_identity is not None
            or self.root_parent_identity_after != self.root_parent_identity_before
        ):
            raise HistoricalCycleContractError("conflict publication receipt is inconsistent")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id, "status": self.status, "did_write": self.did_write,
            "result_digest": None if self.result_digest is None else dict(self.result_digest),
            "manifest_digest": None if self.manifest_digest is None else dict(self.manifest_digest),
            "member_count": self.member_count,
            "output_root_identity": None if self.output_root_identity is None else list(self.output_root_identity),
            "root_parent_identity_before": list(self.root_parent_identity_before),
            "root_parent_identity_after": (
                None if self.root_parent_identity_after is None
                else list(self.root_parent_identity_after)
            ),
        }


def _summary_csv(result: Mapping[str, Any]) -> bytes:
    stream = io.StringIO(newline="")
    fields = ("challenger_arm_id", "sell_overlap_count", "selected_buy_overlap_count", "after_cash_difference", "total_fee_difference", "slippage_cost_difference", "valuation_comparison_status", "pre_trade_next_open_nav_difference", "post_trade_next_open_nav_difference")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in result["comparisons"]:
        writer.writerow({
            "challenger_arm_id": item["challenger_arm_id"],
            "sell_overlap_count": len(item["sell_overlap"]),
            "selected_buy_overlap_count": len(item["selected_buy_overlap"]),
            "after_cash_difference": item["after_cash_difference"],
            "total_fee_difference": item["total_fee_difference"],
            "slippage_cost_difference": item["slippage_cost_difference"],
            "valuation_comparison_status": item["valuation_comparison_status"],
            "pre_trade_next_open_nav_difference": item["pre_trade_next_open_nav_difference"],
            "post_trade_next_open_nav_difference": item["post_trade_next_open_nav_difference"],
        })
    return stream.getvalue().encode("utf-8")


def _report(result: Mapping[str, Any]) -> bytes:
    lines = ["# Stage B2 Single Historical Cycle Shadow Replay", ""]
    lines.extend("- %s" % item for item in WARNING_LINES)
    lines.extend([
        "", "- Status: `%s`" % result["status"],
        "- Anchor / trade date: `%s` / `%s`" % (result["anchor_date"], result["trade_date"]),
        "- Requested / complete arms: `%d` / `%d`" % (result["arm_counts"]["requested"], result["arm_counts"]["complete"]),
        "- Result digest: `%s`" % result["result_digest"]["value"], "",
    ])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _artifact_members(runner: HistoricalShadowCycleReplay, result: HistoricalCycleResult) -> Mapping[str, bytes]:
    payload = result.to_dict()
    members: Dict[str, bytes] = {
        "input_receipt.json": _canonical_json({
            "operation_identity": payload["operation_identity"],
            "stage_a_result_digest": payload["stage_a_result_digest"],
            "engine_source_fingerprint": payload["engine_source_fingerprint"],
            "environment_fingerprint": payload["environment_fingerprint"],
            "profile": payload["profile"], "anchor_price_receipt": payload["anchor_price_receipt"],
        }),
        "result.json": result.to_canonical_json_bytes(),
        "summary.csv": _summary_csv(payload), "report.md": _report(payload),
    }
    by_arm = {item["arm_id"]: item for item in payload["terminal_arms"]}
    for arm in ARM_IDS:
        item = by_arm[arm]
        prefix = "arms/%s/" % arm
        members[prefix + "ranking.csv"] = runner._rankings[arm].to_csv_bytes()
        members[prefix + "intent_plan.json"] = _canonical_json(item["intent_planning_result"])
        members[prefix + "quotes.json"] = _canonical_json(item["quotes"])
        members[prefix + "settlement.json"] = _canonical_json(item["transition_result"])
        members[prefix + "after_state.json"] = _canonical_json(item["after_state"])
    return MappingProxyType({key: members[key] for key in sorted(members)})


def write_historical_cycle_output(
    runner: HistoricalShadowCycleReplay, result: HistoricalCycleResult, output_root: Path,
) -> PublicationReceipt:
    """Recompute a COMPLETE result, then create and verify one new /tmp bundle."""
    if not isinstance(runner, HistoricalShadowCycleReplay) or not isinstance(result, HistoricalCycleResult):
        raise HistoricalCycleContractError("writer requires replay-owned runner and result")
    operation_id = _deterministic_id("B2_PUBLICATION_V1", {
        "result_digest": result["result_digest"]["value"], "output_name": Path(output_root).name,
    })
    fresh = runner.run()
    if fresh.to_canonical_json_bytes() != result.to_canonical_json_bytes() or fresh["status"] != "COMPLETE":
        raise HistoricalCycleContractError("publication revalidation failed before write")
    root = Path(output_root)
    tmp = Path("/tmp").resolve(strict=True)
    if not root.is_absolute() or root.parent.resolve(strict=True) != root.parent:
        raise HistoricalCycleContractError("output root must use a canonical physical parent")
    parent = root.parent.resolve(strict=True)
    try:
        parent.relative_to(tmp)
    except ValueError as exc:
        raise HistoricalCycleContractError("output root must be physically below /tmp") from exc
    if root.parent / root.name != root or not root.name or root.name in (".", ".."):
        raise HistoricalCycleContractError("output root must be a canonical direct path")
    parent_before = _root_identity(parent)
    if root.exists() or root.is_symlink():
        return PublicationReceipt(
            _authority=_AUTHORITY, operation_id=operation_id, status="CONFLICT", did_write=False,
            result_digest=result["result_digest"], manifest_digest=None, member_count=0,
            output_root_identity=None, root_parent_identity_before=parent_before,
            root_parent_identity_after=_try_root_identity(parent),
        )
    members = _artifact_members(runner, fresh)
    directory_fds = []
    did_write = False
    root_identity = None
    written_count = 0
    try:
        try:
            os.mkdir(str(root), 0o700)
        except FileExistsError:
            return PublicationReceipt(
                _authority=_AUTHORITY, operation_id=operation_id, status="CONFLICT", did_write=False,
                result_digest=result["result_digest"], manifest_digest=None, member_count=0,
                output_root_identity=None, root_parent_identity_before=parent_before,
                root_parent_identity_after=_try_root_identity(parent),
            )
        did_write = True
        root_identity = _root_identity(root)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(str(root), directory_flags)
        directory_fds.append(root_fd)
        os.mkdir("arms", 0o700, dir_fd=root_fd)
        arms_fd = os.open("arms", directory_flags, dir_fd=root_fd)
        directory_fds.append(arms_fd)
        arm_fds = {}
        directory_identities = {".": _root_identity(root), "arms": _root_identity(root / "arms")}
        for arm in ARM_IDS:
            os.mkdir(arm, 0o700, dir_fd=arms_fd)
            arm_fd = os.open(arm, directory_flags, dir_fd=arms_fd)
            directory_fds.append(arm_fd)
            arm_fds[arm] = arm_fd
            directory_identities["arms/" + arm] = _root_identity(root / "arms" / arm)
        manifest_rows = []
        for logical, data in members.items():
            parts = Path(logical).parts
            if len(parts) == 1:
                parent_fd, name = root_fd, parts[0]
            elif len(parts) == 3 and parts[0] == "arms" and parts[1] in arm_fds:
                parent_fd, name = arm_fds[parts[1]], parts[2]
            else:
                raise HistoricalCycleContractError("publication member path is outside the frozen layout")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            file_fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
            with os.fdopen(file_fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            observed = _stable_read(root, logical)
            if observed != data:
                raise HistoricalCycleInputError("publication member verification failed")
            manifest_rows.append({"logical_path": logical, "digest": _digest_bytes(data)})
            written_count += 1
        for descriptor in reversed(directory_fds):
            os.fsync(descriptor)
        manifest_payload = {
            "schema_version": 1, "result_digest": fresh["result_digest"],
            "member_count": len(manifest_rows), "members": manifest_rows,
        }
        manifest = _canonical_json(manifest_payload)
        manifest_fd = os.open(
            "output_manifest.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600, dir_fd=root_fd,
        )
        with os.fdopen(manifest_fd, "wb") as handle:
            handle.write(manifest)
            handle.flush()
            os.fsync(handle.fileno())
        if _stable_read(root, "output_manifest.json") != manifest:
            raise HistoricalCycleInputError("manifest verification failed")
        for item in manifest_rows:
            data = _stable_read(root, item["logical_path"])
            if data is None or _digest_bytes(data) != item["digest"]:
                raise HistoricalCycleInputError("final member verification failed")
        for logical, identity in directory_identities.items():
            path = root if logical == "." else root.joinpath(*Path(logical).parts)
            if _root_identity(path) != identity:
                raise HistoricalCycleInputError("publication directory identity changed")
        if _root_identity(root) != root_identity or _root_identity(parent) != parent_before:
            raise HistoricalCycleInputError("publication namespace identity changed")
        os.fsync(root_fd)
        return PublicationReceipt(
            _authority=_AUTHORITY, operation_id=operation_id, status="COMMITTED", did_write=True,
            result_digest=fresh["result_digest"], manifest_digest=_digest_bytes(manifest),
            member_count=len(manifest_rows), output_root_identity=root_identity,
            root_parent_identity_before=parent_before, root_parent_identity_after=_try_root_identity(parent),
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        current_root_identity = _try_root_identity(root)
        return PublicationReceipt(
            _authority=_AUTHORITY, operation_id=operation_id, status="UNCERTAIN", did_write=did_write,
            result_digest=result["result_digest"], manifest_digest=None, member_count=written_count,
            output_root_identity=(
                root_identity if current_root_identity == root_identity else None
            ), root_parent_identity_before=parent_before,
            root_parent_identity_after=_try_root_identity(parent),
        )
    finally:
        for descriptor in reversed(directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
