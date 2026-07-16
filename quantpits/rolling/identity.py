"""Pure canonical identities for Rolling plans, targets, and windows."""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quantpits.rolling.errors import RollingIdentityError
from quantpits.utils.workspace import fingerprint_value


ROLLING_FAMILIES = ("rolling", "cpcv_rolling")
ROLLING_ACTIONS = (
    "cold_start", "merge", "resume", "daily", "retrain_models",
    "retrain_last", "predict_only", "backtest_only", "clear_state",
    "show_state",
)
_METHOD_TO_FAMILY = {"slide": "rolling", "cpcv": "cpcv_rolling"}
_FAMILY_TO_METHOD = {value: key for key, value in _METHOD_TO_FAMILY.items()}
_WINDOW_KEY_RE = re.compile(
    r"^(rolling|cpcv_rolling):(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2}):([0-9a-f]{12})$"
)


def _invalid(message):
    raise RollingIdentityError(message)


def family_for_training_method(method):
    if method not in _METHOD_TO_FAMILY:
        _invalid("unsupported Rolling training method: %r" % (method,))
    return _METHOD_TO_FAMILY[method]


def training_method_for_family(family):
    if family not in _FAMILY_TO_METHOD:
        _invalid("unsupported Rolling family: %r" % (family,))
    return _FAMILY_TO_METHOD[family]


def normalize_iso_date(value, field="date"):
    if not isinstance(value, str):
        _invalid("%s must be an ISO YYYY-MM-DD string" % field)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        _invalid("%s must be a real ISO YYYY-MM-DD date" % field)
    normalized = parsed.isoformat()
    if value != normalized:
        _invalid("%s must use canonical ISO YYYY-MM-DD form" % field)
    return normalized


def workspace_fingerprint(workspace_root):
    """Fingerprint resolved root identity without exposing it in public data."""

    root = str(Path(workspace_root).expanduser().resolve()).encode("utf-8")
    return hashlib.sha256(root).hexdigest()


def parse_rolling_window_key(window_key, expected_family=None):
    match = _WINDOW_KEY_RE.match(window_key) if isinstance(window_key, str) else None
    if match is None:
        _invalid("Rolling window key is not canonical")
    family, test_start, test_end, digest = match.groups()
    if expected_family is not None and family != expected_family:
        _invalid("Rolling window key family does not match its envelope")
    test_start = normalize_iso_date(test_start, "window_test_start")
    test_end = normalize_iso_date(test_end, "window_test_end")
    if test_start > test_end:
        _invalid("Rolling window key test range is reversed")
    canonical = "%s:%s:%s:%s" % (family, test_start, test_end, digest)
    if canonical != window_key:
        _invalid("Rolling window key is not canonical")
    return family, test_start, test_end, digest


def _validate_digest(value, field):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        _invalid("%s must be a lowercase SHA-256 digest" % field)
    return value


@dataclass(frozen=True)
class RollingTargetIdentity:
    model_name: str
    family: str

    def __post_init__(self):
        if self.family not in ROLLING_FAMILIES:
            _invalid("unsupported Rolling family: %r" % (self.family,))
        name = self.model_name
        if not isinstance(name, str) or not name or name != name.strip():
            _invalid("Rolling model name must be a non-empty trimmed string")
        if ("@" in name or "\x00" in name or "/" in name or "\\" in name
                or any(ord(char) < 32 or ord(char) == 127 for char in name)):
            _invalid("Rolling model name contains a reserved character")

    @property
    def target_key(self):
        return "%s@%s" % (self.model_name, self.family)

    @classmethod
    def parse(cls, target_key):
        if not isinstance(target_key, str) or target_key.count("@") != 1:
            _invalid("Rolling target key must be model@family")
        model_name, family = target_key.split("@")
        identity = cls(model_name=model_name, family=family)
        if identity.target_key != target_key:
            _invalid("Rolling target key is not canonical")
        return identity

    def to_public_dict(self):
        return {
            "model_name": self.model_name,
            "family": self.family,
            "target_key": self.target_key,
        }


@dataclass(frozen=True)
class RollingFoldIdentity:
    train_segments: tuple
    valid_start: str
    valid_end: str

    def __post_init__(self):
        valid_start = normalize_iso_date(self.valid_start, "valid_start")
        valid_end = normalize_iso_date(self.valid_end, "valid_end")
        if valid_start > valid_end:
            _invalid("fold valid range is reversed")
        normalized = []
        for index, segment in enumerate(self.train_segments):
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                _invalid("fold train segment %s must contain start/end" % index)
            start = normalize_iso_date(segment[0], "train_segment_start")
            end = normalize_iso_date(segment[1], "train_segment_end")
            if start > end:
                _invalid("fold train segment is reversed")
            if not (end < valid_start or start > valid_end):
                _invalid("fold train segment overlaps its validation range")
            normalized.append((start, end))
        normalized.sort()
        if not normalized:
            _invalid("CPCV fold must contain at least one train segment")
        if len(normalized) != len(set(normalized)):
            _invalid("CPCV fold contains duplicate train segments")
        for previous, current in zip(normalized, normalized[1:]):
            if current[0] <= previous[1]:
                _invalid("CPCV fold train segments overlap")
        object.__setattr__(self, "train_segments", tuple(normalized))
        object.__setattr__(self, "valid_start", valid_start)
        object.__setattr__(self, "valid_end", valid_end)

    @property
    def fold_key(self):
        return fingerprint_value(self.to_fingerprint_dict())

    def to_fingerprint_dict(self):
        return {
            "family": "cpcv_rolling",
            "train_segments": [list(item) for item in self.train_segments],
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
        }

    def to_public_dict(self):
        payload = self.to_fingerprint_dict()
        payload["fold_key"] = self.fold_key
        return payload


@dataclass(frozen=True)
class RollingWindowIdentity:
    family: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    effective_config_fingerprint: str
    valid_start: str = None
    valid_end: str = None
    folds: tuple = ()

    def __post_init__(self):
        if self.family not in ROLLING_FAMILIES:
            _invalid("unsupported Rolling window family")
        train_start = normalize_iso_date(self.train_start, "train_start")
        train_end = normalize_iso_date(self.train_end, "train_end")
        test_start = normalize_iso_date(self.test_start, "test_start")
        test_end = normalize_iso_date(self.test_end, "test_end")
        if train_start > train_end or test_start > test_end:
            _invalid("Rolling window contains a reversed date range")
        _validate_digest(
            self.effective_config_fingerprint, "effective_config_fingerprint",
        )
        folds = tuple(self.folds)
        if self.family == "rolling":
            if self.valid_start is None or self.valid_end is None:
                _invalid("slide window requires a validation range")
            valid_start = normalize_iso_date(self.valid_start, "valid_start")
            valid_end = normalize_iso_date(self.valid_end, "valid_end")
            if valid_start > valid_end:
                _invalid("slide validation range is reversed")
            if not (train_end < valid_start and valid_end < test_start):
                _invalid("slide train/valid/test ranges must be ordered and disjoint")
            if folds:
                _invalid("slide window cannot contain CPCV folds")
        else:
            if self.valid_start is not None or self.valid_end is not None:
                _invalid("CPCV outer window cannot contain a slide validation range")
            valid_start = valid_end = None
            if not folds or any(not isinstance(item, RollingFoldIdentity) for item in folds):
                _invalid("CPCV window requires typed folds")
            fold_keys = [item.fold_key for item in folds]
            if len(fold_keys) != len(set(fold_keys)):
                _invalid("CPCV window contains duplicate fold identities")
            if train_end >= test_start:
                _invalid("CPCV train/test ranges must be ordered and disjoint")
            for fold in folds:
                if not (train_start <= fold.valid_start <= fold.valid_end <= train_end):
                    _invalid("CPCV fold validation range is outside the train domain")
                if any(
                        start < train_start or end > train_end
                        for start, end in fold.train_segments):
                    _invalid("CPCV fold train segment is outside the train domain")
        object.__setattr__(self, "train_start", train_start)
        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "test_start", test_start)
        object.__setattr__(self, "test_end", test_end)
        object.__setattr__(self, "valid_start", valid_start)
        object.__setattr__(self, "valid_end", valid_end)
        object.__setattr__(self, "folds", folds)

    @property
    def fold_keys(self):
        return tuple(item.fold_key for item in self.folds)

    @property
    def content_fingerprint(self):
        return fingerprint_value(self.to_fingerprint_dict())

    @property
    def window_key(self):
        return "%s:%s:%s:%s" % (
            self.family, self.test_start, self.test_end,
            self.content_fingerprint[:12],
        )

    def to_fingerprint_dict(self):
        return {
            "family": self.family,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "fold_keys": list(self.fold_keys),
            "effective_config_fingerprint": self.effective_config_fingerprint,
        }

    def to_public_dict(self):
        payload = self.to_fingerprint_dict()
        payload["window_key"] = self.window_key
        payload["folds"] = [item.to_public_dict() for item in self.folds]
        return payload


@dataclass(frozen=True)
class RollingRunIdentity:
    workspace_fingerprint: str
    family: str
    action: str
    plan_fingerprint: str
    config_fingerprint: str
    anchor_date: str
    target_keys: tuple
    window_keys: tuple
    runtime_params_fingerprint: str
    attempt_id: str = None

    def __post_init__(self):
        for field in (
                "workspace_fingerprint", "plan_fingerprint",
                "config_fingerprint", "runtime_params_fingerprint"):
            _validate_digest(getattr(self, field), field)
        if self.family not in ROLLING_FAMILIES:
            _invalid("unsupported Rolling run family")
        if self.action not in ROLLING_ACTIONS:
            _invalid("unsupported Rolling action: %r" % (self.action,))
        object.__setattr__(
            self, "anchor_date", normalize_iso_date(self.anchor_date, "anchor_date"),
        )
        targets = tuple(self.target_keys)
        for key in targets:
            identity = RollingTargetIdentity.parse(key)
            if identity.family != self.family:
                _invalid("run target family does not match run family")
        windows = tuple(self.window_keys)
        if len(targets) != len(set(targets)) or len(windows) != len(set(windows)):
            _invalid("run target/window identities must be unique")
        object.__setattr__(self, "target_keys", targets)
        object.__setattr__(self, "window_keys", windows)

    @property
    def fingerprint(self):
        return fingerprint_value({
            "workspace_fingerprint": self.workspace_fingerprint,
            "family": self.family,
            "action": self.action,
            "plan_fingerprint": self.plan_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "anchor_date": self.anchor_date,
            "target_keys": list(self.target_keys),
            "window_keys": list(self.window_keys),
            "runtime_params_fingerprint": self.runtime_params_fingerprint,
        })

    def to_public_dict(self):
        return {
            "workspace_fingerprint": self.workspace_fingerprint,
            "family": self.family,
            "action": self.action,
            "plan_fingerprint": self.plan_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "anchor_date": self.anchor_date,
            "target_keys": list(self.target_keys),
            "window_keys": list(self.window_keys),
            "execution_fingerprint": self.fingerprint,
            "attempt_id": self.attempt_id,
        }
