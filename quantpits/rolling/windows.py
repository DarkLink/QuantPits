"""Runtime-only Rolling calendar resolution and stable window identities."""

import copy
import hashlib
import json
from dataclasses import InitVar, dataclass, field

from quantpits.rolling.errors import RollingIdentityError, RollingWindowResolutionError
from quantpits.rolling.identity import (
    RollingFoldIdentity,
    RollingRunIdentity,
    RollingWindowIdentity,
)
from quantpits.utils.workspace import fingerprint_value


_EXECUTION_WINDOW_TOKEN = object()


def _canonical_sessions(values, test_start, test_end):
    from quantpits.rolling.identity import normalize_iso_date

    if isinstance(values, (str, bytes)):
        raise RollingWindowResolutionError("business sessions must be an ordered sequence")
    try:
        sessions = tuple(normalize_iso_date(str(item)[:10], "business_session") for item in values)
    except (TypeError, RollingIdentityError) as exc:
        raise RollingWindowResolutionError("business sessions are invalid: %s" % exc)
    if not sessions or sessions != tuple(sorted(set(sessions))):
        raise RollingWindowResolutionError("business sessions must be non-empty, unique, and increasing")
    if sessions[0] != test_start or sessions[-1] != test_end:
        raise RollingWindowResolutionError("business sessions do not match window boundaries")
    return sessions


@dataclass(frozen=True)
class RollingWindowExecutionDescriptor:
    """One canonical window plus the calendar sessions actually observed for it."""

    window: "RollingWindowDescriptor"
    expected_sessions: tuple
    _authority: InitVar[object] = None
    _observer_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority):
        if _authority is not _EXECUTION_WINDOW_TOKEN:
            raise RollingWindowResolutionError(
                "execution windows are calendar-observer-owned"
            )
        if not isinstance(self.window, RollingWindowDescriptor):
            raise RollingWindowResolutionError("execution window requires a resolved descriptor")
        object.__setattr__(self, "expected_sessions", _canonical_sessions(
            self.expected_sessions, self.window.test_start, self.window.test_end,
        ))
        object.__setattr__(self, "_observer_authority", True)

    @property
    def identity(self):
        return self.window.identity

    @property
    def window_key(self):
        return self.window.window_key

    @property
    def sessions_fingerprint(self):
        return fingerprint_value({
            "window_key": self.window_key,
            "expected_sessions": list(self.expected_sessions),
        })

    def to_public_dict(self):
        payload = self.window.to_public_dict()
        payload.update({
            "expected_sessions": list(self.expected_sessions),
            "sessions_fingerprint": self.sessions_fingerprint,
        })
        return payload


def observe_rolling_business_sessions(windows, calendar_provider=None):
    """Observe exact Qlib business sessions for an ordered resolved-window tuple.

    ``calendar_provider`` is injectable for deterministic contract tests.  The
    default imports Qlib only when this runtime function is called.
    """

    if not isinstance(windows, tuple):
        raise RollingWindowResolutionError("windows must be an ordered tuple")
    if calendar_provider is None:
        from qlib.data import D

        calendar_provider = lambda start, end: D.calendar(start_time=start, end_time=end, freq="day")
    descriptors = []
    for window in windows:
        if not isinstance(window, RollingWindowDescriptor):
            raise RollingWindowResolutionError("windows contain a foreign descriptor")
        try:
            observed = calendar_provider(window.test_start, window.test_end)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            raise RollingWindowResolutionError(
                "business calendar observation failed: %s" % exc.__class__.__name__
            )
        descriptors.append(RollingWindowExecutionDescriptor(
            window, tuple(observed), _authority=_EXECUTION_WINDOW_TOKEN,
        ))
    return tuple(descriptors)


def rolling_sessions_fingerprint(windows):
    if not isinstance(windows, tuple) or any(
        not isinstance(item, RollingWindowExecutionDescriptor) for item in windows
    ):
        raise RollingWindowResolutionError("session fingerprint requires execution descriptors")
    return fingerprint_value([
        {"window_key": item.window_key, "expected_sessions": list(item.expected_sessions)}
        for item in windows
    ])


@dataclass(frozen=True)
class RollingWindowDescriptor:
    display_index: int
    identity: RollingWindowIdentity
    _legacy_window: dict = None

    @property
    def window_key(self):
        return self.identity.window_key

    @property
    def family(self):
        return self.identity.family

    @property
    def train_start(self):
        return self.identity.train_start

    @property
    def train_end(self):
        return self.identity.train_end

    @property
    def valid_start(self):
        return self.identity.valid_start

    @property
    def valid_end(self):
        return self.identity.valid_end

    @property
    def test_start(self):
        return self.identity.test_start

    @property
    def test_end(self):
        return self.identity.test_end

    @property
    def fold_fingerprint(self):
        if not self.identity.fold_keys:
            return None
        return fingerprint_value(list(self.identity.fold_keys))

    def to_legacy_window(self):
        """Return an isolated copy for the mutable legacy execution code."""

        return copy.deepcopy(self._legacy_window)

    def to_fingerprint_dict(self):
        return self.identity.to_fingerprint_dict()

    def to_public_dict(self):
        payload = self.identity.to_public_dict()
        payload["display_index"] = self.display_index
        return payload


@dataclass(frozen=True)
class ResolvedRollingRun:
    prepared: object
    actual_anchor: str
    params: dict
    windows: tuple
    identity: RollingRunIdentity
    runtime_source_policy: str = "qlib_trading_calendar"

    @property
    def execution_fingerprint(self):
        return self.identity.fingerprint

    @property
    def legacy_windows(self):
        return tuple(item.to_legacy_window() for item in self.windows)


def _descriptor(family, window, effective_config_fingerprint):
    try:
        index = int(window["window_idx"])
        train_start = str(window["train_start"])
        train_end = str(window["train_end"])
        test_start = str(window["test_start"])
        test_end = str(window["test_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RollingWindowResolutionError(
            "resolved window has an invalid shape: %s" % exc
        )
    folds = ()
    if family == "cpcv_rolling":
        raw_folds = window.get("cpcv_folds")
        if not isinstance(raw_folds, list) or not raw_folds:
            raise RollingWindowResolutionError(
                "CPCV resolved window %s has no folds" % index
            )
        try:
            folds = tuple(RollingFoldIdentity(
                train_segments=tuple(item["train_segments"]),
                valid_start=item["valid_start_time"],
                valid_end=item["valid_end_time"],
            ) for item in raw_folds)
        except (KeyError, TypeError, RollingIdentityError) as exc:
            raise RollingWindowResolutionError(
                "CPCV resolved window %s has invalid folds: %s" % (index, exc)
            )
    try:
        identity = RollingWindowIdentity(
            family=family,
            train_start=train_start,
            train_end=train_end,
            valid_start=window.get("valid_start"),
            valid_end=window.get("valid_end"),
            test_start=test_start,
            test_end=test_end,
            folds=folds,
            effective_config_fingerprint=effective_config_fingerprint,
        )
    except RollingIdentityError as exc:
        raise RollingWindowResolutionError(
            "resolved window %s has invalid identity: %s" % (index, exc)
        )
    return RollingWindowDescriptor(
        display_index=index,
        identity=identity,
        _legacy_window=copy.deepcopy(window),
    )


def resolve_rolling_run(prepared, params, strategy=None):
    """Resolve one exact runtime window tuple after Qlib initialization."""

    actual_anchor = str(params.get("anchor_date") or "")
    if not actual_anchor:
        raise RollingWindowResolutionError(
            "Qlib trading calendar did not provide an anchor date"
        )
    method = prepared.effective_config["training_method"]
    if strategy is None:
        if method == "cpcv":
            from quantpits.scripts.rolling import strategy_cpcv as strategy
        else:
            from quantpits.scripts.rolling import strategy_slide as strategy
    config = prepared.effective_config
    kwargs = {
        "rolling_start": config["rolling_start"],
        "train_years": config["train_years"],
        "test_step": config["test_step"],
        "anchor_date": actual_anchor,
    }
    if method == "cpcv":
        kwargs["cpcv_cfg"] = config["cpcv"]
        kwargs["freq"] = params.get("freq", "week")
    else:
        kwargs["valid_years"] = config.get("valid_years", 1)
    if prepared.options.action == "backtest_only":
        # Backtest-only operates on existing combined recorders, not on
        # per-window execution units.  It still binds the calendar-derived
        # params and target tuple without inventing an unused window scope.
        raw_windows = ()
    else:
        try:
            raw_windows = strategy.generate_windows(**kwargs)
        except RollingWindowResolutionError:
            raise
        except Exception as exc:
            raise RollingWindowResolutionError(
                "cannot resolve Rolling windows: %s" % exc
            )
    if not raw_windows and prepared.options.action != "backtest_only":
        raise RollingWindowResolutionError(
            "no Rolling windows were resolved for anchor %s" % actual_anchor
        )
    effective_fp = prepared.plan.metadata["effective_config_fingerprint"]
    descriptors = tuple(
        _descriptor(prepared.plan.metadata["family"], window, effective_fp)
        for window in raw_windows
    )
    indices = tuple(item.display_index for item in descriptors)
    if len(indices) != len(set(indices)):
        raise RollingWindowResolutionError("resolved window indices are not unique")
    keys = tuple(item.window_key for item in descriptors)
    if len(keys) != len(set(keys)):
        raise RollingWindowResolutionError("resolved window identities are not unique")
    try:
        identity = RollingRunIdentity(
            workspace_fingerprint=prepared.plan.metadata["workspace_fingerprint"],
            family=prepared.plan.metadata["family"],
            action=prepared.options.action,
            plan_fingerprint=prepared.plan_fingerprint,
            config_fingerprint=effective_fp,
            anchor_date=actual_anchor,
            target_keys=tuple(item.target_key for item in prepared.targets),
            window_keys=keys,
            runtime_params_fingerprint=fingerprint_value(dict(params)),
        )
    except (KeyError, RollingIdentityError) as exc:
        raise RollingWindowResolutionError(
            "cannot build Rolling execution identity: %s" % exc
        )
    return ResolvedRollingRun(
        prepared=prepared,
        actual_anchor=actual_anchor,
        params=dict(params),
        windows=descriptors,
        identity=identity,
    )
