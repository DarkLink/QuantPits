"""Runtime-only Rolling calendar resolution and stable window identities."""

import copy
from dataclasses import dataclass

from quantpits.rolling.errors import RollingWindowResolutionError
from quantpits.utils.workspace import fingerprint_value


@dataclass(frozen=True)
class RollingWindowDescriptor:
    display_index: int
    window_key: str
    family: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    valid_start: str = None
    valid_end: str = None
    fold_fingerprint: str = None
    _legacy_window: dict = None

    def to_legacy_window(self):
        """Return an isolated copy for the mutable legacy execution code."""

        return copy.deepcopy(self._legacy_window)

    def to_fingerprint_dict(self):
        return {
            "display_index": self.display_index,
            "window_key": self.window_key,
            "family": self.family,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "fold_fingerprint": self.fold_fingerprint,
        }


@dataclass(frozen=True)
class ResolvedRollingRun:
    prepared: object
    actual_anchor: str
    params: dict
    windows: tuple
    execution_fingerprint: str
    runtime_source_policy: str = "qlib_trading_calendar"

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
    fold_fingerprint = None
    if family == "cpcv_rolling":
        folds = window.get("cpcv_folds")
        if not isinstance(folds, list) or not folds:
            raise RollingWindowResolutionError(
                "CPCV resolved window %s has no folds" % index
            )
        fold_fingerprint = fingerprint_value(folds)
    identity = {
        "family": family,
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": window.get("valid_start"),
        "valid_end": window.get("valid_end"),
        "test_start": test_start,
        "test_end": test_end,
        "fold_fingerprint": fold_fingerprint,
        "effective_config_fingerprint": effective_config_fingerprint,
    }
    prefix = "cpcv" if family == "cpcv_rolling" else "slide"
    window_key = "%s:%s:%s:%s" % (
        prefix, test_start, test_end, fingerprint_value(identity)[:12],
    )
    return RollingWindowDescriptor(
        display_index=index,
        window_key=window_key,
        family=family,
        train_start=train_start,
        train_end=train_end,
        valid_start=(str(window["valid_start"])
                     if window.get("valid_start") is not None else None),
        valid_end=(str(window["valid_end"])
                   if window.get("valid_end") is not None else None),
        test_start=test_start,
        test_end=test_end,
        fold_fingerprint=fold_fingerprint,
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
    payload = {
        "prepared_plan_fingerprint": prepared.plan_fingerprint,
        "actual_anchor": actual_anchor,
        "family": prepared.plan.metadata["family"],
        "action": prepared.options.action,
        "ordered_target_keys": [item.target_key for item in prepared.targets],
        "ordered_windows": [item.to_fingerprint_dict() for item in descriptors],
        "effective_config_fingerprint": effective_fp,
        "state_baseline_fingerprint": prepared.state.fingerprint,
        "runtime_source_policy": "qlib_trading_calendar",
        "resolved_params_fingerprint": fingerprint_value(dict(params)),
    }
    return ResolvedRollingRun(
        prepared=prepared,
        actual_anchor=actual_anchor,
        params=dict(params),
        windows=descriptors,
        execution_fingerprint=fingerprint_value(payload),
    )
