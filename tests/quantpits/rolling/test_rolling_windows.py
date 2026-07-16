"""Contracts for Phase 28C runtime window resolution."""

from types import SimpleNamespace

import pytest

from quantpits.rolling.errors import RollingWindowResolutionError
from quantpits.rolling.windows import resolve_rolling_run


def _prepared(method="slide"):
    family = "cpcv_rolling" if method == "cpcv" else "rolling"
    config = {
        "rolling_start": "2020-01-01",
        "train_years": 3,
        "valid_years": 1,
        "test_step": "3M",
        "training_method": method,
    }
    if method == "cpcv":
        config["cpcv"] = {
            "n_groups": 5, "n_val_groups": 1,
            "purge_steps": 1, "embargo_steps": 1,
        }
    return SimpleNamespace(
        effective_config=config,
        plan=SimpleNamespace(metadata={
            "family": family,
            "effective_config_fingerprint": "a" * 64,
            "workspace_fingerprint": "b" * 64,
        }),
        plan_fingerprint="c" * 64,
        options=SimpleNamespace(action="daily"),
        targets=(SimpleNamespace(target_key="demo@%s" % family),),
        state=SimpleNamespace(fingerprint="state-baseline"),
    )


def test_resolver_uses_canonical_window_identity_without_display_index():
    calls = []
    strategy = SimpleNamespace(generate_windows=lambda **kwargs: calls.append(kwargs) or [{
        "window_idx": 0,
        "train_start": "2020-01-01", "train_end": "2022-12-31",
        "valid_start": "2023-01-01", "valid_end": "2023-12-31",
        "test_start": "2024-01-01", "test_end": "2024-03-31",
    }])
    params = {"anchor_date": "2024-03-31", "freq": "week"}
    first = resolve_rolling_run(_prepared(), params, strategy=strategy)
    second = resolve_rolling_run(_prepared(), params, strategy=strategy)
    assert len(calls) == 2
    assert len(first.windows) == 1
    assert first.windows[0].window_key.startswith("rolling:2024-01-01:2024-03-31:")
    assert first.execution_fingerprint == second.execution_fingerprint
    assert first.windows[0].identity.window_key == first.windows[0].window_key
    assert first.legacy_windows[0]["window_idx"] == 0

    changed_index_strategy = SimpleNamespace(generate_windows=lambda **_kwargs: [{
        "window_idx": 9,
        "train_start": "2020-01-01", "train_end": "2022-12-31",
        "valid_start": "2023-01-01", "valid_end": "2023-12-31",
        "test_start": "2024-01-01", "test_end": "2024-03-31",
    }])
    changed_index = resolve_rolling_run(
        _prepared(), params, strategy=changed_index_strategy,
    )
    assert changed_index.windows[0].display_index == 9
    assert changed_index.windows[0].window_key == first.windows[0].window_key
    assert changed_index.execution_fingerprint == first.execution_fingerprint

    duplicate_strategy = SimpleNamespace(generate_windows=lambda **_kwargs: [
        dict(changed_index.legacy_windows[0], window_idx=0),
        dict(changed_index.legacy_windows[0], window_idx=1),
    ])
    with pytest.raises(RollingWindowResolutionError, match="identities are not unique"):
        resolve_rolling_run(_prepared(), params, strategy=duplicate_strategy)


def test_cpcv_window_identity_binds_fold_content():
    def make_strategy(valid_end):
        return SimpleNamespace(generate_windows=lambda **_kwargs: [{
            "window_idx": 0,
            "train_start": "2020-01-01", "train_end": "2022-12-31",
            "test_start": "2023-01-01", "test_end": "2023-03-31",
            "cpcv_folds": [{
                "train_segments": [["2020-01-01", "2021-12-31"]],
                "valid_start_time": "2022-01-01",
                "valid_end_time": valid_end,
            }],
        }])
    params = {"anchor_date": "2023-03-31", "freq": "week"}
    first = resolve_rolling_run(
        _prepared("cpcv"), params, strategy=make_strategy("2022-03-31"),
    )
    changed = resolve_rolling_run(
        _prepared("cpcv"), params, strategy=make_strategy("2022-04-30"),
    )
    assert first.windows[0].fold_fingerprint
    assert first.windows[0].window_key != changed.windows[0].window_key
    assert first.execution_fingerprint != changed.execution_fingerprint


def test_execution_identity_changes_for_anchor_target_window_config_or_runtime_params():
    def strategy(test_end="2024-03-31"):
        return SimpleNamespace(generate_windows=lambda **_kwargs: [{
            "window_idx": 0,
            "train_start": "2020-01-01", "train_end": "2022-12-31",
            "valid_start": "2023-01-01", "valid_end": "2023-12-31",
            "test_start": "2024-01-01", "test_end": test_end,
        }])

    base_prepared = _prepared()
    baseline = resolve_rolling_run(
        base_prepared, {"anchor_date": "2024-03-31", "freq": "week"},
        strategy=strategy(),
    )
    changed_anchor = resolve_rolling_run(
        base_prepared, {"anchor_date": "2024-04-01", "freq": "week"},
        strategy=strategy(),
    )
    changed_window = resolve_rolling_run(
        base_prepared, {"anchor_date": "2024-03-31", "freq": "week"},
        strategy=strategy("2024-04-30"),
    )
    changed_target_prepared = _prepared()
    changed_target_prepared.targets = (
        SimpleNamespace(target_key="other@rolling"),
    )
    changed_target = resolve_rolling_run(
        changed_target_prepared,
        {"anchor_date": "2024-03-31", "freq": "week"},
        strategy=strategy(),
    )
    changed_config_prepared = _prepared()
    changed_config_prepared.plan.metadata["effective_config_fingerprint"] = "d" * 64
    changed_config = resolve_rolling_run(
        changed_config_prepared,
        {"anchor_date": "2024-03-31", "freq": "week"},
        strategy=strategy(),
    )
    changed_params = resolve_rolling_run(
        base_prepared, {"anchor_date": "2024-03-31", "freq": "day"},
        strategy=strategy(),
    )
    assert len({
        baseline.execution_fingerprint,
        changed_anchor.execution_fingerprint,
        changed_window.execution_fingerprint,
        changed_target.execution_fingerprint,
        changed_config.execution_fingerprint,
        changed_params.execution_fingerprint,
    }) == 6
