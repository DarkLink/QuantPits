"""Contracts for Phase 28C runtime window resolution."""

from types import SimpleNamespace

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
            "effective_config_fingerprint": "effective-config",
        }),
        plan_fingerprint="prepared-plan",
        options=SimpleNamespace(action="daily"),
        targets=(SimpleNamespace(target_key="demo@%s" % family),),
        state=SimpleNamespace(fingerprint="state-baseline"),
    )


def test_slide_resolution_calls_generator_once_and_has_stable_identity():
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
    assert first.windows[0].window_key.startswith("slide:2024-01-01:2024-03-31:")
    assert first.execution_fingerprint == second.execution_fingerprint
    assert first.legacy_windows[0]["window_idx"] == 0


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
