from argparse import Namespace
from dataclasses import replace

import pytest


def _args(**overrides):
    base = {
        "models": None,
        "from_config": False,
        "from_config_all": False,
        "combo": None,
        "method": "equal",
        "weights": None,
        "training_mode": None,
        "record_file": "latest_train_records.json",
        "output_dir": "output/ensemble",
        "prediction_dir": None,
        "no_backtest": False,
        "no_charts": False,
        "detailed_analysis": False,
        "save_csv": False,
        "no_manifest": False,
        "freq": None,
        "start_date": None,
        "end_date": None,
        "only_last_years": 0,
        "only_last_months": 0,
        "run_id": None,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def ensemble_plan_module(mock_env_constants):
    from quantpits.utils import ensemble_plan

    return ensemble_plan


def test_resolve_models_mode_rejects_any_missing_model(ensemble_plan_module):
    train_records = {"models": {"gru@static": "rid1", "linear@rolling": "rid2"}}

    with pytest.raises(ensemble_plan_module.MissingComboModelError, match="linear"):
        ensemble_plan_module.resolve_ensemble_combos(
            args=_args(models="gru,linear", training_mode="static"),
            train_records=train_records,
            ensemble_config={},
        )


def test_resolve_from_config_default_combo_override_method(ensemble_plan_module):
    train_records = {"models": {"m1@static": "rid1"}}
    ensemble_config = {
        "combos": {
            "c1": {"models": ["m1"], "method": "icir_weighted", "default": True}
        }
    }

    combos = ensemble_plan_module.resolve_ensemble_combos(
        args=_args(from_config=True, method="manual", weights="m1@static:1"),
        train_records=train_records,
        ensemble_config=ensemble_config,
    )

    assert combos[0].name == "c1"
    assert combos[0].models == ("m1@static",)
    assert combos[0].method == "manual"
    assert combos[0].manual_weights == "m1@static:1"


def test_resolve_from_config_all_rejects_missing_combo_member(ensemble_plan_module):
    train_records = {"models": {"m1@static": "rid1"}}
    ensemble_config = {
        "combos": {
            "good": {"models": ["m1"], "default": True},
            "bad": {"models": ["missing"]},
        }
    }

    with pytest.raises(ensemble_plan_module.MissingComboModelError, match="missing"):
        ensemble_plan_module.resolve_ensemble_combos(
            args=_args(from_config_all=True),
            train_records=train_records,
            ensemble_config=ensemble_config,
        )


def test_resolve_combo_mode_success(ensemble_plan_module):
    train_records = {"models": {"m1@rolling": "rid1"}}
    ensemble_config = {
        "combos": {
            "chosen": {
                "models": ["m1"],
                "method": "icir_weighted",
                "training_mode": "rolling",
                "default": False,
            }
        }
    }

    combos = ensemble_plan_module.resolve_ensemble_combos(
        args=_args(combo="chosen"),
        train_records=train_records,
        ensemble_config=ensemble_config,
    )

    assert combos[0].name == "chosen"
    assert combos[0].models == ("m1@rolling",)
    assert combos[0].method == "icir_weighted"


def test_from_config_all_skips_disabled_and_explicit_opt_in_includes_it(ensemble_plan_module):
    records = {"models": {"m1@static": "r1", "m2@static": "r2"}}
    config = {"combos": {
        "active": {"models": ["m1"], "default": True},
        "retired": {"models": ["m2"], "enabled": False},
    }}
    default = ensemble_plan_module.resolve_ensemble_combos(
        args=_args(from_config_all=True), train_records=records, ensemble_config=config
    )
    opted_in = ensemble_plan_module.resolve_ensemble_combos(
        args=_args(from_config_all=True, include_disabled_combos=True),
        train_records=records, ensemble_config=config,
    )
    assert [item.name for item in default] == ["active"]
    assert [item.name for item in opted_in] == ["active", "retired"]
    assert opted_in[1].enabled is False


def test_missing_combo_raises_plan_error(ensemble_plan_module):
    with pytest.raises(ensemble_plan_module.EnsemblePlanError):
        ensemble_plan_module.resolve_ensemble_combos(
            args=_args(combo="missing"),
            train_records={"models": {}},
            ensemble_config={"combos": {"c1": {"models": []}}},
        )


def test_combo_with_no_valid_models_raises_plan_error(ensemble_plan_module):
    with pytest.raises(ensemble_plan_module.EnsemblePlanError):
        ensemble_plan_module.resolve_ensemble_combos(
            args=_args(combo="empty"),
            train_records={"models": {}},
            ensemble_config={"combos": {"empty": {"models": ["missing"]}}},
        )


def test_build_plan_omits_raw_config_and_adds_expected_steps(tmp_path, ensemble_plan_module):
    from quantpits.runtime import fingerprint_command_plan
    from quantpits.utils.workspace import WorkspaceContext

    ctx = WorkspaceContext.from_root(tmp_path)
    (tmp_path / "latest_train_records.json").write_text('{"models": {}}\n')
    combo = ensemble_plan_module.ResolvedCombo(
        name="c1",
        models=("m1@static",),
        method="equal",
        is_default=True,
        source="from-config",
    )

    plan = ensemble_plan_module.build_ensemble_command_plan(
        ctx=ctx,
        args=_args(from_config=True, no_backtest=True, no_charts=True, run_id="run-a"),
        train_records={"anchor_date": "2026-07-09", "experiment_name": "exp"},
        model_config={"freq": "week"},
        ensemble_config={"raw_config": {"secret": True}},
        combos=(combo,),
        run_id="run-a",
        cli_args=("--from-config", "--explain-plan"),
    )

    payload = plan.to_public_dict()
    assert payload["command"] == "ensemble_fusion"
    assert payload["mode"] == "from-config"
    assert "raw_config" not in str(payload)
    assert any(step.name == "backtest" and step.can_skip for step in plan.steps)
    assert any(ref.path == "latest_train_records.json" for ref in plan.inputs)

    same_plan_new_run = replace(plan, run_id="run-b")
    assert fingerprint_command_plan(plan) == fingerprint_command_plan(same_plan_new_run)


def test_build_plan_includes_chart_outputs_when_charts_enabled(tmp_path, ensemble_plan_module):
    from quantpits.utils.workspace import WorkspaceContext

    ctx = WorkspaceContext.from_root(tmp_path)
    combo = ensemble_plan_module.ResolvedCombo(
        name="c1",
        models=("m1@static",),
        method="equal",
        source="from-config",
    )

    plan = ensemble_plan_module.build_ensemble_command_plan(
        ctx=ctx,
        args=_args(from_config=True, run_id="run-a"),
        train_records={"anchor_date": "2026-07-09", "experiment_name": "exp"},
        model_config={"freq": "week"},
        ensemble_config={},
        combos=(combo,),
        run_id="run-a",
        cli_args=("--from-config", "--explain-plan"),
    )

    output_paths = {output.path for output in plan.outputs}
    assert "output/ensemble/ensemble_nav_<combo>_<anchor_date>.png" in output_paths
    assert "output/ensemble/ensemble_weights_<combo>_<anchor_date>.png" in output_paths


@pytest.mark.parametrize("flag", ["no_charts", "no_backtest"])
def test_build_plan_omits_chart_outputs_when_charts_cannot_run(
    tmp_path,
    ensemble_plan_module,
    flag,
):
    from quantpits.utils.workspace import WorkspaceContext

    ctx = WorkspaceContext.from_root(tmp_path)
    combo = ensemble_plan_module.ResolvedCombo(
        name="c1",
        models=("m1@static",),
        method="equal",
        source="from-config",
    )

    plan = ensemble_plan_module.build_ensemble_command_plan(
        ctx=ctx,
        args=_args(from_config=True, run_id="run-a", **{flag: True}),
        train_records={"anchor_date": "2026-07-09", "experiment_name": "exp"},
        model_config={"freq": "week"},
        ensemble_config={},
        combos=(combo,),
        run_id="run-a",
        cli_args=("--from-config", "--explain-plan"),
    )

    output_paths = {output.path for output in plan.outputs}
    assert "output/ensemble/ensemble_nav_<combo>_<anchor_date>.png" not in output_paths
    assert "output/ensemble/ensemble_weights_<combo>_<anchor_date>.png" not in output_paths
