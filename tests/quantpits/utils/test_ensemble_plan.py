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


def test_resolve_models_mode_uses_model_keys(ensemble_plan_module):
    train_records = {"models": {"gru@static": "rid1", "linear@rolling": "rid2"}}

    combos = ensemble_plan_module.resolve_ensemble_combos(
        args=_args(models="gru,linear", training_mode="static"),
        train_records=train_records,
        ensemble_config={},
    )

    assert len(combos) == 1
    assert combos[0].models == ("gru@static",)
    assert "linear" in combos[0].warnings[0]


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


def test_resolve_from_config_all_keeps_empty_combo_with_warning(ensemble_plan_module):
    train_records = {"models": {"m1@static": "rid1"}}
    ensemble_config = {
        "combos": {
            "good": {"models": ["m1"], "default": True},
            "bad": {"models": ["missing"]},
        }
    }

    combos = ensemble_plan_module.resolve_ensemble_combos(
        args=_args(from_config_all=True),
        train_records=train_records,
        ensemble_config=ensemble_config,
    )

    assert [combo.name for combo in combos] == ["good", "bad"]
    assert combos[1].models == ()
    assert "missing" in combos[1].warnings[0]


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
