from quantpits.runtime import CommandPlan, CommandStep, InputRef, OutputRef, StateRef
from quantpits.runtime.render import render_command_plan


def test_render_command_plan_includes_core_sections(tmp_path):
    plan = CommandPlan(
        command="ensemble_fusion",
        workspace=str(tmp_path),
        run_id="run-1",
        mode="from-config-all",
        args=("--all",),
        inputs=(
            InputRef(
                "config/ensemble_config.json",
                kind="config",
                fingerprint="abc",
                description="ensemble config",
            ),
        ),
        outputs=(OutputRef("output/ensemble/pred.pkl", kind="prediction", overwrite=True),),
        states=(StateRef("config/ensemble_records.json", action="read_write"),),
        steps=(CommandStep("backtest", "run Qlib backtest", expensive=True),),
        warnings=("production workspace selected",),
    )

    rendered = render_command_plan(plan)

    assert "--- Execution Plan (dry run) ---" in rendered
    assert "Command: ensemble_fusion" in rendered
    assert "Workspace:" in rendered
    assert "Run ID: run-1" in rendered
    assert "config/ensemble_config.json [config] sha256=abc" in rendered
    assert "backtest: run Qlib backtest [expensive]" in rendered
    assert "output/ensemble/pred.pkl [prediction] overwrite" in rendered
    assert "config/ensemble_records.json [read_write]" in rendered
    assert "production workspace selected" in rendered
    assert not (tmp_path / "output").exists()


def test_render_command_plan_omits_empty_optional_sections():
    rendered = render_command_plan(
        CommandPlan(command="order_gen", workspace="/tmp/ws", run_id="run-1")
    )

    assert "Inputs:" not in rendered
    assert "Would write:" not in rendered
    assert "Warnings:" not in rendered
