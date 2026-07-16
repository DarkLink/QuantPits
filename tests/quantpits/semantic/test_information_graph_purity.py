import json
import os
import subprocess
import sys
from pathlib import Path

from quantpits.ensemble.command import (
    EnsembleCommandDependencies, EnsembleCommandRequest, build_ensemble_arg_parser, run_ensemble_command,
)
from quantpits.ensemble.config import load_ensemble_run_config
from quantpits.order.command import (
    OrderCommandDependencies, OrderCommandRequest, build_order_arg_parser, load_order_run_config, run_order_command,
)
from quantpits.post_trade.command import PostTradeRunOptions, prepare_post_trade_run
from quantpits.rolling.command import RollingRunOptions, prepare_rolling_run
from quantpits.training.command import TrainingRunOptions, prepare_training_run

from .artifact_graph import observe_artifact_graph
from .scenario_workspace import ScenarioWorkspace


def test_programmatic_information_graph_is_zero_write(tmp_path):
    workspace = ScenarioWorkspace.create(tmp_path)
    before = observe_artifact_graph(workspace.root)
    cwd = os.getcwd()
    environment = dict(os.environ)

    training = prepare_training_run(
        ctx=workspace.ctx, options=TrainingRunOptions(family="static", action="full", explain_plan=True)
    )
    rolling = prepare_rolling_run(
        workspace.ctx, RollingRunOptions(action="cold_start", all_enabled=True, json_plan=True),
        cli_args=("--cold-start", "--all-enabled", "--json-plan"),
    )
    ensemble_args = build_ensemble_arg_parser().parse_args(["--from-config", "--json-plan"])
    ensemble = run_ensemble_command(
        EnsembleCommandRequest(ensemble_args, ("--from-config", "--json-plan")),
        EnsembleCommandDependencies(
            get_workspace_context=lambda: workspace.ctx,
            load_run_config=lambda ctx, record_file: load_ensemble_run_config(ctx, record_file=record_file),
            safeguard=lambda label: (_ for _ in ()).throw(AssertionError("safeguard called")),
            service_factory=lambda: (_ for _ in ()).throw(AssertionError("service created")),
        ),
    )
    post_trade = prepare_post_trade_run(
        workspace.ctx,
        PostTradeRunOptions(
            scope="all", start_date="2026-07-17", end_date="2026-07-17", json_plan=True,
        ),
    )
    order_args = build_order_arg_parser().parse_args(["--json-plan"])
    order = run_order_command(
        OrderCommandRequest(order_args, ("--json-plan",)),
        OrderCommandDependencies(
            get_workspace_context=lambda: workspace.ctx,
            load_run_config=lambda ctx, options: load_order_run_config(ctx, options),
            safeguard=lambda label: (_ for _ in ()).throw(AssertionError("safeguard called")),
            execute=lambda prepared: (_ for _ in ()).throw(AssertionError("execute called")),
        ),
    )

    after = observe_artifact_graph(workspace.root)
    assert before.files == after.files
    assert os.getcwd() == cwd and dict(os.environ) == environment
    assert training.plan.metadata["target_keys"] == ["demo@static"]
    assert rolling.plan.metadata["target_keys"] == ["demo@rolling"]
    assert ensemble.mode == order.mode == "json-plan"
    assert ensemble.prepared.plan.metadata["resolved_combos"][0]["models"] == ["m1@static", "m2@static"]
    assert order.prepared.source.record_id is None
    assert post_trade.plan.metadata["scope"] == "all"


def test_rolling_cli_equal_workspace_form_matches_programmatic_plan(tmp_path):
    workspace = ScenarioWorkspace.create(tmp_path)
    before = observe_artifact_graph(workspace.root)
    environment = os.environ.copy()
    environment.pop("QLIB_WORKSPACE_DIR", None)
    environment.pop("MLFLOW_TRACKING_URI", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    completed = subprocess.run(
        [
            sys.executable, "-m", "quantpits.scripts.rolling_train",
            "--workspace=%s" % workspace.root, "--cold-start", "--all-enabled", "--json-plan",
        ],
        cwd=str(tmp_path), env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    cli = json.loads(completed.stdout)
    programmatic = prepare_rolling_run(
        workspace.ctx, RollingRunOptions(action="cold_start", all_enabled=True, json_plan=True),
        cli_args=("--cold-start", "--all-enabled", "--json-plan"),
    )
    assert cli["plan_fingerprint"] == programmatic.plan_fingerprint
    assert cli["plan"]["metadata"]["target_keys"] == ["demo@rolling"]
    assert observe_artifact_graph(workspace.root).files == before.files
