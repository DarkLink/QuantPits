import json
import os
import subprocess
import sys
import builtins
from pathlib import Path

import pytest

from quantpits.ensemble.command import (
    EnsembleCommandDependencies, EnsembleCommandRequest, build_ensemble_arg_parser, run_ensemble_command,
)
from quantpits.ensemble.config import load_ensemble_run_config
from quantpits.order.command import (
    OrderCommandDependencies, OrderCommandRequest, build_order_arg_parser, load_order_run_config, run_order_command,
)
from quantpits.post_trade.command import PostTradeRunOptions, prepare_post_trade_run
from quantpits.rolling.command import RollingRunOptions, prepare_rolling_run
from quantpits.rolling.state import inspect_rolling_state
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
    assert before.artifacts == after.artifacts
    assert before.physical_escapes == after.physical_escapes == ()
    assert os.getcwd() == cwd and dict(os.environ) == environment
    assert training.plan.metadata["target_keys"] == ["demo@static"]
    assert rolling.plan.metadata["target_keys"] == ["demo@rolling"]
    assert ensemble.mode == order.mode == "json-plan"
    assert ensemble.prepared.plan.metadata["resolved_combos"][0]["models"] == ["m1@static", "m2@static"]
    assert order.prepared.source.record_id is None
    assert post_trade.plan.metadata["scope"] == "all"


@pytest.mark.parametrize("workspace_form", ("equal", "separated"))
def test_rolling_cli_workspace_forms_match_programmatic_plan(workspace_form, tmp_path):
    workspace = ScenarioWorkspace.create(tmp_path)
    before = observe_artifact_graph(workspace.root)
    environment = os.environ.copy()
    environment.pop("QLIB_WORKSPACE_DIR", None)
    environment.pop("MLFLOW_TRACKING_URI", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    workspace_args = (
        ["--workspace=%s" % workspace.root]
        if workspace_form == "equal"
        else ["--workspace", str(workspace.root)]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "quantpits.scripts.rolling_train"]
        + workspace_args
        + ["--cold-start", "--all-enabled", "--json-plan"],
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
    assert cli["plan"]["metadata"]["workspace_fingerprint"] == programmatic.plan.metadata["workspace_fingerprint"]
    assert cli["plan"]["metadata"]["state_inspection"] == programmatic.plan.metadata["state_inspection"]
    after = observe_artifact_graph(workspace.root)
    assert after.artifacts == before.artifacts
    assert after.physical_escapes == before.physical_escapes == ()


def test_rolling_state_classifications_are_zero_write_and_backend_free(
    tmp_path, monkeypatch,
):
    from quantpits.rolling.identity import workspace_fingerprint

    workspace = ScenarioWorkspace.create(tmp_path)
    state_path = workspace.root / "data" / "rolling_state.json"
    legacy = json.dumps({
        "anchor_date": "2026-07-17",
        "training_method": "slide",
        "completed_windows": {"0": {"demo": "recorder-demo"}},
        "current_window_idx": 0,
        "current_model": "demo",
        "total_windows": 1,
    }).encode("utf-8")
    versioned = json.dumps({
        "schema_version": 2,
        "workspace_fingerprint": workspace_fingerprint(workspace.root),
        "run_id": "semantic-demo", "family": "rolling", "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
        "anchor_date": "2026-07-17",
        "target_keys": ["demo@rolling"],
        "window_keys": ["rolling:2026-04-01:2026-06-30:abcdef123456"],
        "attempt_id": None, "phase": "executing", "units": [],
    }).encode("utf-8")
    cwd = os.getcwd()
    environment = dict(os.environ)
    imported = []
    real_import = builtins.__import__

    def rejecting_import(name, *args, **kwargs):
        if name.startswith(("qlib", "mlflow", "quantpits.utils.env")):
            imported.append(name)
            raise AssertionError("classifier attempted backend import: %s" % name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    observed = []
    for data in (None, b"", legacy, versioned):
        if data is None:
            state_path.unlink(missing_ok=True)
        else:
            state_path.write_bytes(data)
        before = observe_artifact_graph(workspace.root)
        inspection = inspect_rolling_state(state_path, workspace.root)
        after = observe_artifact_graph(workspace.root)
        observed.append(inspection.classification)
        assert after.artifacts == before.artifacts
        assert after.physical_escapes == before.physical_escapes == ()

    assert observed == ["missing", "corrupt", "valid_legacy", "valid_versioned"]
    assert imported == []
    assert os.getcwd() == cwd
    assert dict(os.environ) == environment
