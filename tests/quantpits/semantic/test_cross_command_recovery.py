import json

import pytest

from quantpits.order.command import OrderRunOptions, load_order_run_config, prepare_order_run
from quantpits.post_trade.transaction import PostTradeTransactionManager

from .artifact_graph import observe_artifact_graph
from .scenario_workspace import ScenarioWorkspace


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def test_interrupted_post_trade_commit_recovers_before_order_consumption(tmp_path):
    workspace = ScenarioWorkspace.create(tmp_path)
    workspace.write_json("config/ensemble_records.json", {
        "default_combo": "sentinel", "combos": {"sentinel": "ensemble-sentinel"},
    })
    post_cashflow = _json_bytes({"cashflows": {"2026-07-17": 25.0}})
    post_state = _json_bytes({
        "current_date": "2026-07-17", "last_processed_date": "2026-07-17",
        "current_cash": 2048.0, "current_holding": [],
    })
    fired = {"value": False}

    def interrupt(event, artifact):
        if not fired["value"] and event == "after_target_write_before_verification" and artifact.role == "cashflow_config":
            fired["value"] = True
            raise OSError("semantic interruption")

    manager = PostTradeTransactionManager(workspace.ctx, event_hook=interrupt)
    journal = manager.prepare(
        transaction_id="recover-semantic", run_id="recover-semantic", scope="state",
        light_fingerprint="light", resolved_fingerprint="resolved",
        cursor_before="2026-07-16", cursor_after="2026-07-17",
        processed_dates=("2026-07-17",), consumed_cashflow_dates=("2026-07-17",),
        payloads=(
            (400, "cashflow_config", workspace.root / "config/cashflow.json", post_cashflow),
            (500, "prod_config_cursor", workspace.root / "config/prod_config.json", post_state),
        ),
    )
    with pytest.raises(OSError, match="semantic interruption"):
        manager.commit(journal)
    interrupted = manager.load("recover-semantic")
    assert interrupted.status == "committing"
    assert workspace.ctx.config_path("prod_config.json").read_bytes() != post_state

    recovered = PostTradeTransactionManager(workspace.ctx).recover("recover-semantic")
    first_graph = observe_artifact_graph(workspace.root)
    repeated = PostTradeTransactionManager(workspace.ctx).recover("recover-semantic")
    second_graph = observe_artifact_graph(workspace.root)
    options = OrderRunOptions(run_id="post-recovery-plan")
    config = load_order_run_config(workspace.ctx, options)
    prepared = prepare_order_run(ctx=workspace.ctx, options=options, cli_args=(), run_config=config)

    assert recovered.status == repeated.status == "state_committed"
    assert recovered.transaction_id == repeated.transaction_id == "recover-semantic"
    assert tuple(item.path for item in recovered.artifacts) == (
        "config/cashflow.json", "config/prod_config.json",
    )
    assert first_graph.artifacts == second_graph.artifacts
    assert first_graph.physical_escapes == second_graph.physical_escapes == ()
    assert config.merged_config["current_cash"] == 2048.0
    assert prepared.plan.metadata["current_state_date"] == "2026-07-17"
