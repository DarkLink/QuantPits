import json

from quantpits.post_trade.transaction import PostTradeTransactionManager

from .artifact_graph import assert_declared_writes, observe_artifact_graph
from .drivers import execute_order
from .scenario_workspace import ScenarioWorkspace


def _payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def test_post_trade_postimage_is_exact_order_account_input(tmp_path):
    workspace = ScenarioWorkspace.create(tmp_path)
    workspace.write_json("config/ensemble_records.json", {
        "default_combo": "sentinel", "combos": {"sentinel": "ensemble-sentinel"},
    })
    baseline = observe_artifact_graph(workspace.root)
    postimage = {
        "current_date": "2026-07-17", "last_processed_date": "2026-07-17",
        "current_cash": 4321.0,
        "current_holding": [{"instrument": "BBB", "quantity": 5}],
    }
    manager = PostTradeTransactionManager(workspace.ctx)
    journal = manager.prepare(
        transaction_id="post-trade-semantic", run_id="post-trade-semantic", scope="state",
        light_fingerprint="light", resolved_fingerprint="resolved",
        cursor_before="2026-07-16", cursor_after="2026-07-17",
        processed_dates=("2026-07-17",), consumed_cashflow_dates=(),
        payloads=((500, "prod_config_cursor", workspace.root / "config/prod_config.json", _payload(postimage)),),
    )
    committed = manager.commit(journal)
    prepared, summary, generator = execute_order(workspace, run_id="order-postimage")
    observed = observe_artifact_graph(workspace.root)

    assert committed.status == "state_committed"
    assert prepared.config.merged_config["current_cash"] == postimage["current_cash"]
    assert prepared.config.merged_config["current_holding"] == postimage["current_holding"]
    assert generator.cash_seen == postimage["current_cash"]
    assert generator.holdings_seen == tuple(postimage["current_holding"])
    order_manifest = observed.json(summary.manifest_path)
    assert order_manifest["status"] == "success"
    assert observed.json("data/.post_trade_transactions/post-trade-semantic/journal.json")["cursor_after"] == "2026-07-17"
    assert observed.jsonl("data/operator_log.jsonl")[-1]["script"] == "order_gen"
    assert not observed.physical_escapes
    assert_declared_writes(
        observed.changed_paths(baseline),
        ("config/prod_config.json", "data/.post_trade_transactions/", "output/", "data/operator_log.jsonl"),
    )

