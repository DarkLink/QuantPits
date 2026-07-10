from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from quantpits.order.command import (
    OrderRunConfig, OrderRunOptions, PreparedOrderRun, ResolvedOrderSource,
)
from quantpits.order.execution import LoadedOrderPrediction, OrderExecutionHooks, OrderSourceUnavailableError
from quantpits.order.persistence import OrderArtifactLedger, OrderPersistenceError
from quantpits.order.service import OrderGenerationService
from quantpits.runtime import CommandPlan, OutputRef
from quantpits.utils.workspace import WorkspaceContext


class _Generator:
    def analyze_positions(self, predictions, prices, holdings):
        frame = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["A"])
        return frame.iloc[0:0], frame.iloc[0:0], frame, frame, 1

    def generate_sell_orders(self, candidates, holdings, trade_date):
        return [], 0.0

    def generate_buy_orders(self, candidates, count, cash, trade_date):
        return [{"instrument": "A", "estimated_amount": 100.0}]


def _prepared(tmp_path, *, dry_run):
    ctx = WorkspaceContext.from_root(tmp_path)
    options = OrderRunOptions(output_dir=(tmp_path / "output").as_posix(), dry_run=dry_run)
    source = ResolvedOrderSource("ensemble", None, "demo", "rid", "Ensemble_Fusion", "ensemble")
    plan = CommandPlan(
        command="order_gen",
        workspace=ctx.root.as_posix(),
        run_id="run-1",
        outputs=(OutputRef("output/planned-but-not-written.csv", kind="report"),),
    )
    return PreparedOrderRun(
        ctx=ctx, options=options, execution_options=options, cli_args=(), validation_result=None,
        config=OrderRunConfig(
            merged_config={"current_cash": 1000, "current_holding": [], "market": "demo"},
            cashflow_config={}, strategy_config={}, ensemble_config={}, ensemble_records={}, train_records={},
        ),
        source=source, plan=plan, plan_fingerprint="abc",
    )


def _hooks(persist_calls):
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-01-01"), "A")], names=["datetime", "instrument"])
    source = ResolvedOrderSource("ensemble", None, "demo", "rid", "Ensemble_Fusion", "ensemble")

    def persist(request):
        persist_calls.append(request)
        return OrderArtifactLedger(
            buy_csv="output/buy_suggestion_ensemble_2026-01-02.csv",
            outputs=(OutputRef("output/buy_suggestion_ensemble_2026-01-02.csv", kind="report"),),
        )

    return OrderExecutionHooks(
        init_qlib=lambda: None, get_anchor_date=lambda: "2026-01-01",
        get_next_trade_date=lambda _: "2026-01-02",
        load_predictions=lambda _: LoadedOrderPrediction(pd.DataFrame({"score": [0.5]}, index=index), source, "demo source"),
        get_price_data=lambda *args, **kwargs: pd.DataFrame({"current_close": [10]}, index=["A"]),
        create_order_generator=lambda _: _Generator(), get_strategy_params=lambda _: {"topk": 1, "n_drop": 1},
        build_model_opinions=lambda _: None, persist_artifacts=persist,
    )


def test_dry_run_calculates_without_persistence_manifest_or_log(tmp_path):
    calls = []
    summary = OrderGenerationService(_hooks(calls)).execute(_prepared(tmp_path, dry_run=True))
    assert calls == []
    assert summary.buy_file is None
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "data").exists()


def test_service_loads_the_exact_prepared_source_once(tmp_path):
    prepared = _prepared(tmp_path, dry_run=True)
    seen = []
    hooks = _hooks([])
    original = hooks.load_predictions

    def load(source):
        seen.append(source)
        return original(source)

    OrderGenerationService(replace(hooks, load_predictions=load)).execute(prepared)
    assert seen == [prepared.source]


def test_real_run_links_actual_outputs_manifest_and_operator_log(tmp_path):
    calls = []
    summary = OrderGenerationService(_hooks(calls)).execute(_prepared(tmp_path, dry_run=False))
    assert len(calls) == 1
    assert summary.buy_file == "output/buy_suggestion_ensemble_2026-01-02.csv"
    assert summary.manifest_path == "output/manifests/order_gen/run-1.json"
    assert (tmp_path / summary.manifest_path).exists()
    assert (tmp_path / "data" / "operator_log.jsonl").exists()
    manifest = json.loads((tmp_path / summary.manifest_path).read_text())
    paths = [item["path"] for item in manifest["outputs"]]
    assert "output/planned-but-not-written.csv" not in paths
    assert manifest["records"]["source"]["record_id"] == "rid"
    assert "raw_config" not in json.dumps(manifest)


def test_no_manifest_keeps_operator_log(tmp_path):
    calls = []
    prepared = _prepared(tmp_path, dry_run=False)
    prepared = replace(
        prepared,
        execution_options=replace(prepared.execution_options, no_manifest=True),
    )
    summary = OrderGenerationService(_hooks(calls)).execute(prepared)
    assert summary.manifest_path is None
    assert not (tmp_path / "output" / "manifests").exists()
    assert (tmp_path / "data" / "operator_log.jsonl").exists()


def test_expected_execution_error_logs_without_failed_manifest(tmp_path):
    hooks = _hooks([])
    hooks = replace(
        hooks,
        load_predictions=lambda _: (_ for _ in ()).throw(OrderSourceUnavailableError("missing")),
    )
    with pytest.raises(OrderSourceUnavailableError):
        OrderGenerationService(hooks).execute(_prepared(tmp_path, dry_run=False))
    assert not (tmp_path / "output" / "manifests").exists()
    assert (tmp_path / "data" / "operator_log.jsonl").exists()


def test_unexpected_failure_manifest_does_not_fallback_to_planned_outputs(tmp_path):
    hooks = replace(_hooks([]), get_price_data=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        OrderGenerationService(hooks).execute(_prepared(tmp_path, dry_run=False))
    path = tmp_path / "output" / "manifests" / "order_gen" / "run-1.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "failed"
    assert [item["path"] for item in manifest["outputs"]] == ["output/manifests/order_gen/run-1.json"]


def test_partial_persistence_failure_manifest_lists_only_committed_outputs(tmp_path):
    committed = OutputRef("output/committed.csv", kind="report")
    hooks = replace(
        _hooks([]),
        persist_artifacts=lambda request: (_ for _ in ()).throw(
            OrderPersistenceError("write failed", committed_outputs=(committed,))
        ),
    )
    with pytest.raises(OrderPersistenceError):
        OrderGenerationService(hooks).execute(_prepared(tmp_path, dry_run=False))
    path = tmp_path / "output" / "manifests" / "order_gen" / "run-1.json"
    manifest = json.loads(path.read_text())
    assert [item["path"] for item in manifest["outputs"]] == [
        "output/committed.csv",
        "output/manifests/order_gen/run-1.json",
    ]
