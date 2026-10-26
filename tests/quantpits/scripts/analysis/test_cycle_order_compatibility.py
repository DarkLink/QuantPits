"""Native cycle orders must reach the existing derived analysis consumers."""

import pandas as pd
import pytest

from quantpits.order.persistence import OrderPersistenceRequest, persist_order_artifacts
from quantpits.order.suggestions import SuggestionConflictError
from quantpits.scripts.analysis import trade_classifier as tc
from quantpits.scripts.analysis.execution_analyzer import ExecutionAnalyzer
from quantpits.utils.workspace import WorkspaceContext


@pytest.fixture
def cycle_orders(tmp_path, monkeypatch):
    root = tmp_path / "account"
    (root / "data").mkdir(parents=True)
    # Tuesday execution, distinct from the Friday anchor in the directory.
    date = "2026-10-27"
    ledger = persist_order_artifacts(OrderPersistenceRequest(
        ctx=WorkspaceContext.from_root(root),
        output_dir=root / "output/cycles/project/week_20261023/orders",
        trade_date=date, source_label="ensemble", opinions=None,
        buy_orders=tuple({"instrument": code, "action": "BUY"} for code in
                         ("SZ000001", "SZ000002", "SZ000003")),
        sell_orders=({"instrument": "SH600001", "action": "SELL"},),
    ))
    trades = pd.DataFrame({
        "成交日期": pd.to_datetime([date, date]),
        "证券代码": ["SZ000002", "SH600001"],
        "交易类别": [tc.BUY_TYPES[1], tc.SELL_TYPES[0]],
        "成交数量": [100, 100], "成交金额": [1000, 1000],
    })
    trades.to_csv(root / "data/trade_log_full.csv", index=False)
    monkeypatch.setattr(tc, "ROOT_DIR", str(root))
    monkeypatch.setattr(tc, "ORDER_DIR", str(root / "data/order_history"))
    monkeypatch.setattr(tc, "OUTPUT_DIR", str(root / "output"))
    monkeypatch.setattr(tc, "TRADE_LOG_FILE", str(root / "data/trade_log_full.csv"))
    monkeypatch.setattr("quantpits.scripts.analysis.utils.ROOT_DIR", str(root))
    monkeypatch.setattr("quantpits.scripts.analysis.utils.get_forward_returns",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("quantpits.scripts.analysis.execution_analyzer.get_daily_features",
                        lambda *a, **k: pd.DataFrame())
    return root, ledger, trades


def test_native_cycle_orders_are_classified_without_history(cycle_orders):
    root, ledger, trades = cycle_orders
    assert not (root / "data/order_history").exists()
    result = tc.classify_trades()
    assert dict(zip(result.instrument, result.trade_class)) == {
        "SZ000002": "A", "SH600001": "S",
    }


def test_prefixed_native_orders_reach_legacy_explicit_directory_analysis(cycle_orders):
    root, ledger, trades = cycle_orders
    analyzer = ExecutionAnalyzer(trade_log_df=trades)
    result = analyzer.analyze_order_discrepancies(
        str((root / ledger.buy_csv).parent), market="all")
    assert result["total_missed_count"] == 1
    assert result["total_substitute_count"] == 1


def test_workspace_analysis_deduplicates_copies_and_preserves_originals(cycle_orders):
    root, ledger, trades = cycle_orders
    original = root / ledger.buy_csv
    payload = original.read_bytes()
    paths = [root / "output/buy_suggestion_20261027.csv",
             root / "data/order_history/buy_suggestion_model_2026-10-27.csv"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    result = ExecutionAnalyzer(trade_log_df=trades).analyze_order_discrepancies(
        workspace_root=root, market="all")
    assert (result["total_missed_count"], result["total_substitute_count"]) == (1, 1)
    assert set(result["suggestion_sources"][0]["paths"]) == {str(original), *(str(p) for p in paths)}
    assert original.read_bytes() == payload
    assert tc.classify_trades(workspace_root=root).iloc[0].trade_class == "A"


def test_conflict_cannot_be_manual_or_zero_bias_and_can_be_selected(cycle_orders):
    root, ledger, trades = cycle_orders
    other = root / "output/buy_suggestion_model_2026-10-27.csv"
    other.write_text("instrument\nSZ000002\nSZ000001\nSZ000003\n")
    analyzer = ExecutionAnalyzer(trade_log_df=trades)
    with pytest.raises(SuggestionConflictError, match=other.name):
        tc.classify_trades(workspace_root=root)
    with pytest.raises(SuggestionConflictError, match=other.name):
        analyzer.analyze_order_discrepancies(workspace_root=root, market="all")
    original = root / ledger.buy_csv
    assert tc.classify_trades(workspace_root=root, suggestion_files=[original]).iloc[0].trade_class == "A"
    assert tc.classify_trades(workspace_root=root, suggestion_files=[other]).iloc[0].trade_class == "S"
    chosen = analyzer.analyze_order_discrepancies(workspace_root=root, market="all", suggestion_files=[original])
    assert chosen["total_substitute_count"] == 1
    assert chosen["suggestion_sources"][0]["paths"] == [str(original)]


def test_main_backup_and_sell_classification_use_same_execution_day(cycle_orders):
    root, ledger, trades = cycle_orders
    main = trades.iloc[[0]].copy()
    main["证券代码"] = "SZ000001"
    pd.concat([trades, main]).to_csv(root / "data/trade_log_full.csv", index=False)
    result = tc.classify_trades(workspace_root=root)
    assert dict(zip(result.instrument, result.trade_class)) == {
        "SZ000001": "S", "SZ000002": "A", "SH600001": "S"}
    assert set(result.suggestion_date) == {"2026-10-27"}


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_source_read_interrupts_propagate(cycle_orders, monkeypatch, control):
    from pathlib import Path
    root, ledger, trades = cycle_orders
    original = Path.read_bytes
    def interrupted(path):
        if path.name.startswith("buy_suggestion_"):
            raise control()
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", interrupted)
    with pytest.raises(control):
        tc.classify_trades(workspace_root=root)
    with pytest.raises(control):
        ExecutionAnalyzer(trade_log_df=trades).analyze_order_discrepancies(workspace_root=root, market="all")


@pytest.mark.parametrize("conflict", [False, True])
def test_formal_run_analysis_consumes_cycle_orders(cycle_orders, monkeypatch, capsys, conflict):
    import sys
    from unittest.mock import MagicMock
    root, ledger, trades = cycle_orders
    monkeypatch.chdir(root)
    from quantpits.scripts import run_analysis as ra
    monkeypatch.setattr(ra, "ROOT_DIR", str(root))
    monkeypatch.setattr(ra, "init_qlib", lambda: None)
    monkeypatch.setattr(ra, "load_market_config", lambda: ("all", "BENCH"))
    analyzer = ExecutionAnalyzer(trade_log_df=trades)
    # Only unrelated market/portfolio computations are replaced. Suggestion
    # discovery, CSV loading, ranking and discrepancy calculations remain real.
    monkeypatch.setattr(analyzer, "calculate_slippage_and_delay", lambda: pd.DataFrame())
    monkeypatch.setattr(analyzer, "calculate_path_dependency", lambda: pd.DataFrame())
    monkeypatch.setattr(analyzer, "analyze_explicit_costs", lambda: {})
    monkeypatch.setattr(ra, "ExecutionAnalyzer", lambda **k: analyzer)
    portfolio = MagicMock()
    for method in ("calculate_traditional_metrics", "calculate_factor_exposure",
                   "calculate_style_exposures", "calculate_holding_metrics"):
        getattr(portfolio, method).return_value = {}
    portfolio.calculate_classified_returns.return_value = {
        "class_df": pd.DataFrame(), "manual_buys": pd.DataFrame(), "manual_sells": pd.DataFrame()}
    monkeypatch.setattr(ra, "PortfolioAnalyzer", lambda **k: portfolio)
    if conflict:
        (root / "output/buy_suggestion_other_20261027.csv").write_text("instrument\nSZ000002\n")
    monkeypatch.setattr(sys, "argv", ["run_analysis", "--output", "output/test_report.md"])
    status = ra.main()
    text = (root / "output/test_report.md").read_text()
    if conflict:
        assert status == 1
        assert "Order Suggestion Analysis Unresolved" in text
        assert "Conflicting buy suggestions" in text
        assert "completed successfully" not in capsys.readouterr().out
        assert "No measurable substitution bias" not in text
        monkeypatch.setattr(sys, "argv", ["run_analysis", "--output", "output/shareable_report.md", "--shareable"])
        assert ra.main() == 1
        shareable = (root / "output/shareable_report.md").read_text()
        assert "Order Suggestion Analysis Unresolved" in shareable
        assert str(root) not in shareable
        monkeypatch.setattr(sys, "argv", ["run_analysis", "--output", "output/selected_report.md",
                                         "--suggestion-file", ledger.buy_csv])
        assert ra.main() == 0
        text = (root / "output/selected_report.md").read_text()
    assert "Missed Top Buy Occurrences: 1, Substitute Buy Occurrences: 1" in text
    assert ledger.buy_csv in text
    assert not (root / "data/order_history").exists()


@pytest.mark.parametrize("conflict", [False, True])
def test_deep_analysis_uses_cycle_orders_and_surfaces_conflicts(cycle_orders, monkeypatch, conflict):
    from quantpits.scripts.deep_analysis.base_agent import AnalysisContext
    from quantpits.scripts.deep_analysis.agents.execution_quality import ExecutionQualityAgent
    from quantpits.scripts.deep_analysis.coordinator import Coordinator
    root, ledger, trades = cycle_orders
    monkeypatch.setattr("quantpits.scripts.analysis.utils.init_qlib", lambda: None)
    monkeypatch.setattr("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.calculate_slippage_and_delay",
                        lambda self: pd.DataFrame())
    monkeypatch.setattr("quantpits.scripts.analysis.execution_analyzer.ExecutionAnalyzer.analyze_explicit_costs",
                        lambda self: {})
    monkeypatch.setattr("quantpits.scripts.analysis.execution_analyzer.load_market_config", lambda: ("all", "BENCH"))
    if conflict:
        (root / "output/buy_suggestion_other_20261027.csv").write_text("instrument\nSZ000002\n")
    coordinator = Coordinator(str(root))
    coordinator._discover_dated_files()
    assert ("2026-10-27", str(root / ledger.buy_csv)) in coordinator._discovered_files["buy_suggestion"]
    assert ("2026-10-27", str(root / ledger.sell_csv)) in coordinator._discovered_files["sell_suggestion"]
    context = AnalysisContext(workspace_root=str(root), window_label="test", start_date="2026-10-27",
                              end_date="2026-10-27", trade_log_df=trades)
    findings = ExecutionQualityAgent().analyze(context)
    if conflict:
        assert "substitution_bias" not in findings.raw_metrics
        assert "Conflicting buy suggestions" in findings.raw_metrics["substitution_bias_error"]
        assert any(f.severity == "warning" and "unresolved" in f.title for f in findings.findings)
    else:
        assert findings.raw_metrics["substitution_bias"]["total_substitute_count"] == 1


@pytest.mark.parametrize("conflict", [False, True])
def test_dashboard_reads_cycle_orders_without_history(cycle_orders, monkeypatch, conflict):
    import importlib
    import sys
    from unittest.mock import MagicMock
    from quantpits.utils import env
    root, ledger, trades = cycle_orders
    monkeypatch.chdir(root)
    monkeypatch.setattr(env, "ROOT_DIR", str(root))
    monkeypatch.setattr("quantpits.scripts.analysis.utils.init_qlib", lambda: None)
    display = MagicMock()
    monkeypatch.setitem(sys.modules, "streamlit", display)
    # Remove the imported module afterwards so UI globals do not leak to tests.
    monkeypatch.delitem(sys.modules, "ui.dashboard", raising=False)
    dashboard = importlib.import_module("ui.dashboard")
    monkeypatch.delitem(sys.modules, "ui.dashboard", raising=False)
    analyzer = ExecutionAnalyzer(trade_log_df=trades)
    monkeypatch.setattr(analyzer, "calculate_slippage_and_delay", lambda: pd.DataFrame())
    monkeypatch.setattr(analyzer, "calculate_path_dependency", lambda: pd.DataFrame())
    monkeypatch.setattr("quantpits.scripts.analysis.execution_analyzer.load_market_config", lambda: ("all", "BENCH"))
    index = pd.MultiIndex.from_tuples(
        [(code, pd.Timestamp("2026-10-27")) for code in ("SZ000001", "SZ000002")],
        names=["instrument", "datetime"])
    monkeypatch.setattr("quantpits.scripts.analysis.utils.get_forward_returns",
                        lambda *a, **k: pd.DataFrame({"return_5d": [0.1, 0.04]}, index=index))
    bar = MagicMock(wraps=dashboard.px.bar)
    monkeypatch.setattr(dashboard.px, "bar", bar)
    if conflict:
        (root / "output/buy_suggestion_other_20261027.csv").write_text("instrument\nSZ000002\n")
    dashboard.render_micro_execution(analyzer)
    if conflict:
        assert "unresolved" in display.warning.call_args[0][0]
        display.json.assert_not_called()
    else:
        assert any("Total Missed Top Buys**: 1" in str(c) for c in display.write.call_args_list)
        assert ledger.buy_csv in str(display.json.call_args)
        assert bar.call_args[0][0]["Return"].tolist() == pytest.approx([0.1, 0.04])


@pytest.mark.parametrize("directory", ["output", "data/order_history"])
@pytest.mark.parametrize("suffix", ["2026-10-27", "model_2026-10-27", "20261027"])
def test_legacy_locations_and_names_keep_counts(cycle_orders, directory, suffix):
    root, ledger, trades = cycle_orders
    target = root / directory
    target.mkdir(parents=True, exist_ok=True)
    for direction, original in (("buy", ledger.buy_csv), ("sell", ledger.sell_csv)):
        (root / original).rename(target / ("%s_suggestion_%s.csv" % (direction, suffix)))
    result = tc.classify_trades(workspace_root=root)
    assert dict(zip(result.instrument, result.trade_class)) == {"SZ000002": "A", "SH600001": "S"}
    analyzer = ExecutionAnalyzer(trade_log_df=trades)
    explicit = analyzer.analyze_order_discrepancies(str(target), market="all")
    workspace = analyzer.analyze_order_discrepancies(workspace_root=root, market="all")
    assert explicit == workspace
    assert (explicit["total_missed_count"], explicit["total_substitute_count"]) == (1, 1)


def test_realized_substitution_math_unchanged_for_native_sources(cycle_orders, monkeypatch):
    root, ledger, trades = cycle_orders
    index = pd.MultiIndex.from_tuples(
        [(code, pd.Timestamp("2026-10-27")) for code in ("SZ000001", "SZ000002")],
        names=["instrument", "datetime"])
    monkeypatch.setattr("quantpits.scripts.analysis.utils.get_forward_returns",
                        lambda *a, **k: pd.DataFrame({"return_5d": [0.1, 0.04]}, index=index))
    monkeypatch.setattr("quantpits.scripts.analysis.execution_analyzer.get_daily_features",
                        lambda *a, **k: pd.DataFrame({"unadj_close": [10, 12]}, index=index))
    result = ExecutionAnalyzer(trade_log_df=trades).analyze_order_discrepancies(
        workspace_root=root, market="all")
    assert result["theoretical_substitute_bias_impact"] == pytest.approx(-0.06)
    assert result["realized_substitute_bias_impact"] == pytest.approx(0.148)


def test_prediction_audit_deduplicates_and_reports_conflicting_sources(cycle_orders, monkeypatch):
    from types import SimpleNamespace
    from quantpits.scripts.deep_analysis.agents.prediction_audit import PredictionAuditAgent
    from quantpits.scripts.deep_analysis.base_agent import AnalysisContext
    root, ledger, trades = cycle_orders
    original = root / ledger.buy_csv
    duplicate = root / "output/buy_suggestion_20261027.csv"
    duplicate.write_bytes(original.read_bytes())
    context = AnalysisContext(workspace_root=str(root), window_label="test", start_date="2026-10-27",
                              end_date="2026-10-27", buy_suggestion_files=[str(original), str(duplicate)])
    index = pd.MultiIndex.from_tuples(
        [(code, pd.Timestamp("2026-10-27")) for code in ("SZ000001", "SZ000002", "SZ000003")],
        names=["instrument", "datetime"])
    monkeypatch.setattr("quantpits.scripts.analysis.utils.init_qlib", lambda: None)
    monkeypatch.setattr("qlib.data.D", SimpleNamespace(
        features=lambda *a, **k: pd.DataFrame({"return": [0.1, 0.04, -0.01]}, index=index)))
    result = PredictionAuditAgent()._analyze_suggestion_hits(context)
    assert result["overall"]["n_suggestions"] == 3
    duplicate.write_text("instrument\nSZ000002\n")
    unresolved = PredictionAuditAgent()._analyze_suggestion_hits(context)
    assert "Conflicting buy suggestions" in unresolved["error"]
    assert "overall" not in unresolved
