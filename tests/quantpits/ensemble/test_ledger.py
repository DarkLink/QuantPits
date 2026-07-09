import json

from quantpits.ensemble.ledger import (
    FusionLedgerEntry,
    append_fusion_ledger,
    build_fusion_ledger_record,
)


def _entry(**overrides):
    data = {
        "run_date": "2026-01-05",
        "combo_name": "combo_a",
        "models": ("M1", "M2"),
        "method": "equal",
        "is_default": True,
        "eval_window": {
            "start": "2025-01-01",
            "end": "2026-01-05",
            "only_last_years": 0,
            "only_last_months": 0,
        },
        "metrics": {"annualized_return": 0.12, "calmar": 1.3},
        "sub_model_metrics": {"M1": {"annualized_return": 0.1}},
        "loo_contributions": {"M1": {"delta": 0.003}},
        "cli_args": ("--from-config", "--freq", "week"),
    }
    data.update(overrides)
    return FusionLedgerEntry(**data)


def test_build_fusion_ledger_record_full_window():
    record = build_fusion_ledger_record(_entry(combo_name=None, cli_args=()))

    assert record["combo_name"] == "default"
    assert record["models"] == ["M1", "M2"]
    assert record["eval_window"]["is_oos"] is False
    assert record["source"] == "ensemble_fusion"
    assert record["cli_args"] is None


def test_build_fusion_ledger_record_oos_flags():
    by_year = build_fusion_ledger_record(
        _entry(eval_window={"start": "2025-01-01", "end": "2026-01-05", "only_last_years": 1})
    )
    by_month = build_fusion_ledger_record(
        _entry(eval_window={"start": "2025-01-01", "end": "2026-01-05", "only_last_months": 6})
    )

    assert by_year["eval_window"]["is_oos"] is True
    assert by_year["eval_window"]["only_last_months"] == 0
    assert by_month["eval_window"]["is_oos"] is True
    assert by_month["eval_window"]["only_last_years"] == 0


def test_append_fusion_ledger_appends_jsonl(tmp_path):
    workspace = tmp_path / "workspace"
    path1 = append_fusion_ledger(workspace, _entry(run_date="2026-01-05"))
    path2 = append_fusion_ledger(workspace, _entry(run_date="2026-01-12"))

    assert path1 == path2
    lines = path1.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["run_date"] for record in records] == ["2026-01-05", "2026-01-12"]
    assert records[0]["cli_args"] == "--from-config --freq week"


def test_append_fusion_ledger_preserves_unicode(tmp_path):
    workspace = tmp_path / "workspace"
    ledger_path = append_fusion_ledger(
        workspace,
        _entry(combo_name="组合A", models=("模型一",)),
    )

    raw = ledger_path.read_text(encoding="utf-8")
    assert "组合A" in raw
    assert "模型一" in raw
