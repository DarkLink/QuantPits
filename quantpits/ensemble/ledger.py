"""Fusion run ledger helpers for ensemble fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FusionLedgerEntry:
    """Data required to append one ensemble fusion ledger record."""

    run_date: str
    combo_name: str | None
    models: Sequence[str]
    method: str
    is_default: bool
    eval_window: Mapping[str, Any]
    metrics: Mapping[str, Any]
    sub_model_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    loo_contributions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    cli_args: Sequence[str] = ()
    source_recorders: Mapping[str, str] = field(default_factory=dict)
    source_anchors: Mapping[str, str] = field(default_factory=dict)
    run_id: str | None = None
    plan_fingerprint: str | None = None


def build_fusion_ledger_record(entry: FusionLedgerEntry) -> dict[str, Any]:
    """Build the JSON-serializable ledger record for one fusion run."""
    oos_last_years = entry.eval_window.get("only_last_years", 0)
    oos_last_months = entry.eval_window.get("only_last_months", 0)
    is_oos = oos_last_years > 0 or oos_last_months > 0

    return {
        "run_date": entry.run_date,
        "combo_name": entry.combo_name or "default",
        "models": list(entry.models),
        "method": entry.method,
        "is_default": entry.is_default,
        "eval_window": {
            "start": entry.eval_window.get("start"),
            "end": entry.eval_window.get("end"),
            "is_oos": is_oos,
            "only_last_years": oos_last_years,
            "only_last_months": oos_last_months,
        },
        "metrics": dict(entry.metrics),
        "sub_model_metrics": dict(entry.sub_model_metrics or {}),
        "loo_contributions": dict(entry.loo_contributions or {}),
        "source": "ensemble_fusion",
        "cli_args": " ".join(entry.cli_args) if entry.cli_args else None,
        "source_recorders": dict(entry.source_recorders),
        "source_anchors": dict(entry.source_anchors),
        "run_id": entry.run_id,
        "plan_fingerprint": entry.plan_fingerprint,
    }


def append_fusion_ledger(workspace_root: str | Path, entry: FusionLedgerEntry) -> Path:
    """Append one fusion ledger record and return the ledger path."""
    root = Path(workspace_root).expanduser().resolve()
    ledger_path = root / "data" / "fusion_run_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    record = build_fusion_ledger_record(entry)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    print(
        f"\n[Ledger] 已追加本次融合结果 (combo={record['combo_name']}, "
        f"oos={record['eval_window']['is_oos']}) → {ledger_path}"
    )
    return ledger_path
