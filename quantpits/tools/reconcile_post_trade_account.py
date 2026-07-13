"""Read-only-by-default post-trade account reconciliation command."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from quantpits.post_trade.state import AccountState, Position, ValuationSnapshot, decimal_value, normalize_instrument
from quantpits.post_trade.valuation_provenance import ValuationQuoteEvidence
from quantpits.post_trade.valuation_reconciliation import reconcile_account
from quantpits.scripts.brokers.gtja import GtjaAdapter
from quantpits.utils.workspace import WorkspaceContext


def build_parser():
    parser = argparse.ArgumentParser(description="Reconcile post-trade account valuation evidence")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--broker", choices=("gtja",), required=True)
    parser.add_argument("--snapshot-file", required=True)
    parser.add_argument("--account-date", required=True)
    parser.add_argument("--snapshot-effective-date")
    parser.add_argument("--snapshot-market-date")
    parser.add_argument("--cash-tolerance", default="0.01")
    parser.add_argument("--value-tolerance", default="0.01")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--run-id")
    return parser


def _contained(ctx, value):
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else ctx.path(path.as_posix()).resolve()
    try: path.relative_to(ctx.root)
    except ValueError as exc: raise ValueError("Snapshot file must be inside workspace") from exc
    return path


def _load_state(ctx, account_date):
    path = ctx.data_path("holding_log_full.csv")
    frame = pd.read_csv(path, dtype={"证券代码": str})
    rows = frame.loc[frame["成交日期"].astype(str) == account_date]
    if rows.empty: raise ValueError("No holding state for requested account date")
    cash_rows = rows.loc[rows["证券代码"] == "CASH"]
    if len(cash_rows) != 1: raise ValueError("Holding state must contain exactly one CASH row")
    positions = []
    for _, row in rows.loc[rows["证券代码"] != "CASH"].iterrows():
        positions.append(Position(
            normalize_instrument(row["证券代码"]),
            decimal_value(row["持仓数量"], field="holding quantity"),
            decimal_value(row["持仓成本"], field="holding cost"),
        ))
    return AccountState(account_date, decimal_value(cash_rows.iloc[0]["持仓数量"], field="cash"), tuple(positions))


def _evidence_from_dict(value):
    decimal_fields = {"price", "raw_close", "factor"}
    kwargs = {key: (Decimal(str(item)) if key in decimal_fields and item is not None else item) for key, item in value.items()}
    return ValuationQuoteEvidence(**kwargs)


def _load_valuation(ctx, account_date):
    path = ctx.data_path("valuation_evidence.jsonl")
    if not path.exists(): raise ValueError("No valuation evidence sidecar exists")
    matches = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line).get("market_date") == account_date]
    if len(matches) != 1: raise ValueError("Expected exactly one valuation evidence record")
    value = matches[0]
    quotes = tuple(_evidence_from_dict(item) for item in value["quotes"])
    benchmark = _evidence_from_dict(value["benchmark"])
    return ValuationSnapshot(account_date, tuple((item.instrument, item.price) for item in quotes), benchmark.price, quotes, benchmark)


def run(args):
    ctx = WorkspaceContext.from_root(args.workspace)
    snapshot_path = _contained(ctx, args.snapshot_file)
    if not snapshot_path.is_file(): raise FileNotFoundError("Snapshot file does not exist")
    snapshot = GtjaAdapter().parse_account_snapshot(
        snapshot_path, effective_date=args.snapshot_effective_date,
        market_date=args.snapshot_market_date,
    )
    report = reconcile_account(
        _load_state(ctx, args.account_date), _load_valuation(ctx, args.account_date), snapshot,
        cash_tolerance=Decimal(args.cash_tolerance), value_tolerance=Decimal(args.value_tolerance),
    )
    payload = report.to_public_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json: print(rendered)
    else:
        print("Account date: %s" % report.account_state_date)
        print("State comparability: %s" % report.state_comparability)
        print("Price comparability: %s" % report.price_comparability)
        print("Analytics eligibility: %s" % report.analytics_eligibility)
    if args.write_report:
        from quantpits.post_trade.ingestion import _atomic_bytes
        run_id = args.run_id or "manual"
        if any(value in run_id for value in ("/", "..")): raise ValueError("Invalid run id")
        output = ctx.output_path("reconciliation", "post_trade_account_%s_%s.json" % (args.account_date, run_id))
        _atomic_bytes(output, (rendered + "\n").encode("utf-8"))
    return 0 if report.analytics_eligibility == "eligible" else 2


def main(argv=None):
    parser = build_parser()
    try: code = run(parser.parse_args(argv))
    except Exception as exc:
        print("Reconciliation failed: %s: %s" % (type(exc).__name__, exc))
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
