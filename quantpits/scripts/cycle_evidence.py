#!/usr/bin/env python3
"""Explicit post-cycle evidence capture command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from quantpits.evidence import CaptureRequest, ContractError, ProductionCycleEvidenceSealer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cycle-evidence")
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture = subparsers.add_parser("capture", help="capture and create-only seal one completed cycle")
    capture.add_argument("--workspace", help="workspace root; defaults to QLIB_WORKSPACE_DIR")
    capture.add_argument("--cycle-id", required=True)
    capture.add_argument("--research-epoch-id", required=True)
    capture.add_argument("--post-trade-manifest", required=True)
    capture.add_argument("--prediction-manifest", required=True)
    capture.add_argument("--ensemble-manifest", required=True)
    capture.add_argument("--order-manifest", required=True)
    capture.add_argument("--deep-analysis-run")
    capture.add_argument("--decision-event")
    capture.add_argument("--qlib-data-dir")
    capture.add_argument("--dry-run", action="store_true")
    return parser


def _workspace(value: Optional[str]) -> Path:
    selected = value or os.environ.get("QLIB_WORKSPACE_DIR")
    if not selected:
        raise ContractError("--workspace or QLIB_WORKSPACE_DIR is required")
    return Path(selected)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = CaptureRequest(
            cycle_id=args.cycle_id,
            research_epoch_id=args.research_epoch_id,
            post_trade_manifest=args.post_trade_manifest,
            prediction_manifest=args.prediction_manifest,
            ensemble_manifest=args.ensemble_manifest,
            order_manifest=args.order_manifest,
            deep_analysis_run=args.deep_analysis_run,
            decision_event=args.decision_event,
        )
        sealer = ProductionCycleEvidenceSealer(
            _workspace(args.workspace),
            qlib_data_dir=Path(args.qlib_data_dir) if args.qlib_data_dir else None,
        )
        result = sealer.capture(request, dry_run=args.dry_run)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        print(json.dumps({
            "status": "blocked", "error": {"type": type(exc).__name__, "message": str(exc)},
        }, sort_keys=True))
        return 2
    print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False))
    return 0 if result.status in {"sealed_complete", "sealed_partial", "adopted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
