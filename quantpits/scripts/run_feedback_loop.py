#!/usr/bin/env python
"""
CLI entry point for the RLFF Feedback Loop.

Usage:
    # Report-only (preview)
    python -m quantpits.scripts.run_feedback_loop \\
        --action-items output/deep_analysis/action_items_2026-04-30.json \\
        --report-only

    # Execute (Playground + Adapter + Retrain + Report)
    python -m quantpits.scripts.run_feedback_loop \\
        --action-items output/deep_analysis/action_items_2026-04-30.json \\
        --execute

    # Promote (push validated changes to production)
    python -m quantpits.scripts.run_feedback_loop \\
        --action-items output/deep_analysis/action_items_2026-04-30.json \\
        --promote

    # With options:
    #   --dry-run               Adapter dry-run (no file writes)
    #   --models m1,m2          Only process specified models
    #   --skip-models m3,m4     Exclude specific models
    #   --max-duration-minutes N Time budget
    #   --skip-retrain          Skip retraining (config changes only)
"""

import argparse
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="RLFF Feedback Loop — ActionItem execution and promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--action-items",
        default=None,
        help="Path to action_items_{date}.json file (not required for --playground-only)",
    )

    # Mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--report-only",
        action="store_true",
        help="Preview changes without executing",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Create Playground, apply changes, retrain, validate, report",
    )
    mode_group.add_argument(
        "--promote",
        action="store_true",
        help="Promote validated changes from Playground to production",
    )
    mode_group.add_argument(
        "--auto-promote",
        action="store_true",
        help="(Not yet implemented) Auto-promote if validation passes",
    )
    mode_group.add_argument(
        "--playground-only",
        action="store_true",
        help="Lightweight: skip ActionItems, let ExperimentAnalyzer drive "
             "param search directly in Playground. Use with --models.",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Adapter dry-run — preview changes without writing files",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to process (overrides priority)",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        default=None,
        help="Comma-separated list of model names to exclude",
    )
    parser.add_argument(
        "--max-duration-minutes",
        type=int,
        default=None,
        help="Time budget in minutes; excess items are deferred",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip retraining (only apply config changes)",
    )
    parser.add_argument(
        "--max-experiment-rounds",
        type=int,
        default=3,
        help="Max retrain rounds per model in Playground (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip param adjustments already tried in previous experiment; "
             "continue with ExperimentAnalyzer suggesting new params",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine mode
    if args.report_only:
        mode = "report-only"
    elif args.execute:
        mode = "execute"
    elif args.promote:
        mode = "promote"
    elif args.auto_promote:
        mode = "auto-promote"
    elif args.playground_only:
        mode = "playground-only"
    else:
        mode = "report-only"

    # Parse model lists
    models = args.models.split(",") if args.models else None
    skip_models = args.skip_models.split(",") if args.skip_models else None

    # Resolve workspace
    from quantpits.utils.env import ROOT_DIR

    print("=" * 60)
    print(f"🔄  RLFF Feedback Loop")
    print(f"Mode           : {mode}")
    print(f"Workspace      : {ROOT_DIR}")
    if mode != "playground-only":
        print(f"Action Items   : {args.action_items}")
    if models:
        print(f"Models         : {models}")
    if skip_models:
        print(f"Skip models    : {skip_models}")
    if args.max_duration_minutes:
        print(f"Time budget    : {args.max_duration_minutes} min")
    print("=" * 60)

    # Resolve action items path (not needed for playground-only mode)
    action_items_path = args.action_items
    if mode == "playground-only":
        if not args.models:
            print("❌ --models is required for --playground-only mode")
            sys.exit(1)
        action_items_path = ""  # placeholder
    else:
        if not action_items_path:
            print("❌ --action-items is required for this mode")
            sys.exit(1)
        if not os.path.isabs(action_items_path):
            action_items_path = os.path.join(ROOT_DIR, action_items_path)
        if not os.path.exists(action_items_path):
            print(f"❌ ActionItems file not found: {action_items_path}")
            sys.exit(1)

    from quantpits.scripts.deep_analysis.feedback_loop import FeedbackLoop

    loop = FeedbackLoop(workspace_root=ROOT_DIR, mode=mode)
    report = loop.run(
        action_items_path=action_items_path if mode != "playground-only" else "",
        models=models,
        skip_models=skip_models,
        max_duration_minutes=args.max_duration_minutes,
        dry_run=args.dry_run,
        skip_retrain=args.skip_retrain,
        max_experiment_rounds=args.max_experiment_rounds,
        resume=args.resume,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"📋  {report.summary}")
    print(f"    Processed: {report.action_items_processed}")
    print(f"    Deferred:  {report.action_items_deferred}")
    if report.validation_results:
        n_passed = sum(1 for v in report.validation_results if v.get("passed"))
        print(f"    Validated: {n_passed}/{len(report.validation_results)} passed")
    if report.promote_result:
        pr = report.promote_result
        status = "✅ Success" if pr.get("success") else f"❌ Failed: {pr.get('error', '')}"
        print(f"    Promote:   {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
