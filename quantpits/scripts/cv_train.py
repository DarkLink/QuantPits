#!/usr/bin/env python
"""
CPCV Training Script (Purged Cross-Validation)

Trains K models per model_name using PurgedKFold cross-validation
following Marcos Lopez de Prado's Advances in Financial Machine Learning.
Each fold trains independently with purged, potentially discontiguous
training segments. Final predictions are averaged across folds.

Usage:
  python quantpits/scripts/cv_train.py --all-enabled
  python quantpits/scripts/cv_train.py --models lightgbm_Alpha158,gru_Alpha158
  python quantpits/scripts/cv_train.py --dry-run --all-enabled
  python quantpits/scripts/cv_train.py --predict-only --all-enabled

Modes:
  --full:           Train all enabled models with CPCV, overwrite records
  (default):        Incremental CPCV training on selected models, merge records
  --predict-only:   Use existing CPCV models to predict on new data
  --dry-run:        Preview fold/window plan without training
  --resume:         Resume from last interruption

Configuration:
  Add the "purged_cv" config block in config/model_config.json.
  CPCV operates independently of data_slice_mode.
  See docs/01_TRAINING_GUIDE.md for details.
"""

import sys
import json
import argparse

from quantpits.utils import env

# ================= CLI =================
def build_parser():
    parser = argparse.ArgumentParser(
        description='CPCV Training: Purged Cross-Validation with K-fold ensemble',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all-enabled                    # CPCV train all enabled models
  %(prog)s --models lightgbm_Alpha158       # CPCV train a single model
  %(prog)s --tag tree                       # CPCV train all tree models
  %(prog)s --dry-run --all-enabled          # Preview folds without training
  %(prog)s --predict-only --all-enabled     # Predict only with existing models
  %(prog)s --models gru --resume            # Resume interrupted CPCV training
  %(prog)s --list                           # List registered models
        """
    )

    mode = parser.add_argument_group('Run Mode')
    mode.add_argument('--full', action='store_true',
                      help='Full CPCV training: train all enabled models, '
                           'overwrite latest_train_records.json')
    mode.add_argument('--predict-only', action='store_true',
                      help='Predict only: use existing CPCV models on new data')

    select = parser.add_argument_group('Model Selection')
    select.add_argument('--models', type=str,
                        help='Model names, comma-separated (e.g., gru,lightgbm_Alpha158)')
    select.add_argument('--algorithm', type=str,
                        help='Filter by algorithm (e.g., lstm, gru, lightgbm)')
    select.add_argument('--dataset', type=str,
                        help='Filter by dataset (e.g., Alpha158, Alpha360)')
    select.add_argument('--market', type=str,
                        help='Filter by market (e.g., csi300)')
    select.add_argument('--tag', type=str,
                        help='Filter by tag (e.g., ts, tree, attention)')
    select.add_argument('--all-enabled', action='store_true',
                        help='All models with enabled=true')

    skip_group = parser.add_argument_group('Exclude & Skip')
    skip_group.add_argument('--skip', type=str,
                            help='Skip models, comma-separated')
    skip_group.add_argument('--resume', action='store_true',
                            help='Resume from last interruption (skip completed models)')

    ctrl = parser.add_argument_group('Run Control')
    ctrl.add_argument('--dry-run', action='store_true',
                      help='Preview plan only, do not train')
    ctrl.add_argument('--experiment-name', type=str, default=None,
                      help='MLflow experiment name '
                           '(default: Prod_Train_CPCV_{FREQ})')
    ctrl.add_argument('--no-pretrain', action='store_true',
                      help='Force random-weight init for basemodels')
    ctrl.add_argument('--source-records', type=str,
                      default='latest_train_records.json',
                      help='Source records file for predict-only')
    ctrl.add_argument('--cache-size', type=int, default=None, metavar='MB',
                      help='Handler cache max memory (MB). Default: auto-detect '
                           '(50%% free RAM). Set 0 to disable.')
    ctrl.add_argument('--workspace', default=None, help='Explicit workspace root')
    ctrl.add_argument('--explain-plan', action='store_true', help='Print a lightweight plan without Qlib or writes')
    ctrl.add_argument('--json-plan', action='store_true', help='Print the lightweight plan as one JSON document')
    ctrl.add_argument('--run-id', default=None, help='Explicit run ID for plan/manifest alignment')
    ctrl.add_argument('--no-manifest', action='store_true', help='Do not write a RunManifest during execution')

    info = parser.add_argument_group('Information')
    info.add_argument('--list', action='store_true',
                      help='List model registry (supports filters)')
    info.add_argument('--show-state', action='store_true',
                      help='Show last run state')
    info.add_argument('--clear-state', action='store_true',
                      help='Clear run state file')

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


# ================= Helpers =================
def _resolve_targets(args, registry):
    """Determine which models to train based on CLI args.

    Delegates to the shared resolve_target_models() in train_utils.
    """
    from quantpits.utils.train_utils import resolve_target_models

    targets = resolve_target_models(args, registry)
    if targets is None or not targets:
        from quantpits.training.errors import TrainingPlanError
        raise TrainingPlanError("specify --models, --all-enabled, --full, or a filter")

    # Apply skip
    if args.skip:
        skip_names = set(s.strip() for s in args.skip.split(',') if s.strip())
        targets = {k: v for k, v in targets.items() if k not in skip_names}

    return targets


# ================= Main =================
def main(argv=None):
    args = parse_args(argv)
    ctx = env.get_workspace_context(args.workspace)

    # --list
    if args.list:
        env.set_root_dir(str(ctx.root))
        from quantpits.utils.train_utils import load_model_registry, print_model_table
        registry = load_model_registry()
        targets = _resolve_targets(args, registry) if (
            args.models or args.algorithm or args.dataset or
            args.market or args.tag
        ) else registry
        print_model_table(targets, title="Model Registry")
        return

    # --show-state
    if args.show_state:
        path = ctx.data_path("run_state.json")
        if path.is_file():
            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, default=str))
        else:
            print("No run state file found.")
        return

    # --clear-state
    if args.clear_state:
        env.safeguard("CPCV Train: clear state", workspace_root=ctx.root)
        from quantpits.training.lease import TrainingExecutionLease
        from quantpits.training.state import TrainingStateRepository
        lease = TrainingExecutionLease.for_workspace(ctx)
        lease.acquire(run_id="clear-training-state")
        try:
            TrainingStateRepository(ctx.data_path("run_state.json")).clear()
        finally:
            lease.release()
        print("Run state cleared.")
        return

    from quantpits.training.command import (
        options_from_namespace, prepare_training_run, prepared_plan_json, render_prepared_plan,
    )
    from quantpits.training.errors import TrainingCommandError
    from quantpits.training.service import TrainingExecutionService, default_execution_hooks
    try:
        options = options_from_namespace(args, "cpcv")
        cli_args = tuple(argv if argv is not None else sys.argv[1:])
        prepared = prepare_training_run(ctx=ctx, options=options, cli_args=cli_args)
        if options.json_plan:
            print(json.dumps(prepared_plan_json(prepared), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if options.explain_plan:
            print(render_prepared_plan(prepared))
            return 0
        env.safeguard("CPCV Train", workspace_root=ctx.root)
        service = TrainingExecutionService(default_execution_hooks(
            activate_workspace=env.set_root_dir,
            init_qlib=env.init_qlib,
        ))
        service.execute(prepared)
        return 0
    except TrainingCommandError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return exc.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
