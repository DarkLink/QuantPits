"""Ensemble runtime service primitives."""

from quantpits.ensemble.config import load_ensemble_run_config
from quantpits.ensemble.ledger import (
    FusionLedgerEntry,
    append_fusion_ledger,
    build_fusion_ledger_record,
)
from quantpits.ensemble.persistence import (
    PredictionSaveRequest,
    PredictionSaveResult,
    save_ensemble_predictions,
)
from quantpits.ensemble.service import (
    EnsembleFusionService,
    prepare_ensemble_run,
    prepared_plan_json,
    render_prepared_plan,
)
from quantpits.ensemble.types import (
    EnsembleExecutionHooks,
    EnsembleRunConfig,
    EnsembleRunOptions,
    EnsembleRunSummary,
    PreparedEnsembleRun,
    options_from_namespace,
    options_to_namespace,
)

__all__ = [
    "EnsembleExecutionHooks",
    "EnsembleFusionService",
    "EnsembleRunConfig",
    "EnsembleRunOptions",
    "EnsembleRunSummary",
    "FusionLedgerEntry",
    "PredictionSaveRequest",
    "PredictionSaveResult",
    "PreparedEnsembleRun",
    "append_fusion_ledger",
    "build_fusion_ledger_record",
    "load_ensemble_run_config",
    "options_from_namespace",
    "options_to_namespace",
    "prepare_ensemble_run",
    "prepared_plan_json",
    "render_prepared_plan",
    "save_ensemble_predictions",
]
