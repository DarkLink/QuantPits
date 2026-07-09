"""Ensemble runtime service primitives."""

from quantpits.ensemble.analytics import (
    CorrelationAnalysisRequest,
    CorrelationAnalysisResult,
    ModelContributionSaveResult,
    build_model_contribution_payload,
    calculate_loo_contribution,
    compute_prediction_correlation,
    run_correlation_analysis,
    save_model_contribution_snapshot,
    summarize_correlation_matrix,
)
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
    "CorrelationAnalysisRequest",
    "CorrelationAnalysisResult",
    "EnsembleExecutionHooks",
    "EnsembleFusionService",
    "EnsembleRunConfig",
    "EnsembleRunOptions",
    "EnsembleRunSummary",
    "FusionLedgerEntry",
    "ModelContributionSaveResult",
    "PredictionSaveRequest",
    "PredictionSaveResult",
    "PreparedEnsembleRun",
    "append_fusion_ledger",
    "build_model_contribution_payload",
    "build_fusion_ledger_record",
    "calculate_loo_contribution",
    "compute_prediction_correlation",
    "load_ensemble_run_config",
    "options_from_namespace",
    "options_to_namespace",
    "prepare_ensemble_run",
    "prepared_plan_json",
    "render_prepared_plan",
    "run_correlation_analysis",
    "save_ensemble_predictions",
    "save_model_contribution_snapshot",
    "summarize_correlation_matrix",
]
