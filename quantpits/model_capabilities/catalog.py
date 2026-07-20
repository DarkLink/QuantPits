"""Repository-owned model capability declarations.

This catalog is public and sanitized.  It never discovers declarations from a
workspace registry, workflow, recorder, or backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .contracts import ACTIONS, EXECUTION_FAMILIES, RawModelCapabilityDeclaration


_PAIRED_MODELS = (
    ("pytorch_adarnn", "ADARNN", "point_in_time"),
    ("pytorch_alstm", "ALSTM", "point_in_time"),
    ("pytorch_alstm_ts", "ALSTM", "time_series"),
    ("pytorch_gats_plus", "GATsPlus", "daily_market_label"),
    ("pytorch_gats_ts", "GATs", "time_series"),
    ("pytorch_gru", "GRU", "point_in_time"),
    ("pytorch_igmtf", "IGMTF", "point_in_time"),
    ("pytorch_krnn", "KRNN", "point_in_time"),
    ("pytorch_localformer", "LocalformerModel", "point_in_time"),
    ("pytorch_localformer_ts", "LocalformerModelIC", "time_series"),
    ("pytorch_lstm", "LSTM", "point_in_time"),
    ("pytorch_sandwich", "Sandwich", "point_in_time"),
    ("pytorch_sfm", "SFM", "point_in_time"),
    ("pytorch_tabnet", "TabnetModel", "point_in_time"),
    ("pytorch_tcn", "TCN", "point_in_time"),
    ("pytorch_tcn_ts", "TCNIC", "time_series"),
    ("pytorch_tra", "TRAModelIC", "memory_time_series"),
    ("pytorch_transformer", "TransformerModel", "point_in_time"),
    ("pytorch_transformer_ts", "TransformerModelIC", "time_series"),
)

_CUSTOM_ONLY_MODELS = (
    ("pytorch_add", "ADD", "daily_market_label"),
    ("pytorch_general_nn", "GeneralPTNN", "point_in_time"),
    ("pytorch_lstm_ic_loss", "LSTMICModel", "time_series"),
    ("pytorch_lstm_rank", "LSTMRankModel", "time_series"),
    ("pytorch_tcts", "TCTS", "multi_label"),
)

_DATASET = {
    "point_in_time": ("qlib.data.dataset", "DatasetH", "standard_infer_no_label_drop"),
    "time_series": ("qlib.data.dataset", "TSDatasetH", "sequence_infer_no_label_drop"),
    "memory_time_series": ("qlib.contrib.data.dataset", "MTSDatasetH", "memory_infer_tail_preserved"),
    "daily_market_label": ("qlib.data.dataset", "DatasetH", "daily_market_label_infer_tail_preserved"),
    "multi_label": ("qlib.data.dataset", "DatasetH", "multi_label_cardinality_preserved"),
}

_PURGED_DATASET = {
    "point_in_time": ("quantpits.data.cpcv_dataset", "PurgedDatasetH"),
    "time_series": ("quantpits.data.cpcv_dataset", "PurgedTSDatasetH"),
    "memory_time_series": ("quantpits.data.cpcv_dataset", "PurgedMTSDatasetH"),
}

_REQUIRED_PREDICATES = (
    "identity_canonical", "catalog_assigned", "dependency_available", "module_imported",
    "class_resolved", "constructor_signature", "fit_signature", "predict_signature",
    "device_available", "protocol_adapter", "capability_identity_match", "action_protocol", "dataset_protocol",
    "processor_tail_safe", "artifact_roundtrip", "prediction_shape",
    "prediction_tail", "prediction_gap", "prediction_unique", "prediction_finite",
    "wrapper_identity_match", "environment_isolated",
)


def repository_wrapper_inventory() -> Tuple[str, ...]:
    """Return the ordered checked-in wrapper module inventory without importing it."""
    wrapper_root = Path(__file__).resolve().parents[1] / "utils" / "model_wrappers"
    modules = []
    for family in ("custom", "lh"):
        for path in sorted((wrapper_root / family).glob("*.py")):
            if path.name == "__init__.py":
                continue
            modules.append("quantpits.utils.model_wrappers.%s.%s" % (family, path.stem))
    return tuple(modules)


def declared_repository_models() -> Tuple[Tuple[str, str], ...]:
    models = []
    for stem, class_name, _protocol in _PAIRED_MODELS:
        models.append(("quantpits.utils.model_wrappers.custom.%s" % stem, class_name))
        models.append(("quantpits.utils.model_wrappers.lh.%s" % stem, class_name))
    for stem, class_name, _protocol in _CUSTOM_ONLY_MODELS:
        models.append(("quantpits.utils.model_wrappers.custom.%s" % stem, class_name))
    return tuple(sorted(models))


def _model_profiles() -> Iterable[Tuple[str, str, str, str]]:
    for stem, class_name, protocol in _PAIRED_MODELS:
        yield ("quantpits.utils.model_wrappers.custom.%s" % stem, class_name, "custom", protocol)
        yield ("quantpits.utils.model_wrappers.lh.%s" % stem, class_name, "loss_history", protocol)
    for stem, class_name, protocol in _CUSTOM_ONLY_MODELS:
        yield ("quantpits.utils.model_wrappers.custom.%s" % stem, class_name, "custom", protocol)


def _dataset_for(protocol: str, family: str) -> Tuple[str, str, str]:
    module, class_name, processor = _DATASET[protocol]
    if family in ("cpcv", "cpcv_rolling") and protocol in _PURGED_DATASET:
        module, class_name = _PURGED_DATASET[protocol]
    return module, class_name, processor


def authoritative_catalog() -> Tuple[RawModelCapabilityDeclaration, ...]:
    """Build the ordered atomic public catalog.

    Every model is expanded across the four action and four execution-family
    vocabularies.  Unsupported projections remain visible and are classified by
    the inspector; no successful subset can redefine this raw inventory.
    """
    declarations = []
    for module, class_name, kind, protocol in _model_profiles():
        for action in ACTIONS:
            for family in EXECUTION_FAMILIES:
                dataset_module, dataset_class, processor = _dataset_for(protocol, family)
                declarations.append(RawModelCapabilityDeclaration(
                    module, class_name, kind, dataset_module, dataset_class, protocol,
                    action, family, processor, "qlib_recorder_model_v1", "python_qlib_torch",
                    _REQUIRED_PREDICATES, True,
                ))
    for action in ACTIONS:
        for family in EXECUTION_FAMILIES:
            dataset_module, dataset_class, processor = _dataset_for("point_in_time", family)
            declarations.append(RawModelCapabilityDeclaration(
                "qlib.contrib.model.linear", "LinearModel", "external_passthrough",
                dataset_module, dataset_class, "point_in_time", action, family, processor,
                "qlib_recorder_model_v1", "python_qlib", _REQUIRED_PREDICATES, True,
            ))
    return tuple(declarations)


AUTHORITATIVE_CATALOG = authoritative_catalog()
