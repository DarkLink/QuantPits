import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from quantpits.ensemble.persistence import (
    PredictionSaveRequest,
    save_ensemble_predictions,
)


def _score_series() -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-01-05"), "AAA")],
        names=["datetime", "instrument"],
    )
    return pd.Series([1.0], index=index)


def test_save_ensemble_predictions_updates_records_and_config(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    records_path = workspace / "config" / "ensemble_records.json"
    records_path.write_text(
        json.dumps({
            "combos": {"old_combo": "old_rec"},
            "default_combo": "old_combo",
            "future_field": {"preserve": True},
        }),
        encoding="utf-8",
    )

    request = PredictionSaveRequest(
        final_score=_score_series(),
        anchor_date="2026-01-05",
        experiment_name="exp1",
        method="equal",
        model_names=("M1", "M2"),
        model_metrics={"M1": 0.12345, "M2": 0.23456},
        static_weights={"M1": 0.5, "M2": 0.5},
        is_dynamic=False,
        output_dir="output/ensemble",
        combo_name="combo_a",
        is_default=True,
        workspace_root=workspace,
    )

    with patch(
        "quantpits.utils.predict_utils.save_predictions_to_recorder",
        return_value="rec_123",
    ) as saver:
        result = save_ensemble_predictions(request)

    assert result.recorder_id == "rec_123"
    assert result.returned_ref == "rec_123"
    saver.assert_called_once()
    _, kwargs = saver.call_args
    assert kwargs["experiment_name"] == "Ensemble_Fusion"
    assert kwargs["model_name"] == "combo_a"
    assert kwargs["tags"]["weight_mode"] == "equal"

    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert records["combos"]["old_combo"] == "old_rec"
    assert records["future_field"] == {"preserve": True}
    assert records["combos"]["combo_a"] == "rec_123"
    assert records["default_combo"] == "combo_a"
    assert records["default_record_id"] == "rec_123"
    assert records["_schema_version"] == 2
    assert records["combo_meta"]["combo_a"]["record_id"] == "rec_123"

    config_path = workspace / "output" / "ensemble" / "ensemble_fusion_config_combo_a_2026-01-05.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["combo_name"] == "combo_a"
    assert config["is_default"] is True
    assert config["models_used"] == ["M1", "M2"]
    assert config["model_metrics"] == {"M1": 0.1235, "M2": 0.2346}
    assert config["static_weights"] == {"M1": 0.5, "M2": 0.5}


def test_output_verification_failure_leaves_existing_pointer_unchanged(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    records_path = workspace / "config" / "ensemble_records.json"
    original = {"combos": {"default": "old"}, "default_combo": "default"}
    records_path.write_text(json.dumps(original), encoding="utf-8")
    request = PredictionSaveRequest(
        final_score=_score_series(), anchor_date="2026-01-05", experiment_name="exp",
        method="equal", model_names=("M1",), model_metrics={}, static_weights={"M1": 1.0},
        is_dynamic=False, output_dir="output/ensemble", combo_name="default",
        is_default=True, workspace_root=workspace,
    )
    with patch("quantpits.utils.predict_utils.save_predictions_to_recorder", return_value="new"):
        with pytest.raises(ValueError, match="unsafe"):
            save_ensemble_predictions(
                request,
                output_inspector=lambda *a, **k: (_ for _ in ()).throw(ValueError("unsafe recorder")),
            )
    assert json.loads(records_path.read_text(encoding="utf-8")) == original


def test_atomic_replace_failure_leaves_old_records_intact(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    records_path = workspace / "config/ensemble_records.json"
    original = {"combos": {"default": "old"}, "default_combo": "default"}
    records_path.write_text(json.dumps(original), encoding="utf-8")
    request = PredictionSaveRequest(
        final_score=_score_series(), anchor_date="2026-01-05", experiment_name="exp",
        method="equal", model_names=("M1",), model_metrics={}, static_weights={"M1": 1.0},
        is_dynamic=False, output_dir="output/ensemble", combo_name="default",
        is_default=True, workspace_root=workspace,
    )
    with patch("quantpits.utils.predict_utils.save_predictions_to_recorder", return_value="new"):
        with patch("quantpits.ensemble.persistence.os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                save_ensemble_predictions(request)
    assert json.loads(records_path.read_text(encoding="utf-8")) == original


def test_verified_output_records_tags_and_lineage_before_pointer_update(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    seen = {}

    def inspector(record_id, **kwargs):
        seen.update(kwargs)
        return {
            "recorder_id": record_id,
            "experiment_id": "demo-experiment",
            "artifact_path": "mlruns/demo/artifacts",
            "contained": True,
        }

    request = PredictionSaveRequest(
        final_score=_score_series(), anchor_date="2026-01-05", experiment_name="source",
        method="equal", model_names=("M1",), model_metrics={}, static_weights={"M1": 1.0},
        is_dynamic=False, output_dir="output/ensemble", combo_name="default", is_default=True,
        workspace_root=workspace, source_recorders={"M1": "source-recorder"},
        source_anchors={"M1": "2026-01-05"}, run_id="demo-run", plan_fingerprint="abc123",
    )
    with patch("quantpits.utils.predict_utils.save_predictions_to_recorder", return_value="new"):
        save_ensemble_predictions(request, output_inspector=inspector)

    assert seen["expected_tags"]["mode"] == "fused_prediction"
    assert seen["expected_tags"]["plan_fingerprint"] == "abc123"
    records = json.loads((workspace / "config/ensemble_records.json").read_text())
    meta = records["combo_meta"]["default"]
    assert meta["source_recorders"] == {"M1": "source-recorder"}
    assert meta["source_anchors"] == {"M1": "2026-01-05"}
    snapshot = json.loads(
        (workspace / "output/ensemble/ensemble_fusion_config_default_2026-01-05.json").read_text()
    )
    assert snapshot["output_recorder"]["artifact_path"] == "mlruns/demo/artifacts"


def test_save_ensemble_predictions_non_default_preserves_default_pointer(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    records_path = workspace / "config" / "ensemble_records.json"
    records_path.write_text(
        json.dumps(
            {
                "combos": {"default_combo": "rec_default"},
                "default_combo": "default_combo",
                "default_record_id": "rec_default",
            }
        ),
        encoding="utf-8",
    )

    request = PredictionSaveRequest(
        final_score=_score_series(),
        anchor_date="2026-01-05",
        experiment_name="exp1",
        method="equal",
        model_names=("M1",),
        model_metrics={"M1": 0.1},
        static_weights={"M1": 1.0},
        is_dynamic=False,
        output_dir="output/ensemble",
        combo_name="candidate_combo",
        is_default=False,
        workspace_root=workspace,
    )

    with patch(
        "quantpits.utils.predict_utils.save_predictions_to_recorder",
        return_value="rec_candidate",
    ):
        save_ensemble_predictions(request)

    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert records["combos"]["candidate_combo"] == "rec_candidate"
    assert records["default_combo"] == "default_combo"
    assert records["default_record_id"] == "rec_default"


def test_save_ensemble_predictions_csv_paths_bind_to_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    request = PredictionSaveRequest(
        final_score=_score_series(),
        anchor_date="2026-01-05",
        experiment_name="exp1",
        method="equal",
        model_names=("M1",),
        model_metrics={"M1": 0.1},
        static_weights={"M1": 1.0},
        is_dynamic=False,
        output_dir="output/ensemble",
        combo_name="combo_a",
        is_default=True,
        save_csv=True,
        workspace_root=workspace,
    )

    with patch(
        "quantpits.utils.predict_utils.save_predictions_to_recorder",
        return_value="rec_123",
    ):
        result = save_ensemble_predictions(request)

    combo_csv = workspace / "output" / "predictions" / "ensemble_combo_a_2026-01-05.csv"
    compat_csv = workspace / "output" / "predictions" / "ensemble_2026-01-05.csv"
    assert result.returned_ref == str(combo_csv)
    assert result.prediction_csv_path == combo_csv
    assert result.compatibility_csv_path == compat_csv
    assert combo_csv.exists()
    assert compat_csv.exists()
    assert not (other_cwd / "output").exists()


def test_save_ensemble_predictions_respects_absolute_paths(tmp_path):
    workspace = tmp_path / "workspace"
    output_dir = tmp_path / "absolute_output"
    prediction_dir = tmp_path / "absolute_predictions"

    request = PredictionSaveRequest(
        final_score=_score_series(),
        anchor_date="2026-01-05",
        experiment_name="exp1",
        method="dynamic",
        model_names=("M1",),
        model_metrics={"M1": 0.1},
        static_weights=None,
        is_dynamic=True,
        output_dir=output_dir,
        combo_name=None,
        save_csv=True,
        prediction_dir=prediction_dir,
        workspace_root=workspace,
    )

    with patch(
        "quantpits.utils.predict_utils.save_predictions_to_recorder",
        return_value="rec_abs",
    ):
        result = save_ensemble_predictions(request)

    assert result.returned_ref == str(prediction_dir / "ensemble_2026-01-05.csv")
    assert (prediction_dir / "ensemble_2026-01-05.csv").exists()
    assert (output_dir / "ensemble_fusion_config_2026-01-05.json").exists()
    assert (workspace / "config" / "ensemble_records.json").exists()
