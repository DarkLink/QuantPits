import os
import json
import pytest
import yaml
from quantpits.scripts.deep_analysis.training_context import TrainingModeContext

def test_training_context_parsing(tmp_path):
    # Setup mock workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "data").mkdir()

    # 1. Setup latest_train_records.json
    records = {
        "anchor_date": "2026-07-03",
        "models": {
            "lstm_Alpha158@static": "run_id_lstm_static",
            "lstm_Alpha158@cpcv": "run_id_lstm_cpcv",
            "catboost_Alpha158@static": "run_id_catboost_static",
            "gru_Alpha360@rolling": "run_id_gru_rolling"
        }
    }
    with open(workspace / "latest_train_records.json", "w") as f:
        json.dump(records, f)

    # 2. Setup rolling_config.yaml
    rolling_cfg = {
        "rolling_start": "2017-01-01",
        "train_years": 5,
        "valid_years": 2,
        "test_step": "3M"
    }
    with open(workspace / "config" / "rolling_config.yaml", "w") as f:
        yaml.dump(rolling_cfg, f)

    # 3. Setup rolling states
    state_slide = {
        "anchor_date": "2026-06-18",
        "total_windows": 10,
        "current_window_idx": 9,
        "current_model": "gru_Alpha360",
        "training_method": "slide",
        "completed_windows": {
            "9": {"gru_Alpha360": "run_id_gru_rolling"},
        },
    }
    with open(workspace / "data" / "rolling_state.json", "w") as f:
        json.dump(state_slide, f)

    # 4. Setup model_config.json
    model_cfg = {
        "purged_cv": {
            "n_groups": 10,
            "n_test_groups": 2,
            "n_val_groups": 1,
            "purge_steps": 5,
            "embargo_steps": 10
        }
    }
    with open(workspace / "config" / "model_config.json", "w") as f:
        json.dump(model_cfg, f)

    # Load context
    tc = TrainingModeContext.from_workspace(str(workspace))

    # Verify fields
    assert tc.anchor_date == "2026-07-03"
    assert tc.available_modes == ["cpcv", "rolling", "static"]
    
    # Verify models_by_name structure
    assert "lstm_Alpha158" in tc.models_by_name
    assert tc.models_by_name["lstm_Alpha158"]["static"] == "run_id_lstm_static"
    assert tc.models_by_name["lstm_Alpha158"]["cpcv"] == "run_id_lstm_cpcv"

    # Verify query helpers
    assert tc.get_cross_mode_models() == ["lstm_Alpha158"]
    assert tc.get_models_with_mode("cpcv") == ["lstm_Alpha158"]
    assert tc.get_models_with_mode("static") == ["catboost_Alpha158", "lstm_Alpha158"]

    # Verify gap days (2026-07-03 - 2026-06-18 = 15 days)
    assert tc.get_rolling_gap_days("rolling") == 15
    assert tc.get_rolling_gap_days("cpcv_rolling") is None

    # Verify resolve model key
    assert tc.resolve_model_key("run_id_lstm_cpcv") == "lstm_Alpha158@cpcv"
    assert tc.resolve_model_key("non_existent") is None


def test_training_context_uses_authoritative_rolling_state_inspection(tmp_path):
    from quantpits.rolling.state import inspect_rolling_state

    workspace = tmp_path / "Demo_Workspace"
    (workspace / "config").mkdir(parents=True)
    (workspace / "data").mkdir()
    state_path = workspace / "data" / "rolling_state.json"
    state_path.write_text(json.dumps({
        "anchor_date": "2026-06-18",
        "training_method": "slide",
        "completed_windows": {"0": {"demo": "recorder-demo"}},
        "current_window_idx": 0,
        "current_model": "demo",
        "total_windows": 1,
    }), encoding="utf-8")

    context = TrainingModeContext.from_workspace(str(workspace))
    direct = inspect_rolling_state(state_path, workspace)
    observed = context.rolling_state_inspections["rolling"]

    assert observed.to_public_dict() == direct.to_public_dict()
    assert observed.fingerprint == direct.fingerprint
    assert context.rolling_states["rolling"]["anchor_date"] == "2026-06-18"


def test_training_context_surfaces_corrupt_state_instead_of_treating_it_as_missing(tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    (workspace / "config").mkdir(parents=True)
    (workspace / "data").mkdir()
    (workspace / "data" / "rolling_state.json").write_text(
        "{broken", encoding="utf-8",
    )

    context = TrainingModeContext.from_workspace(str(workspace))

    inspection = context.rolling_state_inspections["rolling"]
    assert inspection.classification == "corrupt"
    assert inspection.reason_code == "rolling_state_corrupt"
    assert "rolling" not in context.rolling_states
