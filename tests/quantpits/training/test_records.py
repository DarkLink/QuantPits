import pytest
import pandas as pd

from quantpits.training.records import (
    ModelRecordEntry, ModelRecordOutcome, TrainingRecordSnapshot,
    resolve_model_record, snapshot_from_dict,
    build_model_record_entry,
)


def entry(key="m@cpcv", experiment="predict"):
    return ModelRecordEntry(
        key=key, model_name=key.rsplit("@", 1)[0], training_mode=key.rsplit("@", 1)[1],
        operation="cpcv_predict", status="ready", recorder_id="rid",
        experiment_name=experiment, requested_anchor="2026-07-10",
        prediction_start="2026-07-09", prediction_end="2026-07-10", prediction_rows=2,
        source_recorder_id="source", source_experiment_name="train", source_operation="train",
    )


def test_v2_projection_and_resolution_prefers_per_model_identity():
    data = TrainingRecordSnapshot((entry(),)).to_dict()
    data["cpcv_experiment_name"] = "stale"
    resolved = resolve_model_record(data, "m@cpcv")
    assert resolved.experiment_name == "predict"
    assert data["models"] == {"m@cpcv": "rid"}


def test_v1_is_readable_without_mutation():
    original = {"experiment_name": "legacy", "anchor_date": "2026-07-10", "models": {"m@static": "r"}}
    snapshot = snapshot_from_dict(original)
    assert snapshot.entries[0].status == "legacy_unverified"
    assert "schema_version" not in original


def test_failed_outcome_does_not_require_entry():
    assert ModelRecordOutcome("m@static", "train", "failed", error_code="fit_failed").entry is None


def test_entry_rejects_key_field_mismatch():
    with pytest.raises(ValueError):
        ModelRecordEntry("m@static", "x", "static", "train", "ready", "r", "e")


def test_build_ready_entry_uses_persisted_prediction_coverage():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-07-10")], ["SH600000"]], names=["datetime", "instrument"]
    )
    class Recorder:
        info = {"id": "rid", "experiment_id": "1"}
        def load_object(self, name):
            assert name == "pred.pkl"
            return pd.DataFrame({"score": [0.1]}, index=index)
    value = build_model_record_entry(
        key="m@static", operation="predict_only", experiment_name="predict",
        recorder=Recorder(), requested_anchor="2026-07-10", dataset_test_end="2026-07-10",
        source_recorder_id="source", source_experiment_name="train",
        source_operation="train",
    )
    assert value.status == "ready"
    assert value.prediction_end == "2026-07-10"
    assert value.source_recorder_id == "source"


def test_snapshot_rejects_source_lineage_cycle():
    common = dict(requested_anchor="2026-07-10", prediction_start="2026-07-10", prediction_end="2026-07-10", prediction_rows=1, source_operation="train")
    first = ModelRecordEntry("a@static", "a", "static", "predict_only", "ready", "r1", "exp", source_recorder_id="r2", source_experiment_name="exp", **common)
    second = ModelRecordEntry("b@static", "b", "static", "predict_only", "ready", "r2", "exp", source_recorder_id="r1", source_experiment_name="exp", **common)
    with pytest.raises(ValueError, match="cycle"):
        TrainingRecordSnapshot((first, second))


def test_explicit_v2_never_falls_back_to_v1():
    with pytest.raises(ValueError, match="model_records"):
        snapshot_from_dict({"schema_version": 2, "models": {"m@static": "r"}, "experiment_name": "stale"})


def test_outcome_rejects_entry_for_another_model():
    value = ModelRecordEntry(
        "b@static", "b", "static", "train", "ready", "r", "e",
        requested_anchor="2026-07-10", prediction_start="2026-07-10",
        prediction_end="2026-07-10", prediction_rows=1,
    )
    with pytest.raises(ValueError, match="outcome key"):
        ModelRecordOutcome("a@static", "train", "success", value)


def test_snapshot_render_is_deterministic_without_inventing_time():
    value = TrainingRecordSnapshot((entry(),))
    assert value.to_dict() == value.to_dict()
    assert "updated_at" not in value.to_dict()


def test_builder_rejects_external_artifact(tmp_path):
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-07-10")], ["SH600000"]], names=["datetime", "instrument"]
    )
    class Recorder:
        info = {"id": "rid", "artifact_uri": "file://%s" % (tmp_path.parent / "external")}
        def load_object(self, name): return pd.DataFrame({"score": [0.1]}, index=index)
    with pytest.raises(ValueError, match="outside"):
        build_model_record_entry(
            key="m@static", operation="train", experiment_name="exp", recorder=Recorder(),
            requested_anchor="2026-07-10", workspace_root=tmp_path,
        )
