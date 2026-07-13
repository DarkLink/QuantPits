from types import SimpleNamespace

import pandas as pd
import pytest

from quantpits.ensemble.input_integrity import (
    PredictionFreshnessError,
    PredictionLoadIntegrityError,
    load_strict_prediction_bundle,
)


class Recorder:
    def __init__(self, record_id, artifact_uri, pred):
        self.info = {"id": record_id, "experiment_id": "1", "artifact_uri": artifact_uri}
        self.pred = pred

    def load_object(self, name):
        assert name == "pred.pkl"
        return self.pred

    def list_metrics(self):
        return {"ICIR": 0.2}


def _pred(date="2026-07-10"):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date), "AAA")], names=["datetime", "instrument"]
    )
    return pd.DataFrame({"score": [1.0]}, index=index)


def test_strict_bundle_records_lineage_and_exact_anchor(tmp_path):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    recorder = Recorder("r1", f"file://{root / 'mlruns/1/r1/artifacts'}", _pred())
    bundle = load_strict_prediction_bundle(
        {"experiment_name": "Source", "models": {"m@static": "r1"}},
        ["m@static"], workspace_root=root, expected_anchor="2026-07-10",
        recorder_getter=lambda *args: recorder,
    )
    assert bundle.loaded_models == ("m@static",)
    assert bundle.evidence[0].artifact_path.startswith("mlruns/")
    assert bundle.evidence[0].prediction_end == "2026-07-10"


def test_strict_bundle_rejects_stale_member(tmp_path):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    recorder = Recorder("r1", f"file://{root / 'mlruns/1/r1/artifacts'}", _pred("2026-07-02"))
    with pytest.raises(PredictionFreshnessError) as caught:
        load_strict_prediction_bundle(
            {"experiment_name": "Source", "models": {"m@static": "r1"}},
            ["m@static"], workspace_root=root, expected_anchor="2026-07-10",
            recorder_getter=lambda *args: recorder,
        )
    assert caught.value.evidence[0].status == "stale"


def test_strict_bundle_aggregates_missing_and_external(tmp_path):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    external = Recorder("r2", f"file://{tmp_path / 'external'}", _pred())
    records = {"m1@static": "r1", "m2@static": "r2"}

    def getter(record_id, experiment_name):
        if record_id == "r1":
            raise FileNotFoundError(record_id)
        return external

    with pytest.raises(PredictionLoadIntegrityError) as caught:
        load_strict_prediction_bundle(
            {"experiment_name": "Source", "models": records}, list(records),
            workspace_root=root, expected_anchor="2026-07-10", recorder_getter=getter,
        )
    assert len(caught.value.evidence) == 2
    assert all(item.status != "ready" for item in caught.value.evidence)
