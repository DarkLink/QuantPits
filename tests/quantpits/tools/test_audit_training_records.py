import json

import quantpits.tools.audit_training_records as audit_module
from quantpits.tools.audit_training_records import main
from quantpits.tools.audit_training_records import _qlib_recorder_getter, build_parser, run
from quantpits.training.records import ModelRecordEntry, TrainingRecordSnapshot
import pandas as pd


def test_default_prediction_recorder_getter_uses_keyword_only_qlib_api(monkeypatch):
    from qlib.workflow import R

    recorder = object()
    calls = []

    def get_recorder(*, recorder_id=None, experiment_name=None, **kwargs):
        calls.append((recorder_id, experiment_name, kwargs))
        return recorder

    monkeypatch.setattr(R, "get_recorder", get_recorder)

    assert _qlib_recorder_getter("source-recorder", "source-experiment") is recorder
    assert calls == [("source-recorder", "source-experiment", {})]


def test_audit_is_read_only_and_reports_legacy(tmp_path, capsys):
    path = tmp_path / "latest_train_records.json"
    path.write_text(json.dumps({"experiment_name": "exp", "models": {"m@static": "r"}}))
    before = path.read_bytes()
    assert main(["--workspace", str(tmp_path), "--json"]) == 2
    assert path.read_bytes() == before
    assert "legacy_record_schema" in capsys.readouterr().out


def test_optional_mlflow_and_prediction_verification_is_read_only(tmp_path):
    (tmp_path / "mlflow.db").write_bytes(b"metadata")
    entry = ModelRecordEntry(
        "m@static", "m", "static", "train", "ready", "rid", "exp",
        requested_anchor="2026-07-10", prediction_start="2026-07-10",
        prediction_end="2026-07-10", prediction_rows=1,
    )
    path = tmp_path / "latest_train_records.json"
    path.write_text(json.dumps(TrainingRecordSnapshot((entry,)).to_dict()))
    before = path.read_bytes()
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-07-10"), "AAA")], names=["datetime", "instrument"]
    )
    class Client:
        def tracking_uri(self): return "sqlite:///%s" % (tmp_path / "mlflow.db")
        def experiments_by_name(self, name):
            return [{"experiment_id": "1", "name": name, "artifact_location": "file://%s" % (tmp_path / "mlruns/1")}]
        def recorder(self, rid, experiment_name=None):
            return {"id": rid, "experiment_id": "1", "experiment_name": experiment_name, "artifact_uri": "file://%s" % (tmp_path / "mlruns/1/rid/artifacts")}
    class Recorder:
        def load_object(self, name): return pd.DataFrame({"score": [1.0]}, index=index)
    args = build_parser().parse_args([
        "--workspace", str(tmp_path), "--verify-mlflow", "--verify-predictions",
    ])
    assert run(args, client=Client(), recorder_getter=lambda rid, exp: Recorder()) == 0
    assert path.read_bytes() == before


def test_default_prediction_verification_initializes_qlib_for_explicit_workspace(
    tmp_path, monkeypatch,
):
    (tmp_path / "mlflow.db").write_bytes(b"metadata")
    entry = ModelRecordEntry(
        "m@static", "m", "static", "train", "ready", "rid", "exp",
        requested_anchor="2026-07-10", prediction_start="2026-07-10",
        prediction_end="2026-07-10", prediction_rows=1,
    )
    path = tmp_path / "latest_train_records.json"
    path.write_text(json.dumps(TrainingRecordSnapshot((entry,)).to_dict()))
    before = path.read_bytes()
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-07-10"), "AAA")], names=["datetime", "instrument"]
    )

    class Client:
        def tracking_uri(self): return "sqlite:///%s" % (tmp_path / "mlflow.db")
        def experiments_by_name(self, name):
            return [{
                "experiment_id": "1", "name": name,
                "artifact_location": "file://%s" % (tmp_path / "mlruns/1"),
            }]
        def recorder(self, rid, experiment_name=None):
            return {
                "id": rid, "experiment_id": "1", "experiment_name": experiment_name,
                "artifact_uri": "file://%s" % (tmp_path / "mlruns/1/rid/artifacts"),
            }

    class Recorder:
        def load_object(self, name):
            return pd.DataFrame({"score": [1.0]}, index=index)

    initialized = []
    monkeypatch.setattr(
        audit_module, "_init_qlib_for_audit", lambda ctx: initialized.append(ctx)
    )
    monkeypatch.setattr(
        audit_module, "_qlib_recorder_getter", lambda rid, exp: Recorder()
    )
    args = build_parser().parse_args([
        "--workspace", str(tmp_path), "--verify-mlflow", "--verify-predictions",
    ])

    assert run(args, client=Client()) == 0
    assert len(initialized) == 1
    assert initialized[0].root == tmp_path.resolve()
    assert path.read_bytes() == before
