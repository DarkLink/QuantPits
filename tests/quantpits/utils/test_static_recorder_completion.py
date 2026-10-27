"""Deterministic FileStore read/write window, with real Qlib recorder completion."""

import json
import threading
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.quantpits.utils.test_train_utils_pipeline import finite_signal_training, _SavedSignalConsumer


METRIC = "1week.excess_return_with_cost.std"


@pytest.fixture
def completion_probe(monkeypatch, mock_env_constants):
    import mlflow
    import mlflow.store.tracking.file_store as file_store
    from qlib.workflow.recorder import Recorder
    from quantpits.training import records

    train_utils, _ = mock_env_constants
    observed = {"events": [], "writer_errors": [], "closed": False}
    events = observed["events"]
    opened, release = threading.Event(), threading.Event()
    append = file_store.append_to
    original_consumer = _SavedSignalConsumer.generate
    builder = records.build_model_record_entry
    history = train_utils.append_jsonl

    def setup(case, workflow, client, *, fault=None, race=True):
        recorder = case["recorder"]
        observed.update(case=case, recorder=recorder, client=client)
        old_uri = mlflow.get_tracking_uri()
        monkeypatch.setattr(recorder, "_log_uncommitted_code", lambda: None)
        original_tags = recorder.list_tags
        original_load = recorder.load_object
        prior_get = workflow.get_recorder.side_effect
        def get_recorder(**kwargs):
            assert not (observed["closed"] and not kwargs), "implicit recorder lookup after close"
            return prior_get(**kwargs) if prior_get is not None else recorder
        workflow.get_recorder.side_effect = get_recorder

        def widened_write(path, data):
            if Path(path).name == METRIC:
                Path(path).touch()
                observed["metric_path"] = Path(path)
                events.append("metric_opened_empty")
                opened.set()
                if not release.wait(10):
                    observed["writer_errors"].append("metric release timed out")
                answer = append(path, data)
                events.append("metric_written")
                return answer
            return append(path, data)

        def analysis(record):
            original_consumer(record)
            events.append("analysis_done")
            recorder.log_metrics(**{METRIC: .00820673094451452})
            if race:
                assert opened.wait(10), "real asynchronous metric writer did not start"

        def enter():
            recorder.start_run()
            events.append("recorder_started")
            return recorder

        def close(exc_type, exc, traceback):
            events.append("context_exit")
            release.set()
            if fault in ("close", KeyboardInterrupt, SystemExit):
                if fault == "close":
                    raise RuntimeError("recorder close failed")
                raise fault("close interrupted")
            recorder.end_run(Recorder.STATUS_FA if exc_type else Recorder.STATUS_FI)
            observed["closed"] = case["completion"]["closed"] = True
            events.append("recorder_closed")
            if fault == "metric":
                observed["metric_path"].write_text("")
            elif fault == "prediction":
                recorder.save_objects(**{"pred.pkl": pd.DataFrame()})
            elif fault == "identity":
                client.set_tag(recorder.id, "model", "wrong-model")

        def tags():
            events.append("read_tags")
            if fault == "tag_interrupt" and observed["closed"]:
                raise KeyboardInterrupt("closed metadata interrupted")
            if fault == "tag_system_exit" and observed["closed"]:
                raise SystemExit("closed metadata interrupted")
            try:
                return original_tags()
            except ValueError as exc:
                observed["metadata_error"] = str(exc)
                raise

        def load(name, *args, **kwargs):
            if "analysis_done" in events:
                events.append("read_artifact:" + name)
            return original_load(name, *args, **kwargs)

        def build(**kwargs):
            entry = builder(**kwargs)
            events.append("entry_built")
            return entry

        def append_history(path, event):
            events.append("history")
            assert observed["closed"]
            return history(path, event)

        if race:
            monkeypatch.setattr(file_store, "append_to", widened_write)
        monkeypatch.setattr(_SavedSignalConsumer, "generate", analysis)
        monkeypatch.setattr(recorder, "list_tags", tags)
        monkeypatch.setattr(recorder, "load_object", load)
        monkeypatch.setattr(records, "build_model_record_entry", build)
        monkeypatch.setattr(train_utils, "append_jsonl", append_history)
        workflow.start.return_value.__enter__.side_effect = enter
        workflow.start.return_value.__exit__.side_effect = close

        def cleanup():
            release.set()
            # Only failed/interrupted test contexts need cleanup. A completed
            # recorder stays FINISHED even when its later evidence is rejected.
            if recorder.async_log is not None:
                recorder.end_run(Recorder.STATUS_FA)
            mlflow.set_tracking_uri(old_uri)
        return cleanup

    return setup, observed


@pytest.mark.parametrize("predict_only", [False, True])
def test_metadata_evidence_follows_real_async_recorder_close(finite_signal_training, completion_probe, tmp_path, predict_only):
    setup, probe = completion_probe
    history = tmp_path / "history.jsonl"
    case = finite_signal_training("DatasetH", predict_only=predict_only, history_file=str(history),
                                  record_classes=["SignalRecord", "SigAnaRecord", "_SavedSignalConsumer"],
                                  recorder_lifecycle=setup, fit_validation_steps=0 if predict_only else 2)
    assert case["result"]["success"], case["result"]
    events = probe["events"]
    assert events.index("context_exit") < events.index("metric_written") < events.index("recorder_closed")
    assert all(events.index("recorder_closed") < i for i, name in enumerate(events)
               if name == "read_tags" or name.startswith("read_artifact:"))
    assert events.index("recorder_closed") < events.index("entry_built") < events.index("history")
    assert not probe["writer_errors"]
    assert probe["client"].get_run(case["recorder"].id).info.status == "FINISHED"
    assert probe["client"].get_run(case["recorder"].id).data.metrics[METRIC] == .00820673094451452
    assert len(case["model"].predict_datasets) == (1 if predict_only else 3)
    assert case["model"].predict_segments == (["test"] if predict_only else ["valid", "valid", "test"])
    expected = case["model"].returned_predictions[-1].to_frame("score")
    pd.testing.assert_frame_equal(case["recorder"].load_object("pred.pkl"), expected)
    assert case["recorder"].load_object("label.pkl").loc[(case["dates"][-1], slice(None))].isna().all().all()
    assert json.loads(history.read_text())["record_id"] == case["recorder"].id


@pytest.mark.parametrize("predict_only", [False, True])
def test_sqlite_completion_order_is_preserved(finite_signal_training, completion_probe, tmp_path, predict_only):
    setup, probe = completion_probe
    case = finite_signal_training("DatasetH", predict_only=predict_only, history_file=str(tmp_path / "history.jsonl"),
                                  tracking_backend="sqlite", record_classes=["SigAnaRecord", "_SavedSignalConsumer"],
                                  recorder_lifecycle=lambda *a: setup(*a, race=False))
    assert case["result"]["success"], case["result"]
    assert probe["events"].index("recorder_closed") < probe["events"].index("read_tags")


@pytest.mark.parametrize("predict_only", [False, True])
@pytest.mark.parametrize("fault", ["close", "metric", "prediction", "identity"])
def test_completion_errors_never_create_success_evidence(finite_signal_training, completion_probe, tmp_path, predict_only, fault):
    setup, probe = completion_probe
    history = tmp_path / "history.jsonl"
    case = finite_signal_training("DatasetH", predict_only=predict_only, history_file=str(history),
                                  record_classes=["SigAnaRecord", "_SavedSignalConsumer"],
                                  recorder_lifecycle=lambda *a: setup(*a, fault=fault))
    assert not case["result"]["success"]
    assert "record_entry" not in case["result"] and "history_entry" not in case["result"]
    assert not history.exists() and not probe["writer_errors"]
    from types import SimpleNamespace
    from quantpits.training.runners import TrainingTargetRequest, _adapt_result
    operation = "predict_only" if predict_only else "train"
    adapted = _adapt_result(TrainingTargetRequest(
        SimpleNamespace(anchor_date=case["params"]["anchor_date"]),
        SimpleNamespace(key="finite@static", operation=operation)), case["result"])
    assert adapted.outcome == "failed" and adapted.entry is None and adapted.history_event is None
    if fault != "close":
        assert probe["closed"]
        meta = Path(case["recorder"].uri[7:]) / str(case["recorder"].experiment_id) / case["recorder"].id / "meta.yaml"
        assert yaml.safe_load(meta.read_text())["status"] == 3


@pytest.mark.parametrize("predict_only", [False, True])
@pytest.mark.parametrize("fault,control", [(KeyboardInterrupt, KeyboardInterrupt), (SystemExit, SystemExit),
                                          ("tag_interrupt", KeyboardInterrupt), ("tag_system_exit", SystemExit)])
def test_close_or_completed_read_controls_propagate(finite_signal_training, completion_probe, tmp_path, predict_only, fault, control):
    setup, probe = completion_probe
    history = tmp_path / "history.jsonl"
    with pytest.raises(control):
        finite_signal_training("DatasetH", predict_only=predict_only, history_file=str(history),
                               record_classes=["SigAnaRecord", "_SavedSignalConsumer"],
                               recorder_lifecycle=lambda *a: setup(*a, fault=fault))
    assert not history.exists() and "history" not in probe["events"]
