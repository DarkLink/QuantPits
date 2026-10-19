"""Real small model artifacts; no production backend or provider is used."""

import copy
import inspect
from functools import wraps
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import pytest


def _stored_artifact(case, recorder, name):
    """Mutate the temporary store, not a possibly detached download copy."""
    uri = urlparse(recorder.get_artifact_uri())
    assert uri.scheme == "file" and not uri.netloc
    path = (Path(unquote(uri.path)) / name).resolve()
    assert case["workspace"].resolve() in path.parents
    assert path.is_file()
    return path


@pytest.fixture
def reload_case(mock_env_constants, monkeypatch):
    from mlflow.tracking import MlflowClient
    from qlib.contrib.model.linear import LinearModel
    from qlib.utils import init_instance_by_config
    from qlib.workflow.recorder import MLflowRecorder

    train, workspace = mock_env_constants
    dates = pd.to_datetime(["2026-09-25", "2026-09-28", "2026-10-05", "2026-10-09"])
    index = pd.MultiIndex.from_product([dates, ["S1", "S2"]], names=["datetime", "instrument"])
    feature = np.arange(len(index), dtype=float)
    frame = pd.DataFrame({("feature", "x"): feature, ("feature", "y"): feature ** 2,
                          ("label", "LABEL"): feature * 0.1 + 0.2}, index=index)
    dataset_cfg = {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {"class": "DataHandlerLP", "module_path": "qlib.data.dataset.handler",
                        "kwargs": {"data_loader": {"class": "StaticDataLoader",
                                  "module_path": "qlib.data.dataset.loader", "kwargs": {"config": frame}}}},
            "segments": {"train": (str(dates[0].date()), str(dates[1].date())),
                         "test": (str(dates[2].date()), str(dates[3].date()))},
        },
    }
    dataset = init_instance_by_config(dataset_cfg)
    model = LinearModel(estimator="ridge", alpha=1e-6)
    model.fit(dataset)
    expected = model.predict(dataset)
    # No subsequent path may fit, including a deserialized copy of the model.
    def forbidden_fit(*args, **kwargs):
        pytest.fail("reload/predict must not fit")
    monkeypatch.setattr(LinearModel, "fit", forbidden_fit)
    client = MlflowClient(tracking_uri=(workspace / "mlruns").as_uri())
    calls = {}
    supports_trusted = "trusted" in inspect.signature(MLflowRecorder.load_object).parameters

    def recorder(experiment):
        exp = client.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(
            experiment, artifact_location=(workspace / "output" / experiment).as_uri(),
        )
        result = MLflowRecorder(exp_id, client.tracking_uri, mlflow_run=client.create_run(exp_id))
        calls[result.id] = []
        original_load = result.load_object
        @wraps(original_load)
        def observed_load(name, *args, **kwargs):
            calls[result.id].append((name, kwargs))
            return original_load(name, *args, **kwargs)
        result.load_object = observed_load
        return result

    source = recorder("reload_source")
    source.set_tags(model="M1", mode="train")
    source.save_objects(**{"model.pkl": model, "model_fold_0.pkl": model, "model_fold_1.pkl": model})
    output = recorder("reload_output")
    runtime = MagicMock()
    selected = {("reload_source", source.id): source}
    runtime.get_recorder.side_effect = lambda **kw: selected[(kw["experiment_name"], kw["recorder_id"])] if kw else output
    runtime.set_tags.side_effect = output.set_tags
    runtime.log_params.side_effect = output.log_params
    monkeypatch.setattr("qlib.workflow.R", runtime)
    yaml_file = workspace / "config/model.yaml"
    yaml_file.write_text("fixture", encoding="utf-8")
    params = {"anchor_date": "2026-10-09", "test_start_time": "2026-10-05",
              "test_end_time": "2026-10-09", "freq": "week", "cpcv_folds": [{}, {}]}

    def config(_yaml, current_params, *args, **kwargs):
        cfg = copy.deepcopy(dataset_cfg)
        cfg["kwargs"]["segments"]["test"] = (current_params["test_start_time"], current_params["test_end_time"])
        return {"task": {"dataset": cfg, "record": [{
            "class": "SignalRecord", "module_path": "qlib.workflow.record_temp",
            "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
        }]}}
    monkeypatch.setattr(train, "inject_config", config)
    monkeypatch.setattr(train, "inject_config_for_fold", config)
    info = {"yaml_file": str(yaml_file), "record_id": source.id}

    def predict(kind):
        if kind == "static":
            return train.predict_single_model("M1", info, params, "reload_output", {
                "experiment_name": "reload_source", "models": {"M1": info["record_id"]},
            }, history_file=False, workspace_root=workspace)
        return train.predict_cpcv_model("M1", info, params, "reload_output",
            source_experiment_name="reload_source", history_file=False, workspace_root=workspace)

    return {"source": source, "output": output, "recorder": recorder, "selected": selected,
            "calls": calls, "runtime": runtime, "params": params, "info": info, "dates": dates,
            "expected": expected, "predict": predict, "workspace": workspace,
            "model_kwargs": {"trusted": True} if supports_trusted else {}}


@pytest.mark.parametrize("kind", ["static", "cpcv"])
@pytest.mark.parametrize("missing_child", [False, True])
def test_actual_predict_only_reloads_models_and_preserves_requested_coverage(reload_case, kind, missing_child):
    case = reload_case
    source = case["source"]
    if missing_child:
        child = case["recorder"]("reload_source")
        child.save_objects(**{"pred.pkl": case["expected"]})
        child.set_tags(model="M1", mode="predict_only", source_record_id=source.id, source_experiment="reload_source")
        case["selected"][("reload_source", child.id)] = child
        case["info"]["record_id"] = child.id
    model_paths = ["model.pkl"] if kind == "static" else ["model_fold_0.pkl", "model_fold_1.pkl"]
    before = {name: Path(source.download_artifact(name)).read_bytes() for name in model_paths}
    result = case["predict"](kind)
    assert result["success"], result
    entry = result["record_entry"]
    assert entry["prediction_end"] == case["params"]["anchor_date"]
    assert entry["prediction_rows"] == len(case["expected"])
    assert entry["source_recorder_id"] == case["info"]["record_id"]
    assert case["calls"][source.id] == [(name, case["model_kwargs"]) for name in model_paths]
    assert all(kwargs == {} for name, kwargs in case["calls"][case["output"].id])
    prediction = case["output"].load_object("pred.pkl")
    scores = prediction["score"] if isinstance(prediction, pd.DataFrame) else prediction
    pd.testing.assert_series_equal(scores, case["expected"], check_names=False)
    assert all(Path(source.download_artifact(name)).read_bytes() == before[name] for name in model_paths)
    if missing_child:
        assert case["calls"][child.id] == []


@pytest.mark.parametrize("damage", ["corrupt", "missing_module"])
def test_existing_broken_model_keeps_real_qlib_error_and_never_uses_parent(reload_case, damage):
    case = reload_case
    child = case["recorder"]("reload_source")
    child.save_objects(**{"model.pkl": object()})
    child.set_tags(source_record_id=case["source"].id, source_experiment="reload_source")
    path = _stored_artifact(case, child, "model.pkl")
    path.write_bytes(b"!broken pickle" if damage == "corrupt" else b"cquantpits_missing_model_module\nMissingModel\n.")
    case["selected"][("reload_source", child.id)] = child
    case["info"]["record_id"] = child.id
    result = case["predict"]("static")
    assert not result["success"]
    assert "record_entry" not in result
    assert "LoadObjectError" in result["error"]
    assert child.id in result["error"] and "reload_source" in result["error"]
    assert ("invalid load key" if damage == "corrupt" else "quantpits_missing_model_module") in result["error"]
    assert "not found" not in result["error"]
    assert case["calls"][case["source"].id] == []
    assert not case["runtime"].start.called
    assert case["output"].list_artifacts() == []


@pytest.mark.parametrize("phase", ["list", "load"])
@pytest.mark.parametrize("error", [OSError("backend unavailable"), TypeError("invalid model state"),
                                  KeyboardInterrupt("stop"), SystemExit("stop")])
def test_uncertain_source_or_interrupt_never_falls_back(reload_case, monkeypatch, phase, error):
    case = reload_case
    source = case["source"]
    source.set_tags(source_record_id="other", source_experiment="reload_source")
    failed = MagicMock(side_effect=error)
    if phase == "list":
        monkeypatch.setattr(source, "list_artifacts", failed)
    else:
        def load(name, *, trusted=False):
            return failed(name, trusted=trusted)
        monkeypatch.setattr(source, "load_object", load)
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        with pytest.raises(type(error)) as raised:
            case["predict"]("static")
        assert raised.value is error
    else:
        result = case["predict"]("static")
        assert not result["success"]
        assert str(error) in result["error"] and source.id in result["error"]
        assert "record_entry" not in result
    assert failed.call_count == 1
    assert case["runtime"].get_recorder.call_count == 1
    assert not case["runtime"].start.called


@pytest.mark.parametrize("method", ["slide", "cpcv"])
@pytest.mark.parametrize("operation", ["predict", "repair"])
def test_actual_rolling_reload_uses_model_trust_and_default_prediction_reads(reload_case, method, operation):
    case = reload_case
    prediction, repaired = _rolling_reload(case, method, operation)
    if operation == "repair":
        assert repaired
    expected = case["expected"] if operation == "repair" else case["expected"].loc["2026-10-09":]
    assert prediction is not None
    pd.testing.assert_series_equal(prediction["score"], expected, check_names=False)
    model_paths = ["model.pkl"] if method == "slide" else ["model_fold_0.pkl", "model_fold_1.pkl"]
    observed = case["calls"][case["source"].id]
    assert [(name, kwargs) for name, kwargs in observed if name in model_paths] == [
        (name, case["model_kwargs"]) for name in model_paths]
    assert all(kwargs == {} for name, kwargs in observed if name == "pred.pkl")


def _rolling_reload(case, method, operation, legacy_folds=False):
    from quantpits.scripts.rolling import strategy_cpcv, strategy_slide
    strategy = strategy_slide if method == "slide" else strategy_cpcv
    window = {"window_idx": 0, "train_start": "2026-09-25", "train_end": "2026-09-28",
              "valid_start": "2026-09-28", "valid_end": "2026-09-28", "test_start": "2026-10-05",
              "test_end": "2026-10-09" if operation == "repair" else "2026-10-05", "cpcv_folds": [{}, {}]}
    if legacy_folds:
        window.pop("cpcv_folds")
    comp = {"window_idx": 0, "record_id": case["source"].id}
    if operation == "repair":
        case["source"].save_objects(**{"pred.pkl": case["expected"].loc[:"2026-10-05"].to_frame("score")})
        return strategy.repair_truncated("M1", case["info"], comp, {0: window},
                                         "reload_source", case["params"])
    state = MagicMock()
    state.get_completed_record_ids.return_value = [comp]
    return strategy.predict_latest("M1", case["info"], state, "reload_source", case["params"],
                                    "2026-10-09", [window]), False


@pytest.mark.parametrize("operation", ["cpcv_predict", "rolling_predict", "rolling_repair"])
@pytest.mark.parametrize("failure", ["corrupt", "keyboard", "exit"])
def test_second_fold_failure_never_becomes_complete_output(reload_case, monkeypatch, operation, failure):
    from qlib.utils.exceptions import LoadObjectError
    case = reload_case
    source = case["source"]
    if failure == "corrupt":
        _stored_artifact(case, source, "model_fold_1.pkl").write_bytes(b"!broken fold")
    else:
        original_load = source.load_object
        error = KeyboardInterrupt("stop") if failure == "keyboard" else SystemExit("stop")
        @wraps(original_load)
        def interrupted_load(name, *args, **kwargs):
            if name == "model_fold_1.pkl":
                raise error
            return original_load(name, *args, **kwargs)
        monkeypatch.setattr(source, "load_object", interrupted_load)

    def invoke():
        if operation == "cpcv_predict":
            return case["predict"]("cpcv")
        return _rolling_reload(case, "cpcv", "repair" if operation == "rolling_repair" else "predict")

    if failure != "corrupt" or operation == "cpcv_predict":
        expected_error = LoadObjectError if failure == "corrupt" else type(error)
        with pytest.raises(expected_error):
            invoke()
    else:
        prediction, repaired = invoke()
        assert not repaired
        if operation == "rolling_predict":
            assert prediction is None
        else:
            assert prediction.index.get_level_values("datetime").max() == case["dates"][2]
    assert case["output"].list_artifacts() == []
    assert not case["runtime"].start.called
    if operation == "rolling_repair":
        persisted = source.load_object("pred.pkl")
        assert persisted.index.get_level_values("datetime").max() == case["dates"][2]


@pytest.mark.parametrize("corrupt", [False, True])
def test_legacy_rolling_fold_discovery_does_not_use_load_failure_as_terminator(reload_case, corrupt):
    case = reload_case
    if corrupt:
        _stored_artifact(case, case["source"], "model_fold_1.pkl").write_bytes(b"!broken fold")
    prediction, _ = _rolling_reload(case, "cpcv", "predict", legacy_folds=True)
    if corrupt:
        assert prediction is None
    else:
        pd.testing.assert_series_equal(prediction["score"], case["expected"].loc["2026-10-09":], check_names=False)
    assert case["calls"][case["source"].id] == [
        ("model_fold_0.pkl", case["model_kwargs"]), ("model_fold_1.pkl", case["model_kwargs"])]


@pytest.mark.parametrize("operation", ["predict", "repair"])
def test_missing_required_rolling_fold_does_not_reduce_requested_set(reload_case, operation):
    case = reload_case
    # Delete only a temporary artifact, retaining the window's two required folds.
    _stored_artifact(case, case["source"], "model_fold_1.pkl").unlink()
    prediction, repaired = _rolling_reload(case, "cpcv", operation)
    assert not repaired
    if operation == "predict":
        assert prediction is None
    else:
        assert prediction.index.get_level_values("datetime").max() == case["dates"][2]
        persisted = case["source"].load_object("pred.pkl")
        assert persisted.index.get_level_values("datetime").max() == case["dates"][2]


def test_wrapped_internal_file_not_found_with_present_model_is_not_parent_fallback(reload_case, monkeypatch):
    from qlib.utils.exceptions import LoadObjectError
    case = reload_case
    source = case["source"]
    source.set_tags(source_record_id="other", source_experiment="reload_source")
    calls = []
    def load(name, *, trusted=False):
        calls.append((name, trusted))
        try:
            raise FileNotFoundError("model dependency weights.cache is missing")
        except FileNotFoundError as exc:
            raise LoadObjectError(str(exc)) from exc
    monkeypatch.setattr(source, "load_object", load)
    result = case["predict"]("static")
    assert not result["success"]
    assert "LoadObjectError" in result["error"] and "weights.cache" in result["error"]
    assert source.id in result["error"]
    assert calls == [("model.pkl", True)]
    assert case["runtime"].get_recorder.call_count == 1
    assert not case["runtime"].start.called
