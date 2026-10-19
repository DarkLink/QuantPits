import copy
import os
import yaml
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

@pytest.fixture
def mock_qlib_objects():
    mock_model = MagicMock()
    mock_dataset = MagicMock()
    mock_recorder = MagicMock()
    mock_recorder.info = {'id': 'test_rid_123'}
    return mock_model, mock_dataset, mock_recorder

def test_calculate_dates_slide(mock_env_constants):
    train_utils, _ = mock_env_constants
    config_dict = {
        "market": "csi300", "benchmark": "SH000300", "train_date_mode": "last_trade_date",
        "data_slice_mode": "slide", "test_set_window": 1, "valid_set_window": 1,
        "train_set_windows": 3, "freq": "day", "current_full_cash": 200000.0
    }
    with patch('quantpits.utils.config_loader.load_workspace_config', return_value=config_dict):
        with patch('qlib.data.D') as mock_d:
            mock_d.calendar.return_value = [pd.Timestamp("2026-03-01")]
            params = train_utils.calculate_dates()
            assert params["anchor_date"] == "2026-03-01"
            assert params["test_end_time"] == "2026-03-01"

def test_calculate_dates_fixed(mock_env_constants):
    train_utils, _ = mock_env_constants
    config_dict = {
        "market": "csi300", "benchmark": "SH000300", "train_date_mode": "fixed",
        "current_date": "2026-01-01", "data_slice_mode": "fixed",
        "start_time": "2010-01-01", "fit_start_time": "2010-01-01", "fit_end_time": "2015-01-01",
        "valid_start_time": "2015-01-01", "valid_end_time": "2016-01-01",
        "test_start_time": "2016-01-01", "test_end_time": "2026-01-01"
    }
    with patch('quantpits.utils.config_loader.load_workspace_config', return_value=config_dict):
        params = train_utils.calculate_dates()
        assert params["anchor_date"] == "2026-01-01"
        assert params["fit_end_time"] == "2015-01-01"


def test_calculate_dates_slide_with_purged_cv_block(mock_env_constants):
    """Coexistence: data_slice_mode=slide with purged_cv block -> cpcv_folds in output."""
    train_utils, _ = mock_env_constants
    config_dict = {
        "market": "csi300", "benchmark": "SH000300",
        "train_date_mode": "last_trade_date",
        "data_slice_mode": "slide", "test_set_window": 1, "valid_set_window": 1,
        "train_set_windows": 3, "freq": "day", "current_full_cash": 100000.0,
        "purged_cv": {
            "n_groups": 10, "n_test_groups": 2, "n_val_groups": 1,
            "purge_steps": 5, "embargo_steps": 10,
        },
        "start_time": "2015-01-01",
    }
    with patch('quantpits.utils.config_loader.load_workspace_config',
               return_value=config_dict):
        with patch('qlib.data.D') as mock_d:
            # Generate 500 valid trading dates across multiple years
            dates = pd.date_range('2015-01-05', periods=500, freq='W-MON')
            mock_d.calendar.return_value = list(dates)
            params = train_utils.calculate_dates()
            # Slide dates are present (freq=day means anchor is last calendar date)
            assert params["anchor_date"] is not None
            assert params["fit_start_time"] is not None
            assert params["test_end_time"] is not None
            # cpcv_folds are present (coexistence)
            assert "cpcv_folds" in params
            assert len(params["cpcv_folds"]) > 0
            # data_slice_mode is NOT 'purged_cv' in params
            assert params.get("data_slice_mode") != "purged_cv"


def test_inject_config(mock_env_constants):
    train_utils, _ = mock_env_constants
    mock_yaml = {
        'market': 'old_market', 'benchmark': 'old_benchmark', 'data_handler_config': {},
        'task': {'dataset': {'kwargs': {'segments': {}}}}
    }
    params = {
        'freq': 'week', 'market': 'new_market', 'benchmark': 'new_benchmark',
        'start_time': '2000', 'end_time': '2010', 'fit_start_time': '2000', 'fit_end_time': '2005',
        'valid_start_time': '2006', 'valid_end_time': '2008', 'test_start_time': '2009', 'test_end_time': '2010'
    }
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_yaml))):
        config = train_utils.inject_config("dummy.yaml", params)
        assert config['market'] == 'new_market'
        assert config['benchmark'] == 'new_benchmark'
        assert config['data_handler_config']['label'] == ["Ref($close, -6) / Ref($close, -1) - 1"]
        assert config['task']['dataset']['kwargs']['segments']['train'] == ['2000', '2005']

def test_inject_config_extra_branches(mock_env_constants):
    train_utils, _ = mock_env_constants
    mock_yaml = {
        'strategy': {'params': {}}, 'data_handler_config': {},
        'task': {
            'dataset': {'kwargs': {'segments': {}}},
            'record': [{'class': 'PortAnaRecord', 'kwargs': {'config': {}}}, {'class': 'SigAnaRecord'}],
            'model': {'kwargs': {'base_model': 'dummy', 'model_path': 'old_path', 'd_feat': 20}}
        },
        'port_analysis_config': {}
    }
    params = {
        'freq': 'day', 'market': 'new', 'benchmark': 'new', 
        'start_time': '2000', 'end_time': '2010', 'fit_start_time': '2000', 'fit_end_time': '2005',
        'valid_start_time': '2006', 'valid_end_time': '2008', 'test_start_time': '2009', 'test_end_time': '2010',
        'label_formula': 'MyLabel', 'topk': 50, 'n_drop': 5, 'buy_suggestion_factor': 4, 'account': 100000.0
    }
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_yaml))):
        # Ensure backtest has account key
        with patch('quantpits.utils.strategy.generate_port_analysis_config', return_value={'backtest': {'account': 0}}):
            with patch('quantpits.utils.train_utils.resolve_pretrained_path') as mock_resolve:
                cfg1 = train_utils.inject_config("dummy.yaml", params, model_name="dummy", no_pretrain=True)
                assert 'model_path' not in cfg1['task']['model']['kwargs']
                assert cfg1['strategy']['params']['topk'] == 50
                
                mock_resolve.return_value = '/path.pkl'
                with patch('quantpits.utils.train_utils.validate_pretrain_compatibility'):
                    cfg2 = train_utils.inject_config("dummy.yaml", params, model_name="dummy", no_pretrain=False)
                    assert cfg2['task']['model']['kwargs']['model_path'] == '/path.pkl'

def test_train_single_model(mock_env_constants, mock_qlib_objects, tmp_path):
    train_utils, workspace = mock_env_constants
    mock_model, mock_dataset, mock_recorder = mock_qlib_objects
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml.dump({"task": {"model": {}, "dataset": {}, "record": []}}))
    params = {"freq": "day", "market": "c3", "anchor_date": "2026-03-01"}
    
    # Use placeholders to trigger lines 692-695
    task_config = {
        "task": {
            "model": {}, "dataset": {}, 
            "record": [{"class": "SigAnaRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}}]
        }
    }
    prediction = pd.DataFrame(
        {"score": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-03-01"), "S1")], names=["datetime", "instrument"]
        ),
    )
    mock_recorder.info["artifact_uri"] = (workspace / "mlruns/test/artifacts").as_uri()
    mock_recorder.list_tags.return_value = {"model": "M1", "anchor_date": "2026-03-01"}

    def load_object(name, ic_error=False):
        if name == "pred.pkl":
            return prediction
        if name == "sig_analysis/ic.pkl":
            if ic_error:
                raise ValueError("IC Error")
            return pd.Series([0.1, 0.2])
        raise FileNotFoundError(name)
    
    with patch('quantpits.utils.train_utils.inject_config', return_value=task_config):
        with patch('qlib.utils.init_instance_by_config', side_effect=[mock_model, mock_dataset, MagicMock(), mock_model, mock_dataset, MagicMock()]):
            with patch('qlib.workflow.R') as mock_R:
                mock_R.get_recorder.return_value = mock_recorder
                mock_model.predict.return_value = prediction
                mock_recorder.load_object.side_effect = load_object
                
                res = train_utils.train_single_model("M1", str(yaml_file), params, "E")
                assert res['success']
                
                # Test Failure Case: IC Error
                mock_recorder.load_object.side_effect = lambda name: load_object(name, ic_error=True)
                res_ic_err = train_utils.train_single_model("M1", str(yaml_file), params, "E")
                assert res_ic_err['success']
                assert 'IC_Mean' not in res_ic_err['performance']
                
    # Test file doesn't exist (line 649-651)
    res_no_file = train_utils.train_single_model("M1", "not_exist.yaml", params, "E")
    assert not res_no_file['success']

def test_predict_single_model(mock_env_constants, mock_qlib_objects, tmp_path):
    train_utils, _ = mock_env_constants
    mock_model, mock_dataset, mock_recorder = mock_qlib_objects
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("dummy")
    params = {"anchor_date": "2026-03-01"}
    source = {"experiment_name": "S", "models": {"M1": "I"}}
    
    task_config = {
        "task": {
            "dataset": {
                "class": "MTSDatasetH",
                "kwargs": {
                    "handler": {
                        "kwargs": {
                            "learn_processors": [
                                {"class": "DropnaLabel"},
                                {"class": "CSRankNorm"},
                            ]
                        }
                    }
                },
            },
            "record": [{"class": "SigAnaRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}}]
        }
    }
    
    with patch('qlib.utils.init_instance_by_config', side_effect=[mock_model, mock_dataset, MagicMock()]) as mock_init:
        with patch('qlib.workflow.R') as mock_R:
            mock_R.get_recorder.return_value = mock_recorder
            mock_recorder.list_artifacts.return_value = ["model.pkl"]
            mock_recorder.load_object.side_effect = [mock_model, pd.Series([0.5])]
            with patch('quantpits.utils.train_utils.inject_config', return_value=task_config):
                with patch('os.path.exists', return_value=True):
                    res = train_utils.predict_single_model("M1", {"yaml_file": str(yaml_file)}, params, "E", source)
                    assert res['success']
                    dataset_arg = mock_init.call_args_list[0].args[0]
                    assert dataset_arg["kwargs"]["handler"]["kwargs"]["learn_processors"] == [
                        {"class": "CSRankNorm"}
                    ]
                    assert task_config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["learn_processors"][0] == {
                        "class": "DropnaLabel"
                    }


def test_prepare_predict_only_mts_dataset_removes_dropna_label(mock_env_constants):
    train_utils, _ = mock_env_constants
    dataset_cfg = {
        "class": "MTSDatasetH",
        "module_path": "qlib.contrib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "kwargs": {
                    "infer_processors": [{"class": "Fillna"}],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                },
            }
        },
    }

    prepared = train_utils._prepare_predict_only_dataset_config(dataset_cfg)

    assert prepared is not dataset_cfg
    assert prepared["kwargs"]["handler"]["kwargs"]["learn_processors"] == [
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    ]
    assert prepared["kwargs"]["handler"]["kwargs"]["infer_processors"] == [
        {"class": "Fillna"}
    ]
    assert dataset_cfg["kwargs"]["handler"]["kwargs"]["learn_processors"][0] == {
        "class": "DropnaLabel"
    }


def test_prepare_predict_only_dataset_leaves_regular_dataset_unchanged(mock_env_constants):
    train_utils, _ = mock_env_constants
    dataset_cfg = {
        "class": "DatasetH",
        "kwargs": {
            "handler": {
                "kwargs": {"learn_processors": ["DropnaLabel"]}
            }
        },
    }

    prepared = train_utils._prepare_predict_only_dataset_config(dataset_cfg)

    assert prepared is dataset_cfg
    assert prepared["kwargs"]["handler"]["kwargs"]["learn_processors"] == [
        "DropnaLabel"
    ]


class _FiniteSignalModel:
    """Observe real dataset rows without fitting a costly statistical model."""

    def __init__(self):
        self.fit_dataset = None
        self.predict_datasets = []
        self.fit_labels = None
        self.seed = 7
        self.truncate_prediction_tail = False

    @staticmethod
    def _segment_rows(dataset, segment):
        from qlib.data.dataset.handler import DataHandlerLP

        prepared = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if isinstance(prepared, pd.DataFrame):
            return prepared.index, prepared["feature"].values, prepared["label"].values
        rows = np.array([slc.stop - 1 for slc in prepared._batch_slices])
        index = prepared.restore_index(rows).swaplevel().sort_values()
        values = pd.DataFrame(
            {"feature": prepared._data[rows, 0], "label": prepared._label[rows]},
            index=prepared.restore_index(rows).swaplevel(),
        ).sort_index()
        return index, values[["feature"]].values, values[["label"]].values

    def fit(self, dataset, evals_result=None):
        self.fit_dataset = dataset
        self.fit_labels = self._segment_rows(dataset, "train")[2]

    def predict(self, dataset):
        self.predict_datasets.append(dataset)
        if hasattr(dataset, "_batch_slices"):
            index, values, _ = self._segment_rows(dataset, "test")
            prediction = pd.Series(values[:, 0], index=index)
        else:
            from qlib.data.dataset.handler import DataHandlerLP

            features = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
            prediction = pd.Series(features.iloc[:, 0], index=features.index)
        if self.truncate_prediction_tail:
            dates = prediction.index.get_level_values("datetime")
            prediction = prediction.loc[dates < dates.max()]
        return prediction


@pytest.fixture
def finite_signal_training(mock_env_constants, monkeypatch):
    """Real tiny handlers, cache, SignalRecord, MLflow persistence and coverage."""
    from mlflow.tracking import MlflowClient
    from qlib.utils import init_instance_by_config as real_init
    from qlib.workflow.recorder import MLflowRecorder
    from quantpits.utils.handler_cache import HandlerCacheManager

    train_utils, workspace = mock_env_constants
    dates = pd.bdate_range(end="2026-10-09", periods=18)
    index = pd.MultiIndex.from_product(
        [dates, ["S1", "S2", "S3"]], names=["datetime", "instrument"]
    )
    frame = pd.DataFrame(
        {
            ("feature", "VALUE"): np.arange(len(index), dtype=float),
            ("feature", "OTHER"): np.tile([1.0, 3.0, 9.0], len(dates)),
            ("label", "LABEL"): np.tile([0.1, 0.4, -0.2], len(dates)),
        },
        index=index,
    )
    # The final six feature days have no observable forward label. One older
    # training sample is also unlabeled, so training-time DropnaLabel matters.
    frame.loc[(slice(dates[-6], dates[-1]), slice(None)), ("label", "LABEL")] = np.nan
    frame.loc[(dates[2], "S1"), ("label", "LABEL")] = np.nan
    frame.loc[(dates[-1], "S1"), ("feature", "VALUE")] = np.nan
    data_file = workspace / "data/finite_signal.pkl"
    frame.to_pickle(data_file)
    yaml_file = workspace / "config/finite_signal.yaml"
    yaml_file.write_text("fixture", encoding="utf-8")

    def run(dataset_class="MTSDatasetH", use_cache=False, dropna_label=True,
            rebuild_error=None, truncate_prediction=False, recorder_exit_error=None):
        if dataset_class != "DatasetH":
            # Qlib's real MTS datasets import torch; lightweight [test] CI
            # deliberately omits it. Keep DatasetH coverage active there.
            pytest.importorskip("torch", reason="MTS integration requires optional torch")
        learn_processors = [{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}]
        if dropna_label:
            learn_processors.insert(0, {"class": "DropnaLabel"})
        dataset_cfg = {
            "class": dataset_class,
            "module_path": {
                "MTSDatasetH": "qlib.contrib.data.dataset",
                "PurgedMTSDatasetH": "quantpits.data.cpcv_dataset",
                "DatasetH": "qlib.data.dataset",
            }[dataset_class],
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset.handler",
                    "kwargs": {
                        "data_loader": {
                            "class": "StaticDataLoader",
                            "module_path": "qlib.data.dataset.loader",
                            "kwargs": {"config": str(data_file)},
                        },
                        "infer_processors": [
                            {
                                "class": "RobustZScoreNorm",
                                "kwargs": {
                                    "fields_group": "feature",
                                    "fit_start_time": str(dates[0].date()),
                                    "fit_end_time": str(dates[7].date()),
                                    "clip_outlier": True,
                                },
                            },
                            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                        ],
                        "learn_processors": learn_processors,
                    },
                },
                "segments": {
                    "train": [str(dates[0].date()), str(dates[7].date())],
                    "valid": [str(dates[8].date()), str(dates[10].date())],
                    "test": [str(dates[11].date()), str(dates[-1].date())],
                },
            },
        }
        if dataset_class != "DatasetH":
            dataset_cfg["kwargs"].update(horizon=6, seq_len=3, batch_size=3, shuffle=False)
        task_config = {
            "task": {
                "model": {"class": "FiniteSignalModel", "kwargs": {"seed": 7, "n_epochs": 1}},
                "dataset": dataset_cfg,
                "record": [{
                    "class": "SignalRecord",
                    "module_path": "qlib.workflow.record_temp",
                    "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
                }],
            },
        }
        original_dataset_cfg = copy.deepcopy(dataset_cfg)
        original_model_cfg = copy.deepcopy(task_config["task"]["model"])
        model = _FiniteSignalModel()
        model.truncate_prediction_tail = truncate_prediction
        tracking_uri = (workspace / "mlruns").as_uri()
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_id = client.create_experiment(
            "finite_signal", artifact_location=(workspace / "output/finite_signal").as_uri()
        )
        mlflow_run = client.create_run(experiment_id, tags={"mlflow.runName": "finite_signal"})
        recorder = MLflowRecorder(experiment_id, tracking_uri, mlflow_run=mlflow_run)
        constructed = []

        def init(config, *args, **kwargs):
            if isinstance(config, dict) and config.get("class") == "FiniteSignalModel":
                return model
            if isinstance(config, dict) and config.get("class") == dataset_class:
                if constructed and rebuild_error is not None:
                    raise rebuild_error
                dataset = real_init(config, *args, **kwargs)
                constructed.append(dataset)
                return dataset
            return real_init(config, *args, **kwargs)

        monkeypatch.setattr("qlib.utils.init_instance_by_config", init)
        monkeypatch.setattr(train_utils, "inject_config", lambda *args, **kwargs: task_config)
        workflow = MagicMock()
        workflow.get_recorder.return_value = recorder
        workflow.set_tags.side_effect = recorder.set_tags
        workflow.log_params.side_effect = recorder.log_params
        if recorder_exit_error is not None:
            workflow.start.return_value.__exit__.side_effect = recorder_exit_error
        monkeypatch.setattr("qlib.workflow.R", workflow)
        cache = HandlerCacheManager(max_size_mb=16) if use_cache else None
        params = {
            "freq": "week", "market": "finite", "anchor_date": str(dates[-1].date()),
            "fit_start_time": str(dates[0].date()), "fit_end_time": str(dates[7].date()),
            "test_end_time": str(dates[-1].date()), "topk": 22, "n_drop": 3,
        }
        case = {
            "model": model, "recorder": recorder, "constructed": constructed,
            "cache": cache, "task_config": task_config, "original_dataset_cfg": original_dataset_cfg,
            "original_model_cfg": original_model_cfg, "params": params, "dates": dates,
            "raw_frame": frame, "workspace": workspace,
        }
        try:
            case["result"] = train_utils.train_single_model(
                "finite", str(yaml_file), params, "finite_signal", cache_mgr=cache,
                history_file=False, workspace_root=str(workspace),
            )
        finally:
            client.set_terminated(recorder.id)
        return case

    return run


@pytest.mark.parametrize("dataset_class", ["MTSDatasetH", "PurgedMTSDatasetH"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_train_mts_preserves_unlabeled_anchor_after_fit(finite_signal_training, dataset_class, use_cache):
    case = finite_signal_training(dataset_class, use_cache)
    model, recorder, dates = case["model"], case["recorder"], case["dates"]
    prediction = recorder.load_object("pred.pkl")
    # Assert real persisted coverage first: this fails on the original reuse
    # of the DropnaLabel training dataset, before any ready-record assertion.
    assert prediction.index.get_level_values("datetime").max() == dates[-1]
    assert len(prediction) == 21
    assert np.isfinite(prediction["score"]).all()
    assert case["result"]["success"], case["result"]
    entry = case["result"]["record_entry"]
    assert entry["operation"] == "train"
    assert entry["prediction_end"] == entry["requested_anchor"] == "2026-10-09"
    assert entry["prediction_rows"] == len(prediction)
    assert model.fit_labels.shape == (23, 1)
    assert np.isfinite(model.fit_labels).all()
    assert len(case["constructed"]) == 2
    train_dataset, prediction_dataset = case["constructed"]
    assert model.fit_dataset is train_dataset
    assert model.predict_datasets == [prediction_dataset, prediction_dataset]
    assert model.seed == 7 and model.topk == 22 and model.n_drop == 3
    assert case["task_config"]["task"]["model"] == case["original_model_cfg"]
    assert case["task_config"]["task"]["dataset"] == case["original_dataset_cfg"]
    assert [type(p).__name__ for p in train_dataset.handler.learn_processors] == ["DropnaLabel", "CSRankNorm"]
    assert [type(p).__name__ for p in prediction_dataset.handler.learn_processors] == ["CSRankNorm"]
    assert [type(p).__name__ for p in prediction_dataset.handler.infer_processors] == ["RobustZScoreNorm", "Fillna"]
    pd.testing.assert_frame_equal(train_dataset.handler._infer, prediction_dataset.handler._infer)
    pd.testing.assert_frame_equal(
        train_dataset.handler._learn,
        prediction_dataset.handler._learn.loc[train_dataset.handler._learn.index],
    )
    tail = prediction_dataset._index.get_level_values("datetime") >= dates[-6]
    assert np.isnan(prediction_dataset._label[tail]).all()
    raw_labels = recorder.load_object("label.pkl")
    assert raw_labels.loc[(slice(dates[-6], dates[-1]), slice(None))].isna().all().all()
    assert raw_labels.index.get_level_values("datetime").max() == dates[-1]
    if use_cache:
        assert case["cache"].stats["misses"] == 2


@pytest.mark.parametrize("dataset_class", ["DatasetH", "MTSDatasetH", "PurgedMTSDatasetH"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_train_reuses_dataset_when_prediction_config_unchanged(finite_signal_training, dataset_class, use_cache):
    case = finite_signal_training(dataset_class, use_cache, dropna_label=dataset_class == "DatasetH")
    assert case["result"]["success"], case["result"]
    assert len(case["constructed"]) == 1
    model = case["model"]
    assert model.predict_datasets == [model.fit_dataset, model.fit_dataset]
    assert case["result"]["record_entry"]["prediction_end"] == "2026-10-09"
    if use_cache:
        assert case["cache"].stats["misses"] == 1


@pytest.mark.parametrize("use_cache", [False, True])
def test_train_prediction_dataset_rebuild_failure_is_not_success(finite_signal_training, use_cache):
    case = finite_signal_training(use_cache=use_cache, rebuild_error=RuntimeError("prediction rebuild failed"))
    assert case["model"].fit_dataset is not None
    assert not case["result"]["success"]
    assert case["result"]["record_id"] is None
    assert "record_entry" not in case["result"]
    assert case["result"]["error"] == "prediction rebuild failed"
    assert not case["model"].predict_datasets
    assert case["recorder"].list_artifacts() == []


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("phase,dataset_class", [
    ("rebuild", "MTSDatasetH"),
    ("recorder_exit", "MTSDatasetH"),
    ("recorder_exit", "DatasetH"),
])
def test_train_signal_interrupt_propagates(finite_signal_training, interrupt, use_cache, phase, dataset_class):
    error_arg = "rebuild_error" if phase == "rebuild" else "recorder_exit_error"
    with pytest.raises(interrupt):
        finite_signal_training(dataset_class, use_cache=use_cache, **{error_arg: interrupt("stop after fit")})


@pytest.mark.parametrize("dataset_class", ["DatasetH", "MTSDatasetH"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_train_bad_persisted_anchor_does_not_report_success(finite_signal_training, use_cache, dataset_class):
    case = finite_signal_training(dataset_class, use_cache=use_cache, truncate_prediction=True)
    prediction = case["recorder"].load_object("pred.pkl")
    assert prediction.index.get_level_values("datetime").max() == case["dates"][-2]
    assert not case["result"]["success"]
    assert "record_entry" not in case["result"]
    assert case["result"]["record_id"] == case["recorder"].info["id"]
    assert "persisted prediction end differs from requested anchor" in case["result"]["error"]


@pytest.mark.parametrize("dataset_class", ["DatasetH", "MTSDatasetH"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_train_recorder_exit_failure_does_not_report_success(finite_signal_training, use_cache, dataset_class):
    case = finite_signal_training(dataset_class, use_cache=use_cache,
                                  recorder_exit_error=RuntimeError("recorder exit failed"))
    prediction = case["recorder"].load_object("pred.pkl")
    assert prediction.index.get_level_values("datetime").max() == case["dates"][-1]
    assert not case["result"]["success"]
    assert "record_entry" not in case["result"]
    assert case["result"]["record_id"] == case["recorder"].info["id"]
    assert case["result"]["error"] == "recorder exit failed"

def test_show_model_list(mock_env_constants, capsys):
    train_utils, _ = mock_env_constants
    class Args:
        algorithm = None
        dataset = None
        market = None
        tag = None
    args = Args()
    with patch('quantpits.utils.train_utils.load_model_registry', return_value={"m1": {"enabled": True}}):
        train_utils.show_model_list(args)
    captured = capsys.readouterr()
    assert "m1" in captured.out

class TestPretrainManagement:
    def test_get_pretrained_model_path(self, mock_env_constants):
        train_utils, _ = mock_env_constants
        with patch('os.path.exists', return_value=True):
            assert "lstm_latest.pkl" in train_utils.get_pretrained_model_path("lstm")
            assert "lstm_2026.pkl" in train_utils.get_pretrained_model_path("lstm", "2026")

    @patch('quantpits.utils.train_utils._get_inner_model')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_pretrained_model(self, mock_file, mock_copy, mock_makedirs, mock_get_inner, mock_env_constants):
        train_utils, _ = mock_env_constants
        mock_inner = MagicMock(); mock_get_inner.return_value = mock_inner
        mock_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': mock_torch}):
            path = train_utils.save_pretrained_model(MagicMock(), "lstm", "2026", d_feat=2, hidden_size=6, num_layers=2)
            assert "lstm_2026.pkl" in path
            assert mock_torch.save.called
            assert mock_copy.call_count == 2

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"d_feat": 20}')
    def test_load_pretrained_metadata(self, mock_file, mock_exists, mock_env_constants):
        train_utils, _ = mock_env_constants
        meta = train_utils.load_pretrained_metadata("lstm")
        assert meta["d_feat"] == 20
        assert train_utils.load_pretrained_metadata("lstm", "2026")["d_feat"] == 20

    @patch('quantpits.utils.train_utils.load_model_registry', return_value={"gats": {"pretrain_source": "lstm"}})
    @patch('quantpits.utils.train_utils.get_pretrained_model_path', return_value="/p.pkl")
    def test_resolve_pretrained_path(self, mock_get, mock_load, mock_env_constants, capsys):
        train_utils, _ = mock_env_constants
        assert train_utils.resolve_pretrained_path("gats") == "/p.pkl"
        
        # Branch: not found
        mock_get.return_value = None
        assert train_utils.resolve_pretrained_path("gats") is None
        captured = capsys.readouterr()
        assert "未找到" in captured.out

    def test_validate_pretrain_compatibility(self, mock_env_constants):
        train_utils, _ = mock_env_constants
        with patch('quantpits.utils.train_utils.load_pretrained_metadata', return_value={"d_feat": 20}):
            # Match
            train_utils.validate_pretrain_compatibility("m", "/lstm_latest.pkl", 20)
            # Mismatch
            with pytest.raises(ValueError, match="Feature 不匹配"):
                train_utils.validate_pretrain_compatibility("m", "/lstm_latest.pkl", 10)
        # No metadata
        with patch('quantpits.utils.train_utils.load_pretrained_metadata', return_value=None):
            train_utils.validate_pretrain_compatibility("m", "/lstm_latest.pkl", 20)

    def test_get_inner_model(self, mock_env_constants):
        train_utils, _ = mock_env_constants
        class M: pass
        obj = M(); obj.LSTM_model = "inner"
        assert train_utils._get_inner_model(obj) == "inner"
        with pytest.raises(ValueError): train_utils._get_inner_model(M())
