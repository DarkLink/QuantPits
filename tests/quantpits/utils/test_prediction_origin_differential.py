"""Execute both reviewed static/CPCV functions with the same controlled inputs."""
import ast
import copy
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tests.quantpits.research.test_decision_surface_maintenance import reviewed_train_sources


@pytest.mark.parametrize('kind', ['static', 'cpcv'])
@pytest.mark.parametrize('ancestry', ['verified', 'wrong_model', 'missing_parent', 'error', 'interrupt'])
def test_reviewed_prediction_outputs_and_failure_boundary(mock_env_constants, tmp_path, monkeypatch, kind, ancestry):
    train, _ = mock_env_constants
    from quantpits.training import model_identity, records
    import qlib.utils
    import qlib.workflow
    name = 'predict_single_model' if kind == 'static' else 'predict_cpcv_model'
    functions = []
    for source in reviewed_train_sources():
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        scope = dict(vars(train))
        text = '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])
        exec(compile(text, '<reviewed-prediction>', 'exec'), scope)
        functions.append((scope[name], scope))
    config_file = tmp_path / 'model.yaml'
    config_file.write_text('fixture')
    cfg = {'task': {'dataset': {'class': 'DatasetH', 'kwargs': {}}, 'record': []}}
    params = {'anchor_date': '2026-09-11', 'cpcv_folds': [{}]}
    outputs = []
    # Record writer is outside the changed functions; compare its complete
    # invocation and preserve the selected model/dataset/prediction operations.
    entry = MagicMock()
    entry.to_dict.return_value = {'record': 'same'}
    builder = MagicMock(return_value=entry)
    monkeypatch.setattr(records, 'build_model_record_entry', builder)
    class FixedDate(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 12, tzinfo=tz)
    for position, (function, scope) in enumerate(functions):
        model = MagicMock()
        model.predict.return_value = pd.Series([3., 1., 2.], index=['C', 'A', 'B'])
        model.fit.side_effect = AssertionError('prediction must not train')
        dataset = MagicMock()
        source = MagicMock()
        source.list_artifacts.return_value = ['model_fold_0.pkl']
        source.load_object.return_value = model
        def tags():
            if ancestry == 'interrupt':
                raise KeyboardInterrupt()
            if ancestry == 'error':
                raise OSError('private ancestry detail')
            if ancestry == 'wrong_model':
                return {'model': 'OTHER'}
            if ancestry == 'missing_parent':
                return {'model': 'M1', 'source_record_id': 'missing'}
            return {'model': 'M1', 'mode': 'train'}
        source.list_tags.side_effect = tags
        output = MagicMock()
        output.id = 'prediction'
        output.info = {'id': 'prediction'}
        output.load_object.return_value = pd.Series([0.2, 0.4, 0.6])
        runtime = MagicMock()
        runtime.get_recorder.side_effect = lambda **kw: source if kw else output
        monkeypatch.setattr(qlib.workflow, 'R', runtime)
        initialize = MagicMock(return_value=dataset)
        monkeypatch.setattr(qlib.utils, 'init_instance_by_config', initialize)
        scope['inject_config'] = lambda *a, **kw: copy.deepcopy(cfg)
        scope['inject_config_for_fold'] = lambda *a, **kw: copy.deepcopy(cfg)
        scope['datetime'] = FixedDate
        scope['_flatten_rnn_params'] = lambda model: None
        builder.reset_mock()
        def invoke():
            info = {'yaml_file': str(config_file), 'record_id': 'training'}
            if kind == 'static':
                return function('M1', info, params, 'PRED',
                    {'experiment_name': 'TRAIN', 'models': {'M1': 'training'}}, workspace_root=str(tmp_path))
            return function('M1', info, params, 'PRED', source_experiment_name='TRAIN', workspace_root=str(tmp_path))
        if ancestry == 'interrupt' and position == 1:
            with pytest.raises(KeyboardInterrupt):
                invoke()
            assert not runtime.start.called
            assert not model.predict.called
            continue
        result = invoke()
        assert result['success'], result
        assert not model.fit.called
        model.predict.assert_called_once_with(dataset=dataset)
        tag_values = runtime.set_tags.call_args.kwargs
        if position == 1:
            status = 'VERIFIED' if ancestry == 'verified' else 'UNRESOLVED'
            assert tag_values.pop('training_origin_status') == status
            if status == 'VERIFIED':
                assert tag_values.pop('training_origin_experiment') == 'TRAIN'
                assert tag_values.pop('training_origin_record_id') == 'training'
            else:
                assert 'training_origin_record_id' not in tag_values
        saved = [call.kwargs for call in output.save_objects.call_args_list]
        if kind == 'cpcv':
            pd.testing.assert_series_equal(saved[0]['pred.pkl'], model.predict.return_value, check_names=False)
        record_args = {k: v for k, v in builder.call_args.kwargs.items() if k != 'recorder'}
        assert record_args['source_recorder_id'] == 'training'
        assert record_args['source_experiment_name'] == 'TRAIN'
        assert record_args['requested_anchor'] == params['anchor_date']
        assert record_args['experiment_name'] == 'PRED'
        source.load_object.assert_called_once_with('model.pkl' if kind == 'static' else 'model_fold_0.pkl')
        outputs.append((result, tag_values, runtime.log_params.call_args_list,
                        initialize.call_args.args[0], record_args))
    if ancestry != 'interrupt':
        assert outputs[0] == outputs[1]
