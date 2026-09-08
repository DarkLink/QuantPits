import pickle
import struct

import pytest

from quantpits.training.model_identity import ModelIdentityError, model_content_digest, trace_training_origin


def storage(identifier='100', value=1., dtype='FloatStorage'):
    def string(value):
        raw = value.encode()
        return b'X' + struct.pack('<I', len(raw)) + raw
    return (pickle.dumps(119547037146038801333356, protocol=2)
            + pickle.dumps(1001, protocol=2)
            + pickle.dumps({'protocol_version': 1001, 'little_endian': True,
                            'type_sizes': {'short': 2, 'int': 4, 'long': 4}}, protocol=2)
            + b'\x80\x02(' + string('storage') + b'ctorch\n' + dtype.encode() + b'\n'
            + string(identifier) + string('cuda:0') + b'K\x01NtQ.'
            + pickle.dumps([identifier], protocol=2) + struct.pack('<Qf', 1, value))


def model(identifier='100', value=1., config=1, dtype='FloatStorage'):
    return pickle.dumps({'weight': storage(identifier, value, dtype), 'config': config}, protocol=4)


def test_reserialization_keeps_full_model_content_identity():
    assert model('100') != model('123456789012345')
    assert model_content_digest(model('100')) == model_content_digest(model('123456789012345'))


@pytest.mark.parametrize('kwargs', [{'value': 2.}, {'config': 2}, {'dtype': 'IntStorage'}])
def test_weight_configuration_and_dtype_changes_are_not_ignored(kwargs):
    assert model_content_digest(model()) != model_content_digest(model(**kwargs))


def test_storage_aliases_are_not_collapsed():
    shared = pickle.dumps([storage('1'), storage('1')], protocol=4)
    separate = pickle.dumps([storage('1'), storage('2')], protocol=4)
    assert model_content_digest(shared) != model_content_digest(separate)


@pytest.mark.parametrize('data', [b'', b'garbage', model()+b'extra'])
def test_invalid_pickle_fails_closed(data):
    with pytest.raises(ModelIdentityError):
        model_content_digest(data)


def test_pickle_is_never_executed(tmp_path):
    marker = tmp_path / 'executed'
    class Payload:
        def __reduce__(self):
            return (eval, ("open(%r, 'w').write('bad')" % str(marker),))
    assert model_content_digest(pickle.dumps(Payload(), protocol=4))
    assert not marker.exists()


def test_origin_walk_preserves_training_root_across_prediction_copies():
    rows = {'a': {'model': 'M'}, 'b': {'model': 'M', 'mode': 'predict_only',
            'source_record_id': 'a', 'source_experiment': 'E'},
            'c': {'model': 'M', 'mode': 'predict_only', 'source_record_id': 'b',
                  'source_experiment': 'E', 'training_origin_experiment': 'E',
                  'training_origin_record_id': 'a'}}
    assert trace_training_origin('E', 'c', lambda e, r: rows[r], 'M') == {
        'training_origin_record_id': 'a', 'training_origin_experiment': 'E'}


@pytest.mark.parametrize('tags', [
    {'mode': 'predict_only'}, {'source_record_id': 'parent'},
    {'source_record_id': 'a', 'source_experiment': 'E'}, {'model': 'OTHER'},
    {'training_origin_record_id': 'wrong', 'training_origin_experiment': 'E'},
])
def test_unproven_or_conflicting_origin_is_rejected(tags):
    with pytest.raises(ModelIdentityError):
        trace_training_origin('E', 'a', lambda e, r: tags, 'M')


def test_lineage_missing_and_control_interruptions_propagate():
    for error in (FileNotFoundError, KeyboardInterrupt, SystemExit, GeneratorExit):
        def reader(*args):
            raise error()
        with pytest.raises(error):
            trace_training_origin('E', 'a', reader, 'M')


def test_prediction_preserves_availability_without_forging_origin():
    from quantpits.training.model_identity import prediction_origin_tags
    def missing(*args):
        raise FileNotFoundError()
    assert prediction_origin_tags('E', 'a', missing, 'M') == {'training_origin_status': 'UNRESOLVED'}
    def interrupted(*args):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        prediction_origin_tags('E', 'a', interrupted, 'M')


def test_cpcv_training_is_a_terminal_origin():
    assert trace_training_origin('E', 'train', lambda *_: {'model': 'M', 'mode': 'cpcv_train'}, 'M')['training_origin_record_id'] == 'train'


def test_missing_model_identity_is_not_a_training_root():
    with pytest.raises(ModelIdentityError):
        trace_training_origin('E', 'train', lambda *_: {}, 'M')


def test_depth_limit_does_not_accept_partial_ancestry():
    def reader(exp, identifier):
        return {'model': 'M', 'source_experiment': exp, 'source_record_id': str(int(identifier)+1)}
    with pytest.raises(ModelIdentityError):
        trace_training_origin('E', '0', reader, 'M', max_depth=2)
