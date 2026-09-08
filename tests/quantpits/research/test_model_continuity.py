import copy
import pickle

import pytest

from quantpits.evidence.contracts import TypedDigest
from quantpits.research.model_continuity import observe_model_copy_pair
from tests.quantpits.research.test_decision_surface import _lineage
from tests.quantpits.training.test_model_identity import model


@pytest.fixture
def copied_models(tmp_path):
    root = tmp_path / 'workspace'
    experiment = root / 'mlruns/1'
    experiment.mkdir(parents=True)
    (experiment / 'meta.yaml').write_text('name: TRAINING\n')
    manifests = []
    for suffix in ('a', 'b'):
        manifest = _lineage()
        lineage = manifest['model_and_ensemble_lineage']
        for artifact in lineage['source_artifacts']:
            if artifact.get('role') != 'source_training':
                continue
            i = artifact['position']
            identifier = 'source_%d_%s' % (i, suffix)
            run = experiment / identifier
            tags = run / 'tags'
            tags.mkdir(parents=True)
            (run / 'meta.yaml').write_text('experiment_id: "1"\nrun_id: ' + identifier + '\n')
            (tags / 'model').write_text('MODEL_%d' % i)
            if suffix == 'b':
                (tags / 'mode').write_text('predict_only')
                (tags / 'source_record_id').write_text('source_%d_a' % i)
                (tags / 'source_experiment').write_text('TRAINING')
            directory = run / 'artifacts'
            directory.mkdir()
            payloads = {'model.pkl': model('100' if suffix == 'a' else '1234567890123'),
                        'pred.pkl': pickle.dumps({'date': suffix})}
            members = []
            for name, data in payloads.items():
                file = directory / name
                file.write_bytes(data)
                members.append({'path': file.relative_to(root).as_posix(), 'digest': TypedDigest.raw(data).to_dict(),
                                'status': 'observed', 'preservation_status': 'workspace_file', 'detail': ''})
            artifact.update(recorder_id=identifier, artifact_locator=directory.relative_to(root).as_posix(), members=members,
                            artifact_tree_digest=TypedDigest.canonical(
                                [{'path': m['path'], 'digest': m['digest']} for m in members], 'file_inventory').to_dict())
            lineage['source_models'][i]['source_recorder_id'] = identifier
            next(a for a in lineage['source_artifacts'] if a.get('position') == i and 'role' not in a)['source_recorder_id'] = identifier
        manifests.append(manifest)
    return root, manifests


def test_real_readers_admit_copies_with_different_serialization_and_reports(copied_models):
    root, (a, b) = copied_models
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    first, second = observe_model_copy_pair(root, a, root, b)
    assert first == second
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind', ['weight', 'config', 'origin'])
def test_changed_models_or_training_origin_remain_different(copied_models, kind):
    root, (a, b) = copied_models
    artifact = next(x for x in b['model_and_ensemble_lineage']['source_artifacts'] if x.get('role') == 'source_training')
    if kind == 'origin':
        tags = root / artifact['artifact_locator'] / '../tags'
        for name in ('source_record_id', 'source_experiment', 'mode'):
            (tags / name).unlink()
    else:
        member = next(x for x in artifact['members'] if x['path'].endswith('model.pkl'))
        raw = model(value=2.) if kind == 'weight' else model(config=2)
        (root / member['path']).write_bytes(raw)
        member['digest'] = TypedDigest.raw(raw).to_dict()
        artifact['artifact_tree_digest'] = TypedDigest.canonical(
            [{'path': m['path'], 'digest': m['digest']} for m in artifact['members']], 'file_inventory').to_dict()
    first, second = observe_model_copy_pair(root, a, root, b)
    assert first != second


@pytest.mark.parametrize('kind', ['tamper', 'missing_parent', 'cycle', 'symlink'])
def test_missing_tampered_or_ambiguous_evidence_blocks(copied_models, kind):
    root, (a, b) = copied_models
    run = root / 'mlruns/1/source_0_b'
    if kind == 'tamper':
        (run / 'artifacts/model.pkl').write_bytes(model(value=3.))
    elif kind == 'missing_parent':
        (run / 'tags/source_record_id').write_text('missing')
    elif kind == 'cycle':
        (run / 'tags/source_record_id').write_text('source_0_b')
    else:
        file = run / 'artifacts/model.pkl'
        file.unlink()
        file.symlink_to(root / 'mlruns/1/source_0_a/artifacts/model.pkl')
    with pytest.raises(ValueError):
        observe_model_copy_pair(root, a, root, b)


def test_observed_model_change_is_not_accepted(copied_models, monkeypatch):
    from quantpits.research import model_continuity as m
    root, (a, b) = copied_models
    original = m.model_content_digest
    def changed(data):
        value = original(data)
        (root / 'mlruns/1/source_0_b/tags/model').write_text('OTHER')
        return value
    monkeypatch.setattr(m, 'model_content_digest', changed)
    with pytest.raises(ValueError):
        observe_model_copy_pair(root, a, root, b)


def test_process_control_is_not_converted_to_missing_evidence(copied_models, monkeypatch):
    from quantpits.research import model_continuity as m
    root, (a, b) = copied_models
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(m, 'model_content_digest', interrupted)
    with pytest.raises(KeyboardInterrupt):
        observe_model_copy_pair(root, a, root, b)


def test_cpcv_fold_inventory_is_complete_and_ordered(copied_models):
    root, (a, b) = copied_models
    for manifest in (a, b):
        for artifact in manifest['model_and_ensemble_lineage']['source_artifacts']:
            if artifact.get('role') != 'source_training':
                continue
            member = next(x for x in artifact['members'] if x['path'].endswith('/model.pkl'))
            original = root / member['path']
            member['path'] = member['path'].replace('/model.pkl', '/model_fold_0.pkl')
            original.rename(root / member['path'])
            artifact['artifact_tree_digest'] = TypedDigest.canonical(
                [{'path': m['path'], 'digest': m['digest']} for m in artifact['members']], 'file_inventory').to_dict()
    first, second = observe_model_copy_pair(root, a, root, b)
    assert first == second
    artifact = b['model_and_ensemble_lineage']['source_artifacts'][0]
    member = next(x for x in artifact['members'] if x['path'].endswith('model_fold_0.pkl'))
    old = root / member['path']
    member['path'] = member['path'].replace('model_fold_0.pkl', 'model_fold_1.pkl')
    old.rename(root / member['path'])
    artifact['artifact_tree_digest'] = TypedDigest.canonical(
        [{'path': m['path'], 'digest': m['digest']} for m in artifact['members']], 'file_inventory').to_dict()
    with pytest.raises(ValueError):
        observe_model_copy_pair(root, a, root, b)


def test_caller_retains_input_observation_until_preparation_finishes(copied_models):
    root, (a, b) = copied_models
    guards, inventory = [], []
    try:
        observe_model_copy_pair(root, a, root, b, retained_guards=guards, input_inventory=inventory)
        assert guards and len(inventory) == 1
        assert not any(g.mutated() for g in guards)
        (root / 'mlruns/1/source_0_b/tags/source_record_id').write_text('another')
        assert any(g.mutated() for g in guards)
    finally:
        for g in reversed(guards):
            g.close()


def test_new_ambiguous_experiment_invalidates_retained_observation(copied_models):
    root, (a, b) = copied_models
    guards = []
    try:
        observe_model_copy_pair(root, a, root, b, retained_guards=guards)
        duplicate = root / 'mlruns/2'
        duplicate.mkdir()
        (duplicate / 'meta.yaml').write_text('name: TRAINING\n')
        assert any(g.mutated() for g in guards)
    finally:
        for g in reversed(guards):
            g.close()


def test_legacy_wrong_experiment_tag_resolves_only_exact_unique_run_id(copied_models):
    root, (a, b) = copied_models
    (root / 'mlruns/1/source_0_b/tags/source_experiment').write_text('STALE_NAME')
    first, second = observe_model_copy_pair(root, a, root, b)
    assert first == second


@pytest.mark.parametrize('kind', ['duplicate', 'wrong_id', 'wrong_experiment'])
def test_run_metadata_cannot_redirect_or_disambiguate_identity(copied_models, kind):
    root, (a, b) = copied_models
    if kind == 'duplicate':
        directory = root / 'mlruns/2'
        directory.mkdir()
        (directory / 'meta.yaml').write_text('name: OTHER\n')
        duplicate = directory / 'source_0_a'
        duplicate.mkdir()
        (duplicate / 'meta.yaml').write_text('experiment_id: "2"\nrun_id: source_0_a\n')
    else:
        (root / 'mlruns/1/source_0_a/meta.yaml').write_text(
            'experiment_id: "%s"\nrun_id: %s\n' % ('2' if kind == 'wrong_experiment' else '1',
                                                    'wrong' if kind == 'wrong_id' else 'source_0_a'))
    with pytest.raises(ValueError):
        observe_model_copy_pair(root, a, root, b)


def test_duplicate_experiment_names_do_not_override_unique_recorder_identity(copied_models):
    root, (a, b) = copied_models
    directory = root / 'mlruns/2'
    directory.mkdir()
    (directory / 'meta.yaml').write_text('name: TRAINING\n')
    first, second = observe_model_copy_pair(root, a, root, b)
    assert first == second


@pytest.mark.parametrize('change', ['mtime', 'recreate', 'relocate'])
def test_content_inventory_ignores_physical_file_identity(copied_models, tmp_path, change):
    import os
    import shutil
    root, (a, b) = copied_models
    first, second = [], []
    pair = observe_model_copy_pair(root, a, root, b, input_inventory=first)
    tag = root / 'mlruns/1/source_0_b/tags/model'
    if change == 'mtime':
        info = tag.stat()
        os.utime(tag, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
    elif change == 'recreate':
        replacement = tag.with_name('replacement')
        replacement.write_bytes(tag.read_bytes())
        replacement.replace(tag)
    else:
        destination = tmp_path / 'relocated'
        shutil.copytree(root, destination)
        root = destination
    assert observe_model_copy_pair(root, a, root, b, input_inventory=second) == pair
    assert first == second


def test_content_inventory_retains_actual_tag_bytes(copied_models):
    root, (a, b) = copied_models
    inventories = [[], []]
    observe_model_copy_pair(root, a, root, b, input_inventory=inventories[0])
    (root / 'mlruns/1/source_0_b/tags/audit_note').write_text('changed')
    observe_model_copy_pair(root, a, root, b, input_inventory=inventories[1])
    assert inventories[0] != inventories[1]


def test_metadata_change_during_read_still_blocks(copied_models, monkeypatch):
    import os
    from quantpits.research import model_continuity as module
    root, (a, b) = copied_models
    original = module.model_content_digest
    def touched(data):
        tag = root / 'mlruns/1/source_0_a/tags/model'
        info = tag.stat()
        os.utime(tag, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
        return original(data)
    monkeypatch.setattr(module, 'model_content_digest', touched)
    with pytest.raises(ValueError):
        observe_model_copy_pair(root, a, root, b)


def _replace_selected_models(root, manifests, raw):
    for manifest in manifests:
        artifact = next(a for a in manifest['model_and_ensemble_lineage']['source_artifacts']
                        if a.get('role') == 'source_training' and a['position'] == 0)
        member = next(m for m in artifact['members'] if m['path'].endswith('/model.pkl'))
        (root / member['path']).write_bytes(raw)
        member['digest'] = TypedDigest.raw(raw).to_dict()
        artifact['artifact_tree_digest'] = TypedDigest.canonical(
            [{'path': m['path'], 'digest': m['digest']} for m in artifact['members']], 'file_inventory').to_dict()


def test_model_above_32_mib_uses_same_budget_for_read_and_recheck(copied_models, monkeypatch):
    from quantpits.research import decision_surface as surface
    from quantpits.research import model_continuity as module
    root, manifests = copied_models
    raw = pickle.dumps({'model_payload': b'x' * (33 * 1024 * 1024)}, protocol=4)
    _replace_selected_models(root, manifests, raw)
    original = surface._read_regular
    reads = []
    def reader(path, **kwargs):
        if path.name == 'model.pkl':
            reads.append(kwargs['maximum'])
        return original(path, **kwargs)
    monkeypatch.setattr(surface, '_read_regular', reader)
    a, b = observe_model_copy_pair(root, manifests[0], root, manifests[1])
    assert a == b
    assert len(reads) == 16  # Eight model files: actual read and final content check.
    assert set(reads) == {module.MAX_INPUT_BYTES}


def test_model_above_declared_budget_is_rejected_before_parser(copied_models, monkeypatch):
    from quantpits.research import model_continuity as module
    root, (a, b) = copied_models
    path = root / 'mlruns/1/source_0_a/artifacts/model.pkl'
    with path.open('wb') as stream:
        stream.truncate(module.MAX_INPUT_BYTES + 1)
    def unexpected(*args):
        pytest.fail('oversized model reached the parser')
    monkeypatch.setattr(module, 'model_content_digest', unexpected)
    with pytest.raises(ValueError):
        observe_model_copy_pair(root, a, root, b)
