"""Real source binding, surface and engine through C4/D1/D2/D3.

Signal adoption (except the real partial-seal case) and intent-policy extraction are controlled substitutes;
Git, frozen definitions/evidence/bootstrap, lineage, model bytes and readers are real.
"""
import copy
import json
import shutil
from pathlib import Path

import pytest

from quantpits.research import decision_surface as s
from quantpits.research import forward_intent_preparation as c3
from tests.quantpits.research import test_forward_definition_evidence as ev
from tests.quantpits.research import test_forward_bootstrap as bs
from tests.quantpits.research.test_forward_observation import _fresh_split_bundle, _reseal_fresh_cycle, CYCLE
from tests.quantpits.research.test_model_continuity import copied_models
from tests.quantpits.research.test_decision_surface_maintenance import reviewed_engine
from tests.quantpits.research.test_forward_intent_preparation import prepared_inputs as base_prepared_inputs
from tests.quantpits.research.test_forward_intent_publication import publication
from tests.quantpits.research.test_forward_continuation import continuation, publish_intent, settle

_REAL_SURFACE = s.observe_production_decision_surface
_REAL_CYCLE = s._cycle_authority
_REAL_SOURCE = s._source_projection
_REAL_MATCH = s._source_matches_definition
_REAL_ENGINE = c3._engine


def reseal(cycle, manifest, extra=()):
    _reseal_fresh_cycle(cycle, manifest, extra)
    required = set(json.loads((cycle / 'seal.json').read_bytes())['object_digests'])
    # Only temporary fixture objects, after replacing its initial placeholder models.
    for directory in (cycle / 'objects').iterdir():
        for path in directory.iterdir():
            if path.name not in required:
                path.unlink()
        if not list(directory.iterdir()):
            directory.rmdir()


@pytest.fixture
def fresh_evidence_workspace(tmp_path, copied_models):
    production, research, activation, cycle, _ = _fresh_split_bundle(tmp_path)
    origin, variants = copied_models
    shutil.copytree(str(origin / 'mlruns'), str(production / 'mlruns'))
    for run in (production / 'mlruns/1').iterdir():
        if not run.is_dir():
            continue
        params = run / 'params'
        params.mkdir()
        (params / 'fit_start_time').write_text('2020-01-01')
        (params / 'fit_end_time').write_text('2026-08-01')
    manifest = json.loads((cycle / 'manifest.json').read_bytes())
    lineage = manifest['model_and_ensemble_lineage']
    sources = copy.deepcopy(variants[0]['model_and_ensemble_lineage']['source_artifacts'])
    sources = [row for row in sources if row.get('role') == 'source_training']
    extra = []
    for index, source in enumerate(sources):
        for member in source['members']:
            member['preservation_status'] = 'embedded'
            extra.append((production / member['path']).read_bytes())
        lineage['source_models'][index]['source_recorder_id'] = source['recorder_id']
        # Keep frozen source IDs MODEL_A..D, with actual matching ancestry tags.
        family = lineage['source_models'][index]['resolved_key']
        for suffix in ('a', 'b'):
            (production / ('mlruns/1/source_%d_%s/tags/model' % (index, suffix))).write_text(family)
        auxiliary = next(a for a in lineage['source_artifacts'] if a.get('position') == index and 'role' not in a)
        auxiliary['source_recorder_id'] = source['recorder_id']
    lineage['source_artifacts'] = sources + [a for a in lineage['source_artifacts'] if a.get('role') != 'source_training']
    for name in ('ensemble_config.json', 'ensemble_records.json'):
        data = json.dumps({'fixture': name}).encode()
        extras_digest = s._digest(data, 'raw_bytes')
        extra.append(data)
        manifest['referenced_evidence'].append(dict(collection='inputs', evidence_class='ensemble',
            kind='config', locator='config/' + name, locator_type='workspace_file',
            observation=dict(path='config/' + name, detail='', status='observed', preservation_status='embedded', digest=extras_digest)))
    reseal(cycle, manifest, extra)
    shadow = research / 'research/shadow_v1'
    definitions, evidence = shadow / 'definitions', shadow / 'definition_evidence'
    definitions.mkdir(mode=0o700)
    evidence.mkdir(mode=0o700)
    from quantpits.research.forward_definition_publication import prepare_fresh_champion_segment_definition_publication as prepare
    from quantpits.research.forward_definition_publication import publish_fresh_champion_segment_definition_bundle as publish, FRESH_AUTHORIZATION_ACTION
    plan = prepare(production, research, CYCLE, activation, definitions)
    result = publish(production, research, CYCLE, activation, definitions, plan.definition_set_id,
                     dict(plan.request_digest), FRESH_AUTHORIZATION_ACTION)
    assert result.status == 'COMMITTED'
    return production, research, activation, definitions, evidence


@pytest.fixture(params=['maintenance', 'exact'])
def fresh_bootstrap_workspace(fresh_evidence_workspace, tmp_path, request):
    assert ev._fresh_publish(fresh_evidence_workspace).status == 'COMMITTED'
    production, research, activation, definitions, evidence = fresh_evidence_workspace
    engine = tmp_path / 'reviewed-engine'
    engine.mkdir()
    old, new = reviewed_engine(engine)
    # An arbitrary recorder maintenance version, outside the old origin-tag pair.
    from tests.quantpits.research.test_decision_surface import _git
    path = engine / 'quantpits/training/records.py'
    original = path.read_bytes()
    path.write_bytes(original + b'\n# historical recorder maintenance\n')
    _git(engine, 'add', '.')
    _git(engine, 'commit', '-qm', 'recorder maintenance')
    old = _git(engine, 'rev-parse', 'HEAD')
    path.write_bytes(original)
    _git(engine, 'add', '.')
    _git(engine, 'commit', '-qm', 'current recorder implementation')
    new = _git(engine, 'rev-parse', 'HEAD')
    from tests.quantpits.research.test_forward_portfolio_source import _write_bundle, _problem
    cycle = _write_bundle(tmp_path / 'source', problems=(_problem(),))
    destination = production / 'data/evidence/v1/cycles/2026-08-28'
    shutil.move(str(cycle), str(destination))
    cycle = destination
    definition_cycle = production / ('data/evidence/v1/cycles/' + CYCLE)
    definition_manifest = json.loads((definition_cycle / 'manifest.json').read_bytes())
    manifest = json.loads((cycle / 'manifest.json').read_bytes())
    for key in ('model_and_ensemble_lineage', 'data_identity', 'run_evidence'):
        manifest[key] = copy.deepcopy(definition_manifest[key])
    manifest['engine_identity'] = {'commit': old if request.param == 'maintenance' else new}
    for folder in (definition_cycle / 'objects').iterdir():
        target = cycle / 'objects' / folder.name
        target.mkdir(mode=0o700, exist_ok=True)
        for path in folder.iterdir():
            shutil.copyfile(str(path), str(target / path.name))
            (target / path.name).chmod(0o600)
    reseal(cycle, manifest)
    bootstraps = research / 'research/shadow_v1/bootstraps'
    bootstraps.mkdir(mode=0o700)
    return production, research, activation, definitions, evidence, bootstraps


@pytest.fixture
def prepared_inputs(base_prepared_inputs, tmp_path, monkeypatch):
    args, calls, manifest, seal = base_prepared_inputs
    args = list(args)
    args[2] = tmp_path / 'reviewed-engine'
    reference_path, reference, _ = _REAL_CYCLE(args[0], '2026-08-28')
    for key in ('model_and_ensemble_lineage', 'run_evidence'):
        manifest[key] = copy.deepcopy(reference[key])
    for anchor in ('2026-09-04', '2026-09-11'):
        current_path = args[0] / ('data/evidence/v1/cycles/' + anchor)
        current_path.mkdir(mode=0o700, exist_ok=True)
        shutil.copytree(str(reference_path / 'objects'), str(current_path / 'objects'))
    from tests.quantpits.research.test_decision_surface import _git
    manifest['engine_identity'] = {'commit': _git(args[2], 'rev-parse', 'HEAD')}
    monkeypatch.setattr(s, 'observe_production_decision_surface', _REAL_SURFACE)
    monkeypatch.setattr(s, '_source_projection', _REAL_SOURCE)
    monkeypatch.setattr(s, '_source_matches_definition', _REAL_MATCH)
    monkeypatch.setattr(c3, '_engine', _REAL_ENGINE)
    # Current cycle acquisition remains the existing fixture's controlled seal.
    # Reference seal and bootstrap are real and never substituted.
    from quantpits.research.forward_observation import observe_fresh_champion_segment_candidate
    candidate = observe_fresh_champion_segment_candidate(args[0], args[1], CYCLE, args[5])
    monkeypatch.setattr(s, '_intent_projection', lambda *a: s._definition_intent_projection(candidate))
    from quantpits.research import model_artifact_capsule as model
    monkeypatch.setattr(model, 'adopt_definition_bound_model_artifact_capsule', _REAL_MODEL_ADOPT)
    model_args = (args[0], args[1], CYCLE, args[5], args[6], args[7], args[12])
    retained = model.prepare_definition_bound_model_artifact_capsule(*model_args)
    assert retained.status == 'PREPARED', str(retained.to_safe_summary_dict())
    args[13] = retained.capsule_id
    published = model.publish_definition_bound_model_artifact_capsule(*model_args, retained.capsule_id,
        dict(retained.request_digest), model.AUTHORIZATION_ACTION)
    assert published.status == 'COMMITTED', published.to_safe_summary_dict()
    checked = c3.prepare_first_forward_intent(*args)
    assert checked.status == 'PREPARED', str(checked.to_safe_summary_dict()['reason_codes'])
    return tuple(args), calls, manifest, seal


def test_c4_d1_d2_d3_maintenance_readers(continuation, monkeypatch):
    args, kw, previous, prepared, tmp = continuation
    published = publish_intent(continuation)
    from quantpits.research import forward_continuation as d2
    _, target, _ = d2._target(kw['intent_store_root'], kw['epoch_id'], 2)
    first_body = json.loads((kw['first_intent_store_root'] / kw['epoch_id'] / 'request.json').read_bytes())['body']
    next_body = json.loads((target / 'request.json').read_bytes())['body']
    for body in (first_body, next_body):
        provenance = body['input_provenance']
        assert 'maintenance_admission' not in provenance
        change = provenance['source_change']
        assert change['production_comparison']['comparison'] in ('EQUAL', 'DIFFERENT')
        assert change['production_execution_comparison'] == 'EQUAL'
    settled, _ = settle(continuation, published, monkeypatch)
    from quantpits.research import forward_window_report as report
    request = dict(schema_version=1, first_intent_store_root=str(kw['first_intent_store_root']),
        first_settlement_store_root=str(kw['predecessor_settlement_store_root']),
        continuing_intent_store_root=str(kw['intent_store_root']),
        continuing_settlement_store_root=str(kw['settlement_store_root']), epoch_id=kw['epoch_id'],
        requested_cycles=[dict(cycle_index=1, current_cycle_id='2026-09-04',
            expected_intent_request_digest=kw['expected_first_intent_request_digest'],
            expected_settlement_request_digest=previous.request_digest),
            dict(cycle_index=2, current_cycle_id='2026-09-11',
            expected_intent_request_digest=published.request_digest,
            expected_settlement_request_digest=settled.request_digest)])
    # Use the same public report entry as the existing D3 reader suite.
    from tests.quantpits.research.test_forward_window_report import build
    result = build(request)
    assert result['status'] == 'COMPLETE', result
    assert all(row['source_context']['input_provenance']['source_change'] for row in result['rows'])
    assert 'Production vs execution' in report.render_markdown(result)


def test_real_surface_rejects_same_day_bootstrap(prepared_inputs):
    args = prepared_inputs[0]
    with pytest.raises(s.DecisionSurfaceContractError, match='chronology'):
        _REAL_SURFACE(args[1], args[0], args[2], '2026-08-28', args[5], args[6],
                      args[7], args[8], args[9], reference_source='production')


def test_arbitrary_source_change_observed_by_c3_and_c4(publication, tmp_path, monkeypatch):
    args, kw = publication
    baseline = c3.prepare_first_forward_intent(*args)
    assert baseline.status == 'PREPARED'
    from tests.quantpits.research.test_forward_intent_publication import _publish
    baseline_root = tmp_path / 'baseline_intents'
    baseline_root.mkdir(mode=0o700)
    baseline_kw = dict(kw, intent_store_root=baseline_root)
    baseline_published = _publish((args, baseline_kw))
    from tests.quantpits.research.test_decision_surface import _git
    path = args[2] / 'quantpits/utils/train_utils.py'
    original = path.read_bytes()
    path.write_bytes(original + b'\n# arbitrary source change\n')
    _git(args[2], 'add', '.')
    _git(args[2], 'commit', '-qm', 'changed predictions')
    # Production sealed this historical code; research runs the actual loaded
    # files, restored below. No runtime or source comparison is mocked.
    manifest = s._cycle_authority(args[0], args[4])[1]
    manifest['engine_identity']['commit'] = _git(args[2], 'rev-parse', 'HEAD')
    path.write_bytes(original)
    from quantpits.research import forward_intent_publication as c4
    prepared = c3.prepare_first_forward_intent(*args)
    assert prepared.status == 'PREPARED', prepared.to_safe_summary_dict()
    change = json.loads(prepared.pair.input_provenance)['source_change']
    assert change['production_comparison']['comparison'] == 'DIFFERENT'
    assert change['production_execution_comparison'] == 'DIFFERENT'
    assert change['execution_commit'] == change['production_commit']
    assert change['execution_code_digest'] != change['production_digest']
    result = c4.prepare_first_forward_intent_publication(*args, **kw)
    assert result.status == 'READY', result.to_safe_summary_dict()
    from tests.quantpits.research.test_forward_intent_publication import _publish
    published = _publish((args, kw))
    read = c4.inspect_first_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'],
                                              expected_request_digest=published.request_digest)
    assert read.status == 'VERIFIED' and read.d1_inputs is not None
    assert [p.to_dict() for p in baseline.pair.plans] == [p.to_dict() for p in prepared.pair.plans]
    assert baseline.to_safe_summary_dict()['input_digest'] != prepared.to_safe_summary_dict()['input_digest']
    import struct
    from datetime import datetime, timezone
    from quantpits.research import forward_settlement as d1
    from tests.quantpits.research.test_forward_settlement import publish as publish_settlement
    (args[3] / 'calendars/day.txt').write_bytes(b'2026-08-28\n2026-09-04\n2026-09-07\n')
    for directory in (args[3] / 'features').iterdir():
        for field, value in [('open', 10.), ('factor', 1.)]:
            (directory / (field + '.day.bin')).write_bytes(struct.pack('<ffff', 0., value, value, value))
    monkeypatch.setattr(d1, '_clock', lambda: datetime(2026, 9, 7, 2, tzinfo=timezone.utc))
    states = []
    for label, result, store in [('baseline', baseline_published, baseline_root),
                                 ('changed', published, kw['intent_store_root'])]:
        success = tmp_path / (label + '_success.json')
        success.write_bytes(c4.canonical(result.to_safe_summary_dict()))
        success.chmod(0o600)
        root = tmp_path / (label + '_settlements')
        root.mkdir(mode=0o700)
        params = dict(intent_store_root=store, epoch_id=kw['epoch_id'],
            expected_intent_request_digest=result.request_digest, publication_success_record_path=success,
            qlib_provider_root=args[3], settlement_store_root=root)
        settled = publish_settlement(params)
        states.append([row.after_state.to_dict() for row in settled.after_states])
        repeated = d1.publish_first_forward_settlement(**params, expected_request_digest=settled.request_digest)
        assert repeated.status == 'ADOPTED' and not repeated.to_safe_summary_dict()['did_write']
        read_back = d1.inspect_first_forward_settlement(root, kw['epoch_id'], expected_request_digest=settled.request_digest)
        assert [row.after_state.to_dict() for row in read_back.after_states] == states[-1]
    assert states[0] == states[1]


def test_common_partial_real_seal_capsule_c3_c4(publication, prepared_inputs, monkeypatch):
    """Only model capsule/intent extraction remain upstream fixture substitutes."""
    from quantpits.research import signal_input_capsule as capsule
    from quantpits.research import forward_intent_publication as c4
    from tests.quantpits.research.test_forward_intent_publication import _publish
    args, kw = publication
    args = list(args)
    cycle = args[0] / 'data/evidence/v1/cycles' / args[4]
    reference_path, reference, _ = _REAL_CYCLE(args[0], '2026-08-28')
    manifest = copy.deepcopy(reference)
    manifest['cycle_identity']['cycle_id'] = args[4]
    manifest['data_identity'] = copy.deepcopy(prepared_inputs[2]['data_identity'])
    manifest['engine_identity'] = copy.deepcopy(prepared_inputs[2]['engine_identity'])
    # One unheld and one held eligible member are missing; full inventory survives.
    from tests.quantpits.research.test_forward_intent_preparation import INSTRUMENTS, ANCHOR
    from tests.quantpits.research.test_common_coverage import prediction
    universe = INSTRUMENTS + tuple('SH%06d' % (700000 + i) for i in range(243))
    missing = {INSTRUMENTS[0], INSTRUMENTS[1], universe[-1]}
    ids = tuple(i for i in universe if i not in missing)
    universe_bytes = ('\n'.join(i + '\t2020-01-01\t2030-01-01' for i in universe) + '\n').encode()
    (args[3] / 'instruments/csi300.txt').write_bytes(universe_bytes)
    manifest['data_identity']['qlib_materialization_identity']['universe_digest'] = s._digest(universe_bytes, 'raw_bytes')
    raw = prediction(ids, anchor=ANCHOR)
    inputs = {str(i): raw for i in range(4)}
    ranking = c3.replay.rank_common_anchor(prediction_bytes_by_member=inputs,
        member_order=tuple(inputs), anchor_date=ANCHOR, eligible_instruments=universe)
    fused = prediction(ids, values=[i / (len(ids) - 1) for i in range(len(ids))], anchor=ANCHOR)
    extras = [raw, fused]
    for row in manifest['model_and_ensemble_lineage']['source_artifacts']:
        if 'role' in row:
            continue
        value = fused if row['position'] == 'ensemble' else raw
        row['members'] = [dict(path=row['artifact_locator'] + '/pred.pkl',
            digest=s._digest(value, 'raw_bytes'), status='observed', preservation_status='embedded', detail='')]
        row['artifact_tree_digest'] = s._digest([
            dict(path=m['path'], digest=m['digest']) for m in row['members']], 'file_inventory')
    commands = dict(prediction='static_train', post_trade='prod_post_trade', order='order_gen')
    for kind, command in commands.items():
        value = c4.canonical(dict(schema_version=1, command=command, status='success', run_id='FIXTURE'))
        extras.append(value)
        manifest['run_evidence'].append(None)
        manifest['run_evidence'][-1] = {'class': kind, 'path': 'run/' + kind + '.json',
            'status': 'observed', 'digest': s._digest(value, 'raw_bytes'), 'preservation_status': 'embedded', 'detail': ''}
    manifest['ranking'] = dict(status='partial', ranking_digest=s._digest(ranking.to_csv_bytes(), 'raw_bytes'),
        eligible_count=246, scored_count=243, missing_count=3)
    manifest['status'] = 'sealed_partial'
    manifest['problems'].append(dict(code='ranking_coverage_partial', evidence_class='ranking',
        blocks_complete=True, detail='common missing rows'))
    for name in ('portfolio_state.json',):
        shutil.copyfile(str(reference_path / name), str(cycle / name))
        (cycle / name).chmod(0o600)
    (cycle / 'ranking.csv').write_bytes(ranking.to_csv_bytes())
    (cycle / 'ranking.csv').chmod(0o600)
    seal = json.loads((reference_path / 'seal.json').read_bytes())
    seal['cycle_id'] = ANCHOR
    seal['named_file_digests']['ranking.csv'] = manifest['ranking']['ranking_digest']
    (cycle / 'seal.json').write_bytes(c4.canonical(seal))
    (cycle / 'seal.json').chmod(0o600)
    reseal(cycle, manifest, extras)
    monkeypatch.setattr(s, '_cycle_authority', _REAL_CYCLE)
    with pytest.raises(s._ComponentIncomparable, match='CYCLE_BLOCKING_PROBLEM_NOT_SCOPED'):
        _REAL_CYCLE(args[0], ANCHOR)
    _REAL_CYCLE(args[0], ANCHOR, allow_partial_ranking=True)
    before = {p: p.read_bytes() for p in cycle.rglob('*') if p.is_file()}
    # Restore the real adoption function replaced by the base C3 fixture.
    monkeypatch.setattr(capsule, 'adopt_signal_input_capsule', _REAL_SIGNAL_ADOPT)
    plan = capsule.prepare_signal_input_capsule(args[0], args[1], ANCHOR, args[10])
    assert plan.status == 'PREPARED', plan.to_safe_dict()
    args[11] = plan.capsule_id
    retained = capsule.publish_signal_input_capsule(args[0], args[1], ANCHOR, args[10], args[11],
        dict(plan.request_digest), capsule.AUTHORIZATION_ACTION)
    assert retained.status == 'COMMITTED', retained.to_safe_dict()
    adopted = capsule.adopt_signal_input_capsule(args[0], args[1], ANCHOR, args[10], args[11])
    assert adopted.status == 'ADOPTED' and adopted.critical_signal_retention_complete
    prepared = c3.prepare_first_forward_intent(*args)
    assert prepared.status == 'PREPARED', prepared.to_safe_summary_dict()
    for role in prepared.to_safe_summary_dict()['roles']:
        assert role['coverage_counts'] == dict(eligible=246, scored=243, missing=3)
    for prior, ranked, prices in zip(*prepared.pair[:3]):
        assert prices.requested_instruments == tuple(sorted(set(ids) | {p.instrument for p in prior.positions}))
    result = _publish((tuple(args), kw))
    read = c4.inspect_first_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], expected_request_digest=result.request_digest)
    assert read.status == 'VERIFIED' and read.d1_inputs is not None
    assert not read.to_safe_summary_dict()['prospective_claim']
    assert all(p.read_bytes() == data for p, data in before.items())
    for prior, plan in zip(prepared.pair[0], prepared.pair[3]):
        missing_held = {p.instrument for p in prior.positions} & missing
        assert set(plan.holding_classifications['eligible_unscored_retained']) == missing_held
        assert not set(plan.selected_buy_instruments) & missing
    # Corruption remains distinct from a valid common missing-row observation.
    object_path = cycle / 'objects' / s._digest(raw, 'raw_bytes')['value'][:2] / s._digest(raw, 'raw_bytes')['value']
    for corrupt in (False, True):
        if corrupt:
            object_path.write_bytes(b'broken')
        else:
            object_path.unlink()
        rejected = capsule.prepare_signal_input_capsule(args[0], args[1], ANCHOR, args[10])
        assert rejected.status == 'PRECONDITION_BLOCKED'
        object_path.write_bytes(raw)
        object_path.chmod(0o600)
    for kind in ('unknown_problem', 'wrong_origin'):
        changed = copy.deepcopy(manifest)
        if kind == 'unknown_problem':
            changed['problems'].append(dict(code='unknown', evidence_class='ranking', blocks_complete=True, detail=''))
        else:
            changed['model_and_ensemble_lineage']['source_models'][0]['source_recorder_id'] = 'WRONG'
        reseal(cycle, changed)
        rejected = capsule.prepare_signal_input_capsule(args[0], args[1], ANCHOR, args[10])
        assert rejected.status == 'PRECONDITION_BLOCKED'
        reseal(cycle, manifest)
    seal_bytes = (cycle / 'seal.json').read_bytes()
    altered = json.loads(seal_bytes)
    altered['manifest_digest']['value'] = '0' * 64
    (cycle / 'seal.json').write_bytes(c4.canonical(altered))
    assert capsule.prepare_signal_input_capsule(args[0], args[1], ANCHOR, args[10]).status == 'PRECONDITION_BLOCKED'
    (cycle / 'seal.json').write_bytes(seal_bytes)



from quantpits.research.signal_input_capsule import adopt_signal_input_capsule as _REAL_SIGNAL_ADOPT

from quantpits.research.model_artifact_capsule import adopt_definition_bound_model_artifact_capsule as _REAL_MODEL_ADOPT


@pytest.mark.parametrize('missing', ['reference', 'current', 'identity'])
def test_unavailable_source_history_does_not_block_inputs(publication, missing):
    args, kw = publication
    manifest = (_REAL_CYCLE(args[0], '2026-08-28')[1] if missing == 'reference'
                else s._cycle_authority(args[0], args[4])[1])
    commit = manifest['engine_identity']['commit']
    if missing == 'identity':
        manifest['engine_identity'] = None
    elif missing == 'current':
        manifest['engine_identity']['commit'] = 'f' * 40
    else:
        # Keep HEAD available even when the reference shares its commit.
        from tests.quantpits.research.test_decision_surface import _git
        _git(args[2], 'commit', '--allow-empty', '-qm', 'execution identity')
        (args[2] / '.git/objects' / commit[:2] / commit[2:]).unlink()
    prepared = c3.prepare_first_forward_intent(*args)
    assert prepared.status == 'PREPARED', prepared.to_safe_summary_dict()
    change = json.loads(prepared.pair.input_provenance)['source_change']
    assert change['production_comparison']['comparison'] == 'INCOMPARABLE'
    if missing in ('current', 'identity'):
        assert change['production_execution_comparison'] == 'INCOMPARABLE'
    from quantpits.research import forward_intent_publication as c4
    # The current HEAD must remain readable for execution provenance. The
    # reference-only case has that independent authority available.
    result = c4.prepare_first_forward_intent_publication(*args, **kw)
    assert result.status == 'READY', result.to_safe_summary_dict()


@pytest.fixture
def maintained_portable_chain(publication, prepared_inputs, tmp_path, monkeypatch):
    from tests.quantpits.research import test_forward_continuation as chain
    from tests.quantpits.research.test_decision_surface import _git
    original = chain._cycle
    def maintenance(args, anchor, manifest, seal, patch, missing=()):
        result = original(args, anchor, manifest, seal, patch, missing=missing)
        cycle = args[0] / 'data/evidence/v1/cycles' / anchor
        if not (cycle / 'objects').exists():
            shutil.copytree(args[0] / 'data/evidence/v1/cycles/2026-08-28/objects', cycle / 'objects')
        path = args[2] / 'quantpits/training/records.py'
        loaded = path.read_bytes()
        path.write_bytes(loaded + ('\n# recorder maintenance ' + anchor + '\n').encode())
        _git(args[2], 'add', '.')
        _git(args[2], 'commit', '-qm', 'production recorder ' + anchor)
        manifest['engine_identity']['commit'] = _git(args[2], 'rev-parse', 'HEAD')
        # Research uses the independently observed working files after sealing.
        path.write_bytes(loaded)
        return result
    monkeypatch.setattr(chain, '_cycle', maintenance)
    return chain.portable_partial_continuation.__wrapped__(publication, prepared_inputs, tmp_path, monkeypatch)


def test_source_changes_copy_partial_three_periods(maintained_portable_chain, monkeypatch):
    from tests.quantpits.research import test_forward_continuation as chain
    chain.test_copy_partial_two_cycles_then_publish_and_settle_third(maintained_portable_chain, monkeypatch)
    tmp = maintained_portable_chain[-1]
    requests = sorted(tmp.glob('restored_*intent_store_root/**/request.json'))
    changes = [json.loads(p.read_bytes())['body']['input_provenance']['source_change'] for p in requests]
    assert len(changes) == 3
    assert len({c['production_commit'] for c in changes}) == 3
    assert all(c['production_execution_comparison'] == 'DIFFERENT' for c in changes)


@pytest.mark.parametrize('kind', ['schema', 'comparison', 'execution', 'production', 'production_commit', 'unknown_field'])
def test_source_provenance_semantics_after_rehash(publication, kind):
    from quantpits.research import forward_intent_publication as c4
    args, kw = publication
    prepared = c3.prepare_first_forward_intent(*args)
    policy = c4._policy(kw['decision_deadline_utc'], kw['next_open_utc'], kw['market_timezone'], kw['opening_policy'])
    members, _, _ = c4._build(prepared, kw['epoch_id'], policy)
    request = json.loads(members['request.json'])
    body = request['body']
    change = body['input_provenance']['source_change']
    if kind == 'schema':
        change['schema_version'] = True
    elif kind == 'comparison':
        change['production_execution_comparison'] = 'DIFFERENT'
    elif kind == 'execution':
        change['execution_implementation_digest'] = '0' * 64
    elif kind == 'production':
        change['production_digest']['value'] = '0' * 64
    elif kind == 'production_commit':
        change['production_commit'] = None
    else:
        change['approved'] = True
    body['input_digest'] = c4._hash(body['input_provenance'])
    body['preparation_digest'] = c4._hash(dict(input_digest=body['input_digest'], roles=body['roles']))
    digest = c4._hash(body)
    request['request_digest'] = digest
    definitions = json.loads(members['definitions.json'])
    definitions['request_digest'] = digest
    members['definitions.json'] = c4.canonical(definitions)
    members['request.json'] = c4.canonical(request)
    manifest = dict(schema_version=1, domain='FIRST_FORWARD_INTENT_PAIR_BUNDLE_V1', epoch_id=kw['epoch_id'],
                    current_cycle_id=args[4], request_digest=digest, members=c4._inventory(members))
    with pytest.raises(c4._Invalid):
        c4._validate(members, manifest, kw['epoch_id'], digest)


@pytest.mark.parametrize('kind', ['model', 'ranking'])
def test_real_input_errors_still_block_after_source_relaxation(prepared_inputs, kind):
    args = prepared_inputs[0]
    if kind == 'model':
        from quantpits.research.model_artifact_capsule import PAYLOAD_NAME
        (args[12] / args[13] / PAYLOAD_NAME).write_bytes(b'{}')
        reason = 'MODEL_ADOPTION_FAILED'
    else:
        cycle, _, seal = s._cycle_authority(args[0], args[4])
        ranking = c3.replay._ranking_from_csv((cycle / 'ranking.csv').read_bytes())
        scores = {row['instrument']: -float(row['raw_score']) for row in ranking.rows}
        data = c3.replay.canonical_full_ranking(tuple(scores), scores).to_csv_bytes()
        (cycle / 'ranking.csv').write_bytes(data)
        seal['named_file_digests']['ranking.csv'] = s._digest(data, 'raw_bytes')
        reason = 'CHAMPION_PARITY_ORDER'
    result = c3.prepare_first_forward_intent(*args)
    assert result.status == 'PRECONDITION_BLOCKED', result.to_safe_summary_dict()
    assert reason in result.to_safe_summary_dict()['reason_codes']


def test_legacy_maintenance_provenance_reader(publication, tmp_path):
    """Reconstruct the old retained format; no live maintenance authority granted."""
    from quantpits.research import forward_intent_publication as c4
    from quantpits.research.model_continuity import observe_model_copy_pair
    args, kw = publication
    prepared = c3.prepare_first_forward_intent(*args)
    policy = c4._policy(kw['decision_deadline_utc'], kw['next_open_utc'], kw['market_timezone'], kw['opening_policy'])
    members, _, _ = c4._build(prepared, kw['epoch_id'], policy)
    engine = tmp_path / 'legacy_engine'
    engine.mkdir()
    old, new = reviewed_engine(engine)
    body = json.loads(members['request.json'])['body']
    provenance = body['input_provenance']
    provenance.pop('source_change')
    provenance['maintenance_admission'] = dict(protocol='RESEARCH_MAINTENANCE_ADMISSION_V1',
        rule_id=s.ORIGIN_TAGS_RULE, reference_raw_digest=s._digest(s._git_blob_projection(engine, old)),
        current_raw_digest=s._digest(s._git_blob_projection(engine, new)),
        dependency_raw_digest=s._digest((engine / s.ORIGIN_TAGS_HELPER).read_bytes(), 'raw_bytes'))
    reference = _REAL_CYCLE(args[0], '2026-08-28')[1]
    inventory = []
    provenance['model_copy_continuity'] = observe_model_copy_pair(args[0], reference, args[0], reference,
                                                                input_inventory=inventory)
    provenance['model_copy_observed_inputs'] = inventory
    body['input_digest'] = c4._hash(provenance)
    body['preparation_digest'] = c4._hash(dict(input_digest=body['input_digest'], roles=body['roles']))
    digest = c4._hash(body)
    definitions = json.loads(members['definitions.json'])
    definitions['request_digest'] = digest
    members['definitions.json'] = c4.canonical(definitions)
    members['request.json'] = c4.canonical(dict(body=body, request_digest=digest))
    manifest = dict(schema_version=1, domain='FIRST_FORWARD_INTENT_PAIR_BUNDLE_V1', epoch_id=kw['epoch_id'],
                    current_cycle_id=args[4], request_digest=digest, members=c4._inventory(members))
    c4._validate(members, manifest, kw['epoch_id'], digest)
