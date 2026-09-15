"""Real source binding, surface and engine through C4/D1/D2/D3.

Capsule adoption and intent-policy extraction are controlled upstream substitutes;
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
    reference = _REAL_CYCLE(args[0], '2026-08-28')[1]
    reference_digest = s._digest(s._git_blob_projection(args[2], s._engine_commit(reference)))
    compatible = reference_digest['value'] == s._ORIGIN_TAGS_OLD
    for body in (first_body, next_body):
        provenance = body['input_provenance']
        assert ('maintenance_admission' in provenance) is compatible
        if compatible:
            s.validate_maintenance_admission(provenance['maintenance_admission'])
            assert provenance['model_copy_continuity'][0] == provenance['model_copy_continuity'][1]
            assert provenance['model_copy_observed_inputs']
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


def test_real_surface_rejects_same_day_bootstrap(prepared_inputs):
    args = prepared_inputs[0]
    with pytest.raises(s.DecisionSurfaceContractError, match='chronology'):
        _REAL_SURFACE(args[1], args[0], args[2], '2026-08-28', args[5], args[6],
                      args[7], args[8], args[9], reference_source='production')


def test_unreviewed_economic_change_rejected_by_c3_and_c4(publication):
    args, kw = publication
    from tests.quantpits.research.test_decision_surface import _git
    path = args[2] / 'quantpits/utils/train_utils.py'
    path.write_text(path.read_text().replace('fm.predict(dataset=dataset)', 'fm.predict(dataset=dataset) * 2'))
    _git(args[2], 'add', '.')
    _git(args[2], 'commit', '-qm', 'changed predictions')
    # The selected current seal, supplied by the existing fixture, now names
    # this implementation. Refusal must come from surface, before runtime gates.
    manifest = s._cycle_authority(args[0], args[4])[1]
    manifest['engine_identity']['commit'] = _git(args[2], 'rev-parse', 'HEAD')
    from quantpits.research import forward_intent_publication as c4
    for result in (c3.prepare_first_forward_intent(*args), c4.prepare_first_forward_intent_publication(*args, **kw)):
        assert result.status == 'VERSION_BREAK', result.to_safe_summary_dict()
        assert 'ECONOMIC_COMPATIBILITY_NOT_ESTABLISHED' in result.to_safe_summary_dict()['reason_codes']
    assert not list(kw['intent_store_root'].iterdir())
