"""C4 exercises real C3 arithmetic; only upstream admission uses C3's fixtures."""
import json
import os
from datetime import datetime, timezone, timedelta

import pytest

from quantpits.research import forward_intent_publication as m
from tests.quantpits.research.test_forward_definition_evidence import fresh_evidence_workspace
from tests.quantpits.research.test_forward_bootstrap import fresh_bootstrap_workspace
from tests.quantpits.research.test_forward_intent_preparation import prepared_inputs


@pytest.fixture
def publication(prepared_inputs, tmp_path, monkeypatch):
    root = tmp_path / 'intent_pairs'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(m, '_clock', lambda: (datetime(2026, 9, 6, tzinfo=timezone.utc), 100.0))
    kwargs = dict(intent_store_root=root, epoch_id='epoch.first',
                  decision_deadline_utc='2026-09-07T01:00:00Z', next_open_utc='2026-09-07T01:30:00Z',
                  market_timezone='Asia/Shanghai', opening_policy=m.OPENING_POLICY)
    return prepared_inputs[0], kwargs


def _prepare(publication):
    args, kwargs = publication
    result = m.prepare_first_forward_intent_publication(*args, **kwargs)
    assert result.status == 'READY', result.to_safe_summary_dict()
    return result


def _publish(publication):
    plan = _prepare(publication)
    args, kwargs = publication
    result = m.publish_first_forward_intent_pair(*args, **kwargs, expected_request_digest=plan.request_digest)
    assert result.status == 'COMMITTED', result.to_safe_summary_dict()
    return result


def test_roundtrip(publication):
    args, kw = publication
    prepared = m.c3.prepare_first_forward_intent(*args)
    m._build(prepared, kw['epoch_id'], m._policy(kw['decision_deadline_utc'], kw['next_open_utc'], kw['market_timezone'], kw['opening_policy']))
    first = _prepare(publication)
    assert _prepare(publication).request_digest == first.request_digest
    result = _publish(publication)
    assert result.to_safe_summary_dict()['prospective_claim']
    args, kwargs = publication
    observed = m.inspect_first_forward_intent_pair(kwargs['intent_store_root'], kwargs['epoch_id'], expected_request_digest=result.request_digest)
    assert observed.status == 'VERIFIED', observed.to_safe_summary_dict()
    assert observed.to_safe_summary_dict()['d1_readable']
    assert not observed.to_safe_summary_dict()['prospective_claim']
    assert len(observed.d1_inputs) == 2
    assert observed.d1_inputs[0].prior.portfolio_id != observed.d1_inputs[1].prior.portfolio_id
    assert all(i.intents.portfolio_id == i.prior.portfolio_id for i in observed.d1_inputs)


def test_adoption_without_live_sources(publication, monkeypatch):
    result = _publish(publication)
    args, kw = publication
    def forbidden(*a, **k):
        raise AssertionError('live preparation forbidden')
    monkeypatch.setattr(m.c3, '_prepare_first_forward_intent', forbidden)
    monkeypatch.setattr(m, '_clock', forbidden)
    adopted = m.publish_first_forward_intent_pair(*([None] * 14), **kw, expected_request_digest=result.request_digest)
    assert adopted.status == 'ADOPTED'
    safe = adopted.to_safe_summary_dict()
    assert not safe['did_write'] and not safe['epoch_started'] and not safe['prospective_claim']
    assert safe['recorded_time_claim'] == result.to_safe_summary_dict()['recorded_time_claim']
    assert adopted.d1_inputs == result.d1_inputs


@pytest.mark.parametrize('name', m.MEMBERS + ('manifest.json', 'completion.json'))
def test_missing_member_cannot_adopt(publication, name):
    result = _publish(publication)
    args, kw = publication
    (kw['intent_store_root'] / kw['epoch_id'] / name).unlink()
    result = m.inspect_first_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], expected_request_digest=result.request_digest)
    assert result.status == 'INCOMPLETE'
    assert not result.to_safe_summary_dict()['d1_readable']
    assert result.d1_inputs is None


def test_same_slot_conflict(publication):
    result = _publish(publication)
    args, kw = publication
    conflict = m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest='f' * 64)
    assert conflict.status == 'CONFLICT'
    assert not conflict.to_safe_summary_dict()['did_write']
    (kw['intent_store_root'] / kw['epoch_id'] / 'extra').write_text('extra')
    assert m.inspect_first_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], expected_request_digest=result.request_digest).status == 'CONFLICT'


def test_request_mismatch_before_write(publication):
    args, kw = publication
    result = m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest='f' * 64)
    assert result.status == 'REQUEST_MISMATCH'
    assert list(kw['intent_store_root'].iterdir()) == []


@pytest.mark.parametrize('name', m.MEMBERS + ('manifest.json', 'completion.json'))
@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_write_boundary_failure(publication, monkeypatch, name, error):
    plan = _prepare(publication)
    args, kw = publication
    original = m._write_member
    def write(fd, member, data):
        original(fd, member, data)
        if member == name:
            raise error('private detail')
    monkeypatch.setattr(m, '_write_member', write)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    else:
        result = m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
        assert result.status == 'UNCERTAIN'
        safe = result.to_safe_summary_dict()
        assert safe['did_write'] and not safe['prospective_claim'] and not safe['epoch_started']
        assert 'private detail' not in json.dumps(safe)
    target = kw['intent_store_root'] / kw['epoch_id']
    assert (target / name).exists()
    observation = m.inspect_first_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], expected_request_digest=plan.request_digest)
    assert observation.status == ('VERIFIED' if name == 'completion.json' else 'INCOMPLETE')
    assert not observation.to_safe_summary_dict()['prospective_claim']


@pytest.mark.parametrize('event', ['BEFORE_PREPARATION', 'BEFORE_WRITE', 'DATA_BUNDLE_VERIFIED', 'FINAL_PUBLICATION_VERIFIED'])
@pytest.mark.parametrize('fault', ['deadline', 'backward', 'monotonic', 'naive'])
def test_clock_failures(publication, monkeypatch, event, fault):
    plan = _prepare(publication)
    args, kw = publication
    original = m._TimeGate.check
    def check(self, name):
        if name == event:
            wall, mono = m._clock()
            if fault == 'deadline':
                wall = m._utc(kw['decision_deadline_utc'])
                mono += (wall - m._clock()[0]).total_seconds()
            elif fault == 'backward':
                wall -= timedelta(seconds=3)
            elif fault == 'monotonic':
                mono = float('nan')
            else:
                wall = wall.replace(tzinfo=None)
            monkeypatch.setattr(m, '_clock', lambda: (wall, mono))
        return original(self, name)
    monkeypatch.setattr(m._TimeGate, 'check', check)
    result = m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    if event == 'BEFORE_PREPARATION' and fault == 'backward':
        assert result.status == 'COMMITTED'  # No earlier observation to compare.
        return
    post = event in ('DATA_BUNDLE_VERIFIED', 'FINAL_PUBLICATION_VERIFIED')
    assert result.status == ('UNCERTAIN' if post else 'PRECONDITION_BLOCKED'), result.to_safe_summary_dict()
    assert result.to_safe_summary_dict()['did_write'] == post
    assert not result.to_safe_summary_dict()['prospective_claim']


def test_source_change_during_writer(publication, monkeypatch):
    plan = _prepare(publication)
    args, kw = publication
    original = m._write_member
    def write(fd, name, data):
        original(fd, name, data)
        if name == 'plans.json':
            path = args[3] / 'calendars/day.txt'
            path.write_bytes(path.read_bytes() + b'2026-09-05\n')
    monkeypatch.setattr(m, '_write_member', write)
    result = m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
    assert 'INPUT_STABILITY_LOST' in result.to_safe_summary_dict()['reason_codes']


def test_target_replacement_during_writer(publication, monkeypatch):
    plan = _prepare(publication)
    args, kw = publication
    original = m._write_member
    target = kw['intent_store_root'] / kw['epoch_id']
    def write(fd, name, data):
        original(fd, name, data)
        if name == 'plans.json':
            target.rename(target.with_name('displaced'))
            target.mkdir(mode=0o700)
    monkeypatch.setattr(m, '_write_member', write)
    result = m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
    assert list(target.iterdir()) == []


def test_symlink_root_rejected(publication, tmp_path):
    args, kw = publication
    alias = tmp_path / 'alias'
    alias.symlink_to(kw['intent_store_root'], target_is_directory=True)
    kw = dict(kw, intent_store_root=alias)
    result = m.prepare_first_forward_intent_publication(*args, **kw)
    assert result.status == 'PRECONDITION_BLOCKED'


def test_digest_order_independent(publication):
    result = _publish(publication)
    args, kw = publication
    target = kw['intent_store_root'] / kw['epoch_id']
    import hashlib
    request = json.loads((target / 'request.json').read_bytes())
    assert hashlib.sha256(m.canonical(request['body'])).hexdigest() == result.request_digest
    for row in request['body']['semantic_members']:
        data = (target / row['name']).read_bytes()
        if row['name'] == 'definitions.json':
            value = json.loads(data)
            assert value.pop('request_digest') == result.request_digest
            data = m.canonical(value)
        assert hashlib.sha256(data).hexdigest() == row['sha256']
    assert 'request.json' not in [r['name'] for r in request['body']['semantic_members']]


def test_reader_cross_fields_even_with_rehashed_inventory(publication):
    args, kw = publication
    prepared = m.c3.prepare_first_forward_intent(*args)
    members, manifest, digest = m._build(prepared, kw['epoch_id'], m._policy(kw['decision_deadline_utc'], kw['next_open_utc'], kw['market_timezone'], kw['opening_policy']))
    receipt = json.loads(members['anchor_prices.json'])['receipt']
    receipt['rows'][0]['numerator']['feature_float32_bits'] = '0000803f'
    receipt['digest'] = m.prices_module._digest_payload({k: v for k, v in receipt.items() if k != 'digest'})
    with pytest.raises(m._Invalid):
        m._validate_words(receipt, m.c3._calendar(members['calendar_day.txt']), args[4])


def test_concurrent_slot_has_at_most_one_commit(publication, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    args, kw = publication
    prepared = m.c3.prepare_first_forward_intent(*args)
    policy = m._policy(kw['decision_deadline_utc'], kw['next_open_utc'], kw['market_timezone'], kw['opening_policy'])
    members, manifest, digest = m._build(prepared, kw['epoch_id'], policy)
    root, target, identity = m._target(kw['intent_store_root'], kw['epoch_id'])
    barrier = Barrier(2)
    original = os.mkdir
    def mkdir(path, *a, **k):
        if path == kw['epoch_id']:
            barrier.wait(timeout=10)
        return original(path, *a, **k)
    monkeypatch.setattr(os, 'mkdir', mkdir)
    class Owner:
        def check(self):
            pass
        def close(self):
            pass
    def run():
        gate = m._TimeGate(policy)
        gate.check('BEFORE_PREPARATION')
        return m._publish_bundle(root, target, identity, kw['epoch_id'], members, manifest, digest, gate, Owner())[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(), range(2)))
    assert sum(r['status'] == 'COMMITTED' for r in results) == 1, results
    assert sum(r['did_write'] is True for r in results) == 1
    assert all(r['status'] in ('COMMITTED', 'ADOPTED', 'INCOMPLETE', 'UNCERTAIN') for r in results)


def test_metadata_touch_keeps_request_but_bytes_change_does_not(publication):
    first = _prepare(publication)
    args, kw = publication
    path = args[3] / 'features/sh600000/close.day.bin'
    os.utime(path, None)
    assert _prepare(publication).request_digest == first.request_digest
    import struct
    path.write_bytes(struct.pack('<fff', 0., 11., 11.))
    assert _prepare(publication).request_digest != first.request_digest


def test_business_partial_and_zero_intents(publication):
    args, kw = publication
    for path in (args[3] / 'features').glob('*/close.day.bin'):
        path.unlink()
    result = _publish(publication)
    assert all(not item.intents.intents for item in result.d1_inputs)
    assert result.to_safe_summary_dict()['order_counts'] == [0, 0]


@pytest.mark.parametrize('limit', ['MEMBER_LIMIT', 'TOTAL_LIMIT'])
def test_bundle_budget_prewrite(publication, monkeypatch, limit):
    monkeypatch.setattr(m, limit, 1)
    args, kw = publication
    result = m.prepare_first_forward_intent_publication(*args, **kw)
    assert result.status == 'PRECONDITION_BLOCKED'
    assert result.to_safe_summary_dict()['reason_codes'] == ['BUNDLE_BUDGET_EXCEEDED']
    assert not list(kw['intent_store_root'].iterdir())


def test_final_guard_close_failure_is_uncertain(publication, monkeypatch):
    plan = _prepare(publication)
    args, kw = publication
    original = m.c3._PreparationGuards.close
    def close(owner):
        had_guards = bool(owner.guards)
        original(owner)
        if had_guards:
            raise OSError('private close error')
    monkeypatch.setattr(m.c3._PreparationGuards, 'close', close)
    result = m.publish_first_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    assert result.status == 'UNCERTAIN'
    assert result.to_safe_summary_dict()['did_write']
    assert not result.to_safe_summary_dict()['prospective_claim']


@pytest.mark.parametrize('kind', ['role', 'intent', 'prior', 'price_projection', 'definition', 'ranking', 'schema_bool'])
def test_semantic_mutation_rejected_after_all_outer_hashes_rebuilt(publication, kind):
    args, kw = publication
    prepared = m.c3.prepare_first_forward_intent(*args)
    policy = m._policy(kw['decision_deadline_utc'], kw['next_open_utc'], kw['market_timezone'], kw['opening_policy'])
    members, _, _ = m._build(prepared, kw['epoch_id'], policy)
    name = 'plans.json'
    doc = json.loads(members[name])
    plan = doc['roles'][0]['plan']
    if kind == 'role':
        doc['roles'].reverse()
    elif kind == 'intent':
        plan['intents']['portfolio_id'] = 'other'
    elif kind == 'prior':
        name = 'priors.json'
        doc = json.loads(members[name])
        doc['roles'][0]['state']['cash'] = '999'
    elif kind == 'price_projection':
        plan['prices']['rows'][0]['cash_close'] = '999'
    elif kind == 'definition':
        plan['definition_digest'] = '0' * 64
    elif kind == 'ranking':
        plan['ranking_digest'] = '0' * 64
    else:
        doc['schema_version'] = True
    members[name] = m.canonical(doc)
    # Independently rebuild the entire non-cyclic envelope: the semantic
    # validators must still reject incompatible accounting/report fields.
    request = json.loads(members['request.json'])
    logical = {k: v for k, v in members.items() if k != 'request.json'}
    definitions = json.loads(logical['definitions.json'])
    definitions.pop('request_digest')
    logical['definitions.json'] = m.canonical(definitions)
    request['body']['semantic_members'] = m._inventory(logical)
    digest = m._hash(request['body'])
    request['request_digest'] = digest
    definitions['request_digest'] = digest
    members['definitions.json'] = m.canonical(definitions)
    members['request.json'] = m.canonical(request)
    manifest = dict(schema_version=1, domain='FIRST_FORWARD_INTENT_PAIR_BUNDLE_V1', epoch_id=kw['epoch_id'],
                    current_cycle_id=args[4], request_digest=digest, members=m._inventory(members))
    with pytest.raises(Exception):
        m._validate(members, manifest, kw['epoch_id'], digest)


def test_existing_contradictory_member_is_conflict(publication):
    result = _publish(publication)
    _, kw = publication
    target = kw['intent_store_root'] / kw['epoch_id']
    path = target / 'plans.json'
    path.write_bytes(path.read_bytes() + b' ')
    observed = m.inspect_first_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'],
        expected_request_digest=result.request_digest)
    assert observed.status == 'CONFLICT'
    assert not observed.to_safe_summary_dict()['did_write']
