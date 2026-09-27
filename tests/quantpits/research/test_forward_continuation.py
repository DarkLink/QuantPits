"""Real C4 -> D1 -> D2 -> D2 chain; only upstream admission uses C3 substitutes."""
import io
import json
import shutil
import struct
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from quantpits.research import forward_continuation as m
from tests.quantpits.research.test_forward_definition_evidence import fresh_evidence_workspace
from tests.quantpits.research.test_forward_bootstrap import fresh_bootstrap_workspace
from tests.quantpits.research.test_forward_intent_preparation import prepared_inputs
from tests.quantpits.research.test_forward_intent_publication import publication, _publish
from tests.quantpits.research.test_forward_settlement import publish as publish_first_settlement

DATES = ('2026-08-28', '2026-09-04', '2026-09-07', '2026-09-11', '2026-09-14',
         '2026-09-18', '2026-09-21', '2026-09-25', '2026-09-28')
INSTRUMENTS = ('SH600000', 'SZ000002', 'SZ000001') + tuple('SH6001%02d' % i for i in range(31))


def _calendar(path, dates):
    path.write_text('\n'.join(dates) + '\n')


def _cycle(args, anchor, manifest, seal, monkeypatch, missing=()):
    args = list(args)
    args[4] = anchor
    args[11] = 'signal.' + anchor
    root = args[10] / args[11]
    root.mkdir(mode=0o700)
    from quantpits.research.signal_input_capsule import MEMBER_NAMES
    predictions, inventory = {}, []
    for n, name in enumerate(MEMBER_NAMES):
        values = list(range(len(INSTRUMENTS)))
        if n == 1:
            values[1] = 100
        frame = pd.DataFrame({'score': values}, index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp(anchor), i) for i in INSTRUMENTS], names=('datetime', 'instrument')))
        frame = frame.loc[~frame.index.get_level_values("instrument").isin(missing)]
        stream = io.BytesIO()
        frame.to_pickle(stream)
        data = stream.getvalue()
        (root / name).write_bytes(data)
        inventory.append(dict(logical_path=name, raw_digest=m.c3.surface._digest(data, 'raw_bytes')))
        predictions[str(n)] = data
    (root / 'capsule_manifest.json').write_bytes(m.c4.canonical(dict(members=inventory)))
    (root / 'capsule_manifest.json').chmod(0o600)
    universe = ('\n'.join(i + '\t2020-01-01\t2030-01-01' for i in INSTRUMENTS) + '\n').encode()
    (args[3] / 'instruments/csi300.txt').write_bytes(universe)
    manifest['data_identity']['qlib_materialization_identity']['universe_digest'] = m.c3.surface._digest(universe, 'raw_bytes')
    ranking = m.c3.replay.rank_common_anchor(prediction_bytes_by_member={str(i): predictions[str(i)] for i in range(4)},
        member_order=tuple(str(i) for i in range(4)), anchor_date=anchor, eligible_instruments=INSTRUMENTS)
    if missing:
        fused = pd.DataFrame({'score': [float(r['raw_score']) for r in ranking.rows if r['scored']]},
            index=pd.MultiIndex.from_tuples([(pd.Timestamp(anchor), r['instrument'])
                for r in ranking.rows if r['scored']], names=('datetime', 'instrument')))
        stream = io.BytesIO()
        fused.to_pickle(stream)
        data = stream.getvalue()
        (root / MEMBER_NAMES[4]).write_bytes(data)
        inventory[4]['raw_digest'] = m.c3.surface._digest(data, 'raw_bytes')
        (root / 'capsule_manifest.json').write_bytes(m.c4.canonical(dict(members=inventory)))
    cycle = args[0] / 'data/evidence/v1/cycles' / anchor
    cycle.mkdir(mode=0o700, exist_ok=True)
    (cycle / 'ranking.csv').write_bytes(ranking.to_csv_bytes())
    seal = dict(named_file_digests={'ranking.csv': m.c3.surface._digest(ranking.to_csv_bytes(), 'raw_bytes')})
    original = m.c3.surface._cycle_authority
    monkeypatch.setattr(m.c3.surface, '_cycle_authority', lambda root, day, **kw: (cycle, manifest, seal) if day == anchor else original(root, day))
    _calendar(args[3] / 'calendars/day.txt', [d for d in DATES if d <= anchor])
    _calendar(args[3] / 'calendars/day_future.txt', DATES)
    for instrument in set(INSTRUMENTS) | {'SH600001'}:
        directory = args[3] / 'features' / instrument.lower()
        directory.mkdir(parents=True, exist_ok=True)
        for field, value in [('close', 10. if anchor == '2026-09-04' else .1), ('factor', 1.), ('open', 10. if anchor == '2026-09-04' else .1)]:
            (directory / (field + '.day.bin')).write_bytes(struct.pack('<' + 'f' * (len(DATES) + 1), 0., *([value] * len(DATES))))
    return tuple(args)


@pytest.fixture
def continuation(publication, prepared_inputs, tmp_path, monkeypatch):
    args, first_kw = publication
    args = _cycle(args, '2026-09-04', prepared_inputs[2], prepared_inputs[3], monkeypatch)
    first = _publish((args, first_kw))
    success = tmp_path / 'first_success.json'
    success.write_bytes(m.c4.canonical(first.to_safe_summary_dict()))
    success.chmod(0o600)
    first_settlements = tmp_path / 'first_settlements'
    first_settlements.mkdir(mode=0o700)
    _calendar(args[3] / 'calendars/day.txt', DATES[:3])
    monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 7, 2, tzinfo=timezone.utc))
    settled = publish_first_settlement(dict(intent_store_root=first_kw['intent_store_root'], epoch_id=first_kw['epoch_id'],
        expected_intent_request_digest=first.request_digest, publication_success_record_path=success,
        qlib_provider_root=args[3], settlement_store_root=first_settlements))
    # Economic divergence must originate in actual ranking/planning/settlement.
    assert settled.after_states[0].after_state.positions != settled.after_states[1].after_state.positions
    roots = []
    for name in ('continuing_intents', 'continuing_settlements'):
        root = tmp_path / name
        root.mkdir(mode=0o700)
        (root / first_kw['epoch_id']).mkdir(mode=0o700)
        roots.append(root)
    args = _cycle(args, '2026-09-11', prepared_inputs[2], prepared_inputs[3], monkeypatch)
    monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 13, tzinfo=timezone.utc), 100.))
    kw = dict(intent_store_root=roots[0], settlement_store_root=roots[1], epoch_id=first_kw['epoch_id'], cycle_index=2,
        first_intent_store_root=first_kw['intent_store_root'], expected_first_intent_request_digest=first.request_digest,
        predecessor_settlement_store_root=first_settlements, predecessor_cycle_index=1,
        expected_predecessor_request_digest=settled.request_digest, decision_deadline_utc='2026-09-14T01:00:00Z',
        next_open_utc='2026-09-14T01:30:00Z', market_timezone='Asia/Shanghai')
    return args, kw, settled, prepared_inputs, tmp_path


def prepare_intent(fixture):
    args, kw = fixture[:2]
    result = m.prepare_next_forward_intent_publication(*args, **kw)
    assert result.status == 'READY', result.to_safe_summary_dict()
    return result


def publish_intent(fixture):
    plan = prepare_intent(fixture)
    args, kw = fixture[:2]
    result = m.publish_next_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    assert result.status == 'COMMITTED', result.to_safe_summary_dict()
    return result


def settle(fixture, published, monkeypatch):
    args, kw, _, _, tmp = fixture
    trade = published.to_safe_summary_dict()['trade_date']
    _calendar(args[3] / 'calendars/day.txt', [d for d in DATES if d <= trade])
    now = datetime.fromisoformat(trade + 'T02:00:00+00:00')
    monkeypatch.setattr(m.d1, '_clock', lambda: now)
    path = tmp / ('success_%s.json' % kw['cycle_index'])
    path.write_bytes(m.c4.canonical(published.to_safe_summary_dict()))
    path.chmod(0o600)
    params = dict(intent_store_root=kw['intent_store_root'], epoch_id=kw['epoch_id'], cycle_index=kw['cycle_index'],
        expected_intent_request_digest=published.request_digest, publication_success_record_path=path,
        qlib_provider_root=args[3], settlement_store_root=kw['settlement_store_root'])
    plan = m.prepare_next_forward_settlement(**params)
    assert plan.status == 'READY', plan.to_safe_summary_dict()
    result = m.publish_next_forward_settlement(**params, expected_request_digest=plan.request_digest)
    assert result.status == 'COMMITTED', result.to_safe_summary_dict()
    return result, params


def test_three_persistent_cycles(continuation, monkeypatch):
    args, kw, previous, prepared, tmp = continuation
    for index in (2, 3):
        fixture = (args, kw, previous, prepared, tmp)
        published = publish_intent(fixture)
        safe = published.to_safe_summary_dict()
        assert safe['cycle_intent_published'] and not safe['epoch_started'] and not safe['whole_chain_verified']
        assert [i.prior.to_dict() for i in published.d1_inputs] == [s.after_state.to_dict() for s in previous.after_states]
        settled, params = settle(fixture, published, monkeypatch)
        observation = m.inspect_next_forward_settlement(kw['settlement_store_root'], kw['epoch_id'], index,
            expected_request_digest=settled.request_digest)
        assert observation.status == 'VERIFIED', observation.to_safe_summary_dict()
        assert observation.after_states == settled.after_states
        assert [s.prior_digest for s in settled.after_states] == [s.after_state.digest for s in previous.after_states]
        assert not observation.to_safe_summary_dict()['whole_chain_verified']
        if index == 2:
            previous = observation
            args = _cycle(args, '2026-09-18', prepared[2], prepared[3], monkeypatch)
            kw = dict(kw, cycle_index=3, predecessor_cycle_index=2,
                predecessor_settlement_store_root=kw['settlement_store_root'], expected_predecessor_request_digest=settled.request_digest,
                decision_deadline_utc='2026-09-21T01:00:00Z', next_open_utc='2026-09-21T01:30:00Z')
            monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 20, tzinfo=timezone.utc), 100.))


@pytest.mark.parametrize('dates,anchor,trade,code', [
    (DATES, '2026-09-11', '2026-09-14', None),
    (DATES, '2026-09-18', '2026-09-21', 'GAP_DETECTED'),
    (DATES[:4], '2026-09-11', '2026-09-14', 'SCHEDULE_UNAVAILABLE'),
    (('2026-09-04', '2026-09-09', '2026-09-14'), '2026-09-09', '2026-09-14', None),
    (('2026-09-04', '2026-09-18', '2026-09-21'), '2026-09-18', '2026-09-21', None),
    (('2026-09-03', '2026-09-04', '2026-09-11', '2026-09-14'), '2026-09-11', '2026-09-14', None),
])
def test_weekly_schedule(dates, anchor, trade, code):
    value = dict(schedule=dict(rule=m.RULE, first_anchor='2026-09-04'), predecessor=dict(current_cycle_id='2026-09-04'))
    future = ('\n'.join(dates) + '\n').encode()
    day = ('\n'.join(d for d in dates if d <= anchor) + '\n').encode()
    if code:
        with pytest.raises(m.c4._Invalid) as exc:
            m._schedule(day, future, value, anchor, trade)
        assert exc.value.code == code
    else:
        m._schedule(day, future, value, anchor, trade)


@pytest.mark.parametrize('key,value,code', [
    ('cycle_index', True, 'CYCLE_INDEX_INVALID'), ('cycle_index', '02', 'CYCLE_INDEX_INVALID'),
    ('cycle_index', 3, 'PREDECESSOR_INDEX_INVALID'), ('predecessor_cycle_index', 2, 'PREDECESSOR_INDEX_INVALID'),
    ('expected_predecessor_request_digest', 'f' * 64, 'PREDECESSOR_SETTLEMENT_INVALID'),
    ('expected_first_intent_request_digest', 'f' * 64, 'FIRST_INTENT_INVALID'),
    ('market_timezone', 'UTC', 'SCHEDULE_INVALID'),
])
def test_bad_selection_stops_before_planning(continuation, monkeypatch, key, value, code):
    args, kw = continuation[:2]
    monkeypatch.setattr(m.c3.CurrentRuleShadowIntentPlanner, 'plan', lambda **kwargs: pytest.fail('planning forbidden'))
    result = m.prepare_next_forward_intent_publication(*args, **dict(kw, **{key: value}))
    assert result.status == 'PRECONDITION_BLOCKED'
    assert result.to_safe_summary_dict()['reason_codes'] == [code]
    assert not list((kw['intent_store_root'] / kw['epoch_id']).iterdir())


@pytest.mark.parametrize('kind', ['missing', 'incomplete', 'other_epoch'])
def test_unsettled_predecessor_cannot_advance(continuation, kind):
    args, kw = continuation[:2]
    target = kw['predecessor_settlement_store_root'] / kw['epoch_id']
    if kind == 'missing':
        shutil.rmtree(target)
    elif kind == 'incomplete':
        (target / 'completion.json').unlink()
    else:
        target.rename(target.with_name('other_epoch'))
    result = m.prepare_next_forward_intent_publication(*args, **kw)
    assert result.to_safe_summary_dict()['reason_codes'] == ['PREDECESSOR_SETTLEMENT_INVALID']


@pytest.mark.parametrize('kind,code', [('gap', 'GAP_DETECTED'), ('truncated', 'SCHEDULE_UNAVAILABLE'),
    ('deadline', 'DEADLINE_EXCEEDED'), ('version', 'VERSION_BREAK')])
def test_stop_conditions(continuation, monkeypatch, kind, code):
    args, kw, previous, prepared, tmp = continuation
    if kind == 'gap':
        args = _cycle(args, '2026-09-18', prepared[2], prepared[3], monkeypatch)
    elif kind == 'truncated':
        _calendar(args[3] / 'calendars/day_future.txt', DATES[:4])
    elif kind == 'deadline':
        monkeypatch.setattr(m.c4, '_clock', lambda: (m.c4._utc(kw['decision_deadline_utc']), 100.))
    else:
        from tests.quantpits.research.test_decision_surface import _result
        monkeypatch.setattr(m.c3.surface, 'observe_production_decision_surface',
            lambda *a, **k: _result(change=m.c3.surface.COMPONENT_NAMES[5]))
    result = m.prepare_next_forward_intent_publication(*args, **kw)
    expected = ([code, 'PORTFOLIO_INTENT_POLICY_DIFFERENT', 'STABLE_COMPONENT_DIFFERENT']
                if kind == 'version' else [code])
    assert result.to_safe_summary_dict()['reason_codes'] == expected
    assert result.status == ('VERSION_BREAK' if kind == 'version' else 'PRECONDITION_BLOCKED')


def test_offline_inspect_adopt_and_first_mode_remains_strict(continuation, monkeypatch):
    args, kw = continuation[:2]
    published = publish_intent(continuation)
    settled, params = settle(continuation, published, monkeypatch)
    assert m.c4.inspect_first_forward_intent_pair(kw['intent_store_root'] / kw['epoch_id'], '2',
        expected_request_digest=published.request_digest).status != 'VERIFIED'
    assert m.d1.inspect_first_forward_settlement(kw['settlement_store_root'] / kw['epoch_id'], '2',
        expected_request_digest=settled.request_digest).status != 'VERIFIED'
    for root in (args[0], args[1], args[3], kw['first_intent_store_root'], kw['predecessor_settlement_store_root']):
        shutil.rmtree(root)
    def forbidden(*a, **k):
        pytest.fail('live source forbidden')
    monkeypatch.setattr(m.c3, '_prepare_forward_intent', forbidden)
    monkeypatch.setattr(m.d1, '_build', forbidden)
    monkeypatch.setattr(m.c4, '_clock', forbidden)
    for _ in range(2):
        observed = m.inspect_next_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], 2,
            expected_request_digest=published.request_digest)
        assert observed.status == 'VERIFIED'
        assert not observed.to_safe_summary_dict()['predecessor_join_observed']
        assert not observed.to_safe_summary_dict()['cycle_intent_published']
        observed.d1_metadata.clear()  # Defensive copies never mutate the observation.
        assert observed.d1_metadata
        adopted = m.publish_next_forward_intent_pair(*args, **kw, expected_request_digest=published.request_digest)
        assert adopted.status == 'ADOPTED' and not adopted.to_safe_summary_dict()['did_write']
        result = m.publish_next_forward_settlement(**params, expected_request_digest=settled.request_digest)
        assert result.status == 'ADOPTED' and result.after_states == settled.after_states
    params['publication_success_record_path'].unlink()
    shutil.rmtree(kw['intent_store_root'])
    assert m.inspect_next_forward_settlement(kw['settlement_store_root'], kw['epoch_id'], 2,
        expected_request_digest=settled.request_digest).after_states == settled.after_states


@pytest.mark.parametrize('kind', ['request', 'cycle', 'root'])
def test_existing_slot_and_bound_roots(continuation, monkeypatch, kind):
    args, kw = continuation[:2]
    published = publish_intent(continuation)
    if kind == 'root':
        other = continuation[-1] / 'other_settlements'
        other.mkdir(mode=0o700)
        (other / kw['epoch_id']).mkdir(mode=0o700)
        trade = published.to_safe_summary_dict()['trade_date']
        _calendar(args[3] / 'calendars/day.txt', [d for d in DATES if d <= trade])
        monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 14, 2, tzinfo=timezone.utc))
        success = continuation[-1] / 'success.json'
        success.write_bytes(m.c4.canonical(published.to_safe_summary_dict()))
        success.chmod(0o600)
        result = m.prepare_next_forward_settlement(kw['intent_store_root'], kw['epoch_id'], 2,
            published.request_digest, success, args[3], settlement_store_root=other)
        assert result.status == 'READY', result.to_safe_summary_dict()
    else:
        if kind == 'cycle':
            args = list(args)
            args[4] = '2026-09-18'
        result = m.publish_next_forward_intent_pair(*args, **kw, expected_request_digest='f' * 64)
        assert result.status == 'CONFLICT' and not result.to_safe_summary_dict()['did_write']


@pytest.mark.parametrize('name', ['plans.json', 'completion.json'])
@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_intent_write_failure_and_recovery(continuation, monkeypatch, name, error):
    plan = prepare_intent(continuation)
    args, kw = continuation[:2]
    original = m.c4._write_member
    def write(fd, member, data):
        original(fd, member, data)
        if member == name:
            raise error('PRIVATE_PATH_AND_HOLDINGS')
    monkeypatch.setattr(m.c4, '_write_member', write)
    if error is KeyboardInterrupt:
        with pytest.raises(error):
            m.publish_next_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    else:
        result = m.publish_next_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
        assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
        assert 'PRIVATE' not in json.dumps(result.to_safe_summary_dict())
    observed = m.inspect_next_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], 2,
        expected_request_digest=plan.request_digest)
    assert observed.status == ('VERIFIED' if name == 'completion.json' else 'INCOMPLETE')
    assert not observed.to_safe_summary_dict()['prospective_claim']


@pytest.mark.parametrize('kind', ['predecessor', 'first', 'calendar', 'root'])
def test_guards_retained_through_intent_write(continuation, monkeypatch, kind):
    plan = prepare_intent(continuation)
    args, kw = continuation[:2]
    original = m.c4._write_member
    def write(fd, name, data):
        original(fd, name, data)
        if name == 'plans.json':
            if kind == 'root':
                root = kw['settlement_store_root'] / kw['epoch_id']
                root.rename(root.with_name(root.name + '.old'))
                root.mkdir(mode=0o700)
            else:
                path = (kw['predecessor_settlement_store_root'] / kw['epoch_id'] / 'settlements.json' if kind == 'predecessor'
                    else kw['first_intent_store_root'] / kw['epoch_id'] / 'priors.json' if kind == 'first'
                    else args[3] / 'calendars/day.txt')
                path.write_bytes(path.read_bytes())  # Even transient rewrites invalidate the source window.
    monkeypatch.setattr(m.c4, '_write_member', write)
    result = m.publish_next_forward_intent_pair(*args, **kw, expected_request_digest=plan.request_digest)
    assert result.status == 'UNCERTAIN'
    assert result.to_safe_summary_dict()['reason_codes'] == ['INPUT_STABILITY_LOST']
    assert not result.to_safe_summary_dict()['cycle_intent_published']


@pytest.mark.parametrize('kind', ['role', 'portfolio', 'state', 'index', 'epoch', 'schedule', 'schema'])
def test_rehashed_continuation_semantics(continuation, kind):
    published = publish_intent(continuation)
    args, kw = continuation[:2]
    target = kw['intent_store_root'] / kw['epoch_id'] / '2'
    members = {n: (target / n).read_bytes() for n in m.c4.MEMBERS}
    request = m.c4._json(members['request.json'])
    body = request['body']
    value = body['continuation']
    if kind == 'role':
        value['predecessor']['roles'].reverse()
    elif kind == 'portfolio':
        value['predecessor']['roles'][0]['portfolio_id'] = value['predecessor']['roles'][1]['portfolio_id']
    elif kind == 'state':
        value['predecessor']['roles'][0]['after_state_digest'] = 'f' * 64
    elif kind == 'index':
        value['cycle_index'] = 3
    elif kind == 'epoch':
        value['epoch_id'] = 'other'
    elif kind == 'schema':
        body['schema_version'] = 2.0
    else:
        value['schedule']['rule'] = 'IGNORE_GAPS'
    digest = m.c4._hash(body)
    members['request.json'] = m.c4.canonical(dict(body=body, request_digest=digest))
    definitions = m.c4._json(members['definitions.json'])
    definitions['request_digest'] = digest
    members['definitions.json'] = m.c4.canonical(definitions)
    manifest = m.c4._json((target / 'manifest.json').read_bytes())
    manifest.update(request_digest=digest, members=m.c4._inventory(members))
    with pytest.raises((m.c4._Invalid, m.c3._Blocked)):
        m.c4._validate(members, manifest, kw['epoch_id'], digest, continuing=True)


def test_arm_failure_preserves_diagnostics_and_interrupts(continuation, monkeypatch):
    args, kw = continuation[:2]
    original = m.c3.CurrentRuleShadowIntentPlanner.plan
    calls = []
    def plan(**kwargs):
        calls.append(kwargs['portfolio_id'])
        if len(calls) == 1:
            raise ValueError('private')
        return original(**kwargs)
    monkeypatch.setattr(m.c3.CurrentRuleShadowIntentPlanner, 'plan', plan)
    result = m.prepare_next_forward_intent_publication(*args, **kw)
    assert result.status == 'PRECONDITION_BLOCKED' and len(calls) == 2
    assert [r['status'] for r in result.to_safe_summary_dict()['planning_roles']] == ['FAILED', 'COMPLETE']
    def stop(**kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(m.c3.CurrentRuleShadowIntentPlanner, 'plan', stop)
    with pytest.raises(KeyboardInterrupt):
        m.prepare_next_forward_intent_publication(*args, **kw)


@pytest.fixture
def next_settlement(continuation, monkeypatch):
    args, kw = continuation[:2]
    published = publish_intent(continuation)
    trade = published.to_safe_summary_dict()['trade_date']
    _calendar(args[3] / 'calendars/day.txt', [d for d in DATES if d <= trade])
    monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 14, 2, tzinfo=timezone.utc))
    success = continuation[-1] / 'next_success.json'
    success.write_bytes(m.c4.canonical(published.to_safe_summary_dict()))
    success.chmod(0o600)
    return published, dict(intent_store_root=kw['intent_store_root'], settlement_store_root=kw['settlement_store_root'],
        epoch_id=kw['epoch_id'], cycle_index=2, expected_intent_request_digest=published.request_digest,
        publication_success_record_path=success, qlib_provider_root=args[3])


@pytest.mark.parametrize('kind', ['first_log', 'other_request', 'index', 'epoch_started', 'adopted', 'join', 'time'])
def test_original_current_success_required(next_settlement, continuation, kind):
    published, params = next_settlement
    path = params['publication_success_record_path']
    doc = json.loads(path.read_bytes())
    if kind == 'first_log':
        path.write_bytes((continuation[-1] / 'first_success.json').read_bytes())
    else:
        if kind == 'other_request':
            doc['request_digest'] = 'f' * 64
        elif kind == 'index':
            doc['cycle_index'] = 3
        elif kind == 'epoch_started':
            doc['epoch_started'] = True
        elif kind == 'adopted':
            doc['status'] = 'ADOPTED'
        elif kind == 'join':
            doc['predecessor_join_observed'] = False
        else:
            doc['time_observations'][-1]['elapsed_seconds'] = 601
        path.write_bytes(m.c4.canonical(doc))
    result = m.prepare_next_forward_settlement(**params)
    assert result.status == 'PRECONDITION_BLOCKED'
    assert not list((params['settlement_store_root'] / params['epoch_id']).iterdir())


def test_partial_nav_continues_and_uses_current_holdings(next_settlement, continuation, monkeypatch):
    published, params = next_settlement
    args, kw, previous, prepared, tmp = continuation
    instrument = published.d1_inputs[0].prior.positions[0].instrument
    (args[3] / 'features' / instrument.lower() / 'open.day.bin').unlink()
    plan = m.prepare_next_forward_settlement(**params)
    assert plan.status == 'READY', plan.to_safe_summary_dict()
    assert plan.to_safe_summary_dict()['roles'][0]['valuation_status'] == 'PARTIAL'
    settled = m.publish_next_forward_settlement(**params, expected_request_digest=plan.request_digest)
    assert settled.status == 'COMMITTED' and settled.to_safe_summary_dict()['state_chain_ready']
    args = _cycle(args, '2026-09-18', prepared[2], prepared[3], monkeypatch)
    kw = dict(kw, cycle_index=3, predecessor_cycle_index=2, predecessor_settlement_store_root=kw['settlement_store_root'],
        expected_predecessor_request_digest=settled.request_digest, decision_deadline_utc='2026-09-21T01:00:00Z',
        next_open_utc='2026-09-21T01:30:00Z')
    monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 20, tzinfo=timezone.utc), 100.))
    result = publish_intent((args, kw, settled, prepared, tmp))
    assert [i.prior.to_dict() for i in result.d1_inputs] == [s.after_state.to_dict() for s in settled.after_states]
    target = kw['intent_store_root'] / kw['epoch_id'] / '3'
    receipt = json.loads((target / 'anchor_prices.json').read_bytes())['receipt']
    expected = set(INSTRUMENTS) | {p.instrument for s in settled.after_states for p in s.after_state.positions}
    assert receipt['requested_instruments'] == sorted(expected)
    assert 'SH600001' not in expected  # Bootstrap holding was sold in the first cycle.


@pytest.mark.parametrize('kind', ['version', 'model'])
def test_old_intent_settles_despite_current_break(next_settlement, monkeypatch, kind):
    _, params = next_settlement
    def forbidden(*args, **kwargs):
        pytest.fail('settlement must not observe current definitions/models')
    monkeypatch.setattr(m.c3.surface, 'observe_production_decision_surface', forbidden)
    monkeypatch.setattr(m.c3, '_engine', forbidden)
    plan = m.prepare_next_forward_settlement(**params)
    assert plan.status == 'READY', plan.to_safe_summary_dict()


@pytest.mark.parametrize('kind', ['all_missing', 'preopen', 'arm_failure', 'interrupt'])
def test_settlement_failure_semantics(next_settlement, monkeypatch, kind):
    _, params = next_settlement
    if kind == 'all_missing':
        for path in (params['qlib_provider_root'] / 'features').glob('*/open.day.bin'):
            path.unlink()
    elif kind == 'preopen':
        monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 13, tzinfo=timezone.utc))
    else:
        original = m.d1.ShadowPortfolioTransition.apply
        calls = []
        def apply(**kwargs):
            calls.append(kwargs['prior'].portfolio_id)
            if kind == 'interrupt':
                raise KeyboardInterrupt()
            if len(calls) == 1:
                raise ValueError('private')
            return original(**kwargs)
        monkeypatch.setattr(m.d1.ShadowPortfolioTransition, 'apply', apply)
    if kind == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            m.prepare_next_forward_settlement(**params)
        assert len(calls) == 1
    else:
        result = m.prepare_next_forward_settlement(**params)
        assert result.status == ('PRECONDITION_BLOCKED' if kind == 'arm_failure' else 'WAITING_FOR_DATA')
        if kind == 'arm_failure':
            assert [r['status'] for r in result.to_safe_summary_dict()['roles']] == ['FAILED', 'COMPLETE']
    assert not list((params['settlement_store_root'] / params['epoch_id']).iterdir())


@pytest.mark.parametrize('kind', ['completion', 'fsync', 'close', 'interrupt'])
def test_settlement_write_uncertainty(next_settlement, monkeypatch, kind):
    import os
    _, params = next_settlement
    plan = m.prepare_next_forward_settlement(**params)
    assert plan.status == 'READY', plan.to_safe_summary_dict()
    original = m.d1._write_member
    active = []
    def write(fd, name, data):
        original(fd, name, data)
        if name == 'completion.json':
            active.append(fd)
            if kind in ('completion', 'interrupt'):
                raise KeyboardInterrupt() if kind == 'interrupt' else OSError('private')
    monkeypatch.setattr(m.d1, '_write_member', write)
    if kind in ('fsync', 'close'):
        operation = getattr(os, kind)
        def fail(fd):
            result = operation(fd)
            if fd in active:
                raise OSError('private')
            return result
        monkeypatch.setattr(os, kind, fail)
    if kind == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            m.publish_next_forward_settlement(**params, expected_request_digest=plan.request_digest)
    else:
        result = m.publish_next_forward_settlement(**params, expected_request_digest=plan.request_digest)
        assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
        assert not result.to_safe_summary_dict()['state_chain_ready']
    monkeypatch.setattr(m.d1, '_write_member', original)
    if kind in ('fsync', 'close'):
        monkeypatch.setattr(os, kind, operation)
    result = m.inspect_next_forward_settlement(params['settlement_store_root'], params['epoch_id'], 2,
        expected_request_digest=plan.request_digest)
    assert result.status == 'VERIFIED' and result.to_safe_summary_dict()['state_chain_ready']


def test_concurrent_intent_slot_has_one_committer(continuation, monkeypatch):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    args, kw = continuation[:2]
    selection = {k: v for k, v in kw.items() if k not in ('decision_deadline_utc', 'next_open_utc')}
    owners = [m.c3._PreparationGuards(), m.c3._PreparationGuards()]
    try:
        prepared = [m.c3._prepare_forward_intent(*args, _guard_owner=o, _continuation=selection) for o in owners]
        assert all(p.intent_pair_prepared for p in prepared)
        policy = m.c4._policy(kw['decision_deadline_utc'], kw['next_open_utc'], kw['market_timezone'], m.OPENING_POLICY, continuing=True)
        bundles = [m.c4._build(p, kw['epoch_id'], policy, continuing=True) for p in prepared]
        assert bundles[0][2] == bundles[1][2]
        root, target, identity = m._target(kw['intent_store_root'], kw['epoch_id'], 2)
        bindings = m._bindings(kw['intent_store_root'], kw['settlement_store_root'], kw['epoch_id'], 2,
            body=m.c4._json(bundles[0][0]['request.json'])["body"])
        barrier = Barrier(2)
        original = os.mkdir
        def mkdir(path, *a, **k):
            if path == '2':
                barrier.wait(timeout=10)
            return original(path, *a, **k)
        monkeypatch.setattr(os, 'mkdir', mkdir)
        def run(i):
            gate = m.c4._TimeGate(policy)
            gate.check('BEFORE_PREPARATION')
            gate.check('PREPARATION_VERIFIED')
            members, manifest, digest = bundles[i]
            return m.c4._publish_bundle(root, target, identity, kw['epoch_id'], members, manifest, digest,
                gate, owners[i], cycle_index=2, store_bindings=bindings)[0]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(run, range(2)))
        assert sum(r['status'] == 'COMMITTED' for r in results) == 1, results
        assert sum(r['did_write'] is True for r in results) == 1
    finally:
        for owner in owners:
            owner.close()


@pytest.mark.parametrize('kind', ['nonweekly_first', 'mismatch', 'rule'])
def test_schedule_rejects_invalid_observations(kind):
    value = dict(schedule=dict(rule=m.RULE, first_anchor='2026-09-04'), predecessor=dict(current_cycle_id='2026-09-04'))
    day, future = b'2026-09-04\n2026-09-11\n', b'2026-09-04\n2026-09-11\n2026-09-14\n'
    if kind == 'nonweekly_first':
        future = b'2026-09-04\n2026-09-05\n2026-09-11\n2026-09-14\n'
        day = b'2026-09-04\n2026-09-05\n2026-09-11\n'
    elif kind == 'mismatch':
        day = b'2026-09-04\n2026-09-10\n2026-09-11\n'
    else:
        value['schedule']['rule'] = 'IGNORE_GAPS'
    with pytest.raises(m.c4._Invalid):
        m._schedule(day, future, value, '2026-09-11', '2026-09-14')


def test_success_target_digest_freezes_both_store_bindings(continuation):
    published = publish_intent(continuation)
    args, kw = continuation[:2]
    path = kw['intent_store_root'] / kw['epoch_id'] / '2' / 'completion.json'
    record = json.loads(path.read_bytes())
    record['store_bindings']['settlement'] = 'f' * 64
    path.write_bytes(m.c4.canonical(record))
    observed = m.inspect_next_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], 2,
        expected_request_digest=published.request_digest)
    assert observed.status == 'UNCERTAIN'
    assert observed.to_safe_summary_dict()['reason_codes'] == ['STORE_BINDING_INVALID']
    assert observed.d1_inputs is None
    adopted = m.publish_next_forward_intent_pair(*args, **kw, expected_request_digest=published.request_digest)
    assert adopted.status == 'UNCERTAIN' and not adopted.to_safe_summary_dict()['did_write']


def test_reader_replacement_is_uncertain(continuation, monkeypatch):
    published = publish_intent(continuation)
    _, kw = continuation[:2]
    original = m.c3.surface._read_regular
    count = []
    def read(path, **kwargs):
        result = original(path, **kwargs)
        if path.name == 'plans.json' and path.parent.name == '2':
            count.append(1)
            if len(count) == 2:
                raise m.c4._Invalid('TARGET_REPLACED')
        return result
    monkeypatch.setattr(m.c3.surface, '_read_regular', read)
    observed = m.inspect_next_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], 2,
        expected_request_digest=published.request_digest)
    assert observed.status == 'UNCERTAIN' and observed.d1_inputs is None


@pytest.fixture
def partial_continuation(publication, prepared_inputs, tmp_path, monkeypatch):
    original = _cycle
    def common(args, anchor, manifest, seal, patch):
        missing = ('SH600000',) if anchor == '2026-09-04' else ('SH600100',)
        return original(args, anchor, manifest, seal, patch, missing=missing)
    monkeypatch.setattr(__import__(__name__, fromlist=['_cycle']), '_cycle', common)
    return continuation.__wrapped__(publication, prepared_inputs, tmp_path, monkeypatch)


def test_partial_coverage_changes_keep_predecessor(partial_continuation, monkeypatch):
    fixture = partial_continuation
    published = publish_intent(fixture)
    args, kw, previous = fixture[:3]
    _, target, _ = m._target(kw['intent_store_root'], kw['epoch_id'], 2)
    first_root = kw['first_intent_store_root'] / kw['epoch_id']
    first = m.c3.replay._ranking_from_csv((first_root / 'champion_ranking.csv').read_bytes())
    second = m.c3.replay._ranking_from_csv((target / 'champion_ranking.csv').read_bytes())
    assert {r['instrument'] for r in first.rows if not r['scored']} == {'SH600000'}
    assert {r['instrument'] for r in second.rows if not r['scored']} == {'SH600100'}
    assert first.eligible_count == second.eligible_count == len(INSTRUMENTS)
    body = json.loads((target / 'request.json').read_bytes())['body']
    assert body['input_provenance']['coverage_policy'] == 'COMMON_ANCHOR_RANK_EQUAL_V1'
    assert published.d1_inputs[0].prior.digest == previous.after_states[0].after_state.digest
    held = published.d1_inputs[0].prior.positions[0].instrument
    (args[3] / 'features' / held.lower() / 'open.day.bin').unlink()
    settled, _ = settle(fixture, published, monkeypatch)
    assert settled.status == 'COMMITTED'
    assert any(r['valuation_status'] == 'PARTIAL' for r in settled.to_safe_summary_dict()['roles'])
    from tests.quantpits.research.test_forward_window_report import build
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
    report = build(request)
    assert report['status'] == 'PARTIAL' and len(report['rows']) == 2
    assert report['metrics']['arms'][0]['window_return'] is None


@pytest.fixture
def portable_partial_continuation(publication, prepared_inputs, tmp_path, monkeypatch):
    original = _cycle
    def common(args, anchor, manifest, seal, patch):
        missing = ('SH600000',) if anchor == '2026-09-04' else ('SZ000002',)
        return original(args, anchor, manifest, seal, patch, missing=missing)
    monkeypatch.setattr(__import__(__name__, fromlist=['_cycle']), '_cycle', common)
    return continuation.__wrapped__(publication, prepared_inputs, tmp_path, monkeypatch)


def test_copy_partial_two_cycles_then_publish_and_settle_third(portable_partial_continuation, monkeypatch):
    from tests.quantpits.research.test_forward_window_report import build
    args, kw, previous, prepared, tmp = portable_partial_continuation
    published = publish_intent(portable_partial_continuation)
    plans = json.loads((kw['intent_store_root'] / kw['epoch_id'] / '2' / 'plans.json').read_bytes())
    assert 'SZ000002' in plans['roles'][0]['plan']['holding_classifications']['eligible_unscored_retained']
    held = 'SZ000002'
    (args[3] / 'features' / held.lower() / 'open.day.bin').unlink()
    settled, params = settle(portable_partial_continuation, published, monkeypatch)
    request = dict(schema_version=1, first_intent_store_root=str(kw['first_intent_store_root']),
        first_settlement_store_root=str(kw['predecessor_settlement_store_root']),
        continuing_intent_store_root=str(kw['intent_store_root']),
        continuing_settlement_store_root=str(kw['settlement_store_root']), epoch_id=kw['epoch_id'],
        requested_cycles=[dict(cycle_index=1, current_cycle_id='2026-09-04',
            expected_intent_request_digest=kw['expected_first_intent_request_digest'],
            expected_settlement_request_digest=previous.request_digest),
            dict(cycle_index=2, current_cycle_id='2026-09-11', expected_intent_request_digest=published.request_digest,
                 expected_settlement_request_digest=settled.request_digest)])
    baseline = build(request)
    assert baseline['status'] == 'PARTIAL' and len(baseline['rows']) == 2
    mapping = {}
    for name in ('first_intent_store_root', 'first_settlement_store_root',
                 'continuing_intent_store_root', 'continuing_settlement_store_root'):
        from pathlib import Path
        old = Path(request[name])
        new = tmp / ('restored_' + name)
        shutil.copytree(old, new)
        before = {p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()}
        assert before == {p.relative_to(new): p.read_bytes() for p in new.rglob('*') if p.is_file()}
        mapping[old] = new
        shutil.rmtree(old)
        request[name] = str(new)
    copied_success = tmp / 'restored_success.json'
    shutil.copy2(params['publication_success_record_path'], copied_success)
    params['publication_success_record_path'].unlink()
    report = build(request)
    assert report['semantic_digest'] == baseline['semantic_digest']
    assert report['rows'] == baseline['rows']
    args = _cycle(args, '2026-09-18', prepared[2], prepared[3], monkeypatch)
    kw = dict(kw, intent_store_root=mapping[kw['intent_store_root']],
        settlement_store_root=mapping[kw['settlement_store_root']],
        first_intent_store_root=mapping[kw['first_intent_store_root']], cycle_index=3, predecessor_cycle_index=2,
        predecessor_settlement_store_root=mapping[kw['settlement_store_root']],
        expected_predecessor_request_digest=settled.request_digest,
        decision_deadline_utc='2026-09-21T01:00:00Z', next_open_utc='2026-09-21T01:30:00Z')
    monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 20, tzinfo=timezone.utc), 100.))
    fixture = (args, kw, settled, prepared, tmp)
    third = publish_intent(fixture)
    assert [i.prior.digest for i in third.d1_inputs] == [s.after_state.digest for s in settled.after_states]
    final, _ = settle(fixture, third, monkeypatch)
    request['requested_cycles'].append(dict(cycle_index=3, current_cycle_id='2026-09-18',
        expected_intent_request_digest=third.request_digest, expected_settlement_request_digest=final.request_digest))
    report = build(request)
    assert len(report['rows']) == 3 and report['status'] == 'PARTIAL'


@pytest.mark.parametrize('kind', ['role', 'first', 'definition', 'index', 'after_state'])
def test_portable_binding_rejects_rehashed_wrong_chain(continuation, kind):
    published = publish_intent(continuation)
    kw = continuation[1]
    root = kw['intent_store_root'] / kw['epoch_id'] / '2'
    record = json.loads((root / 'completion.json').read_bytes())
    body = json.loads((root / 'request.json').read_bytes())['body']
    import copy
    wrong = copy.deepcopy(body)
    if kind == 'role':
        record['store_bindings']['intent'], record['store_bindings']['settlement'] = (
            record['store_bindings']['settlement'], record['store_bindings']['intent'])
    elif kind == 'first':
        wrong['continuation']['first_intent']['request_digest'] = 'f' * 64
        record['store_bindings'] = m.c4._logical_stores(wrong)
    elif kind == 'definition':
        wrong['continuation']['definition_request_digest'] = {'other': 'f' * 64}
        record['store_bindings'] = m.c4._logical_stores(wrong)
    elif kind == 'index':
        wrong['continuation']['cycle_index'] = 3
    else:
        wrong['continuation']['predecessor']['roles'][0]['after_state_digest'] = 'f' * 64
    record['target_binding_digest'] = m.c4._portable_binding('intent', wrong, record['manifest_digest'])
    (root / 'completion.json').write_bytes(m.c4.canonical(record))
    result = m.inspect_next_forward_intent_pair(kw['intent_store_root'], kw['epoch_id'], 2,
                                              expected_request_digest=published.request_digest)
    assert result.status != 'VERIFIED' and result.d1_inputs is None


@pytest.mark.parametrize("copy_new_settlement", [False, True])
def test_legacy_continuing_predecessor_accepts_new_writer(continuation, monkeypatch, copy_new_settlement):
    from tests.quantpits.research.legacy_forward_records import convert_intent, physical
    args, kw, previous, prepared, tmp = continuation
    published = publish_intent(continuation)
    success = tmp / 'legacy_success.json'
    success.write_bytes(m.c4.canonical(published.to_safe_summary_dict()))
    success.chmod(0o600)
    convert_intent(kw['intent_store_root'], kw['epoch_id'], success, index=2,
                   stores={role: kw[role + '_store_root'] for role in ('intent', 'settlement')})
    original = (kw['intent_store_root'] / kw['epoch_id'] / '2' / 'completion.json').read_bytes()
    _calendar(args[3] / 'calendars/day.txt', DATES[:5])
    monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 14, 2, tzinfo=timezone.utc))
    params = dict(intent_store_root=kw['intent_store_root'], epoch_id=kw['epoch_id'], cycle_index=2,
        expected_intent_request_digest=published.request_digest, publication_success_record_path=success,
        qlib_provider_root=args[3], settlement_store_root=kw['settlement_store_root'])
    plan = m.prepare_next_forward_settlement(**params)
    assert plan.status == 'READY', plan.to_safe_summary_dict()
    settled = m.publish_next_forward_settlement(**params, expected_request_digest=plan.request_digest)
    assert settled.status == 'COMMITTED', settled.to_safe_summary_dict()
    path = kw['settlement_store_root'] / kw['epoch_id'] / '2' / 'completion.json'
    record = json.loads(path.read_bytes())
    if copy_new_settlement:
        old = kw['settlement_store_root']
        copied = tmp / 'restored_v2_settlement_from_v1_source'
        shutil.copytree(old, copied)
        shutil.rmtree(old)
        kw = dict(kw, settlement_store_root=copied)
        path = copied / kw['epoch_id'] / '2' / 'completion.json'
    else:
        record['schema_version'] = 1
        record['target_binding_digest'] = physical(path.parent.parent, '2')
        path.write_bytes(m.c4.canonical(record))
    frozen = path.read_bytes()
    args = _cycle(args, '2026-09-18', prepared[2], prepared[3], monkeypatch)
    kw = dict(kw, cycle_index=3, predecessor_cycle_index=2,
        predecessor_settlement_store_root=kw['settlement_store_root'], expected_predecessor_request_digest=settled.request_digest,
        decision_deadline_utc='2026-09-21T01:00:00Z', next_open_utc='2026-09-21T01:30:00Z')
    monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 20, tzinfo=timezone.utc), 100.))
    fixture = (args, kw, settled, prepared, tmp)
    third = publish_intent(fixture)
    assert [i.prior.digest for i in third.d1_inputs] == [s.after_state.digest for s in settled.after_states]
    settle(fixture, third, monkeypatch)
    assert path.read_bytes() == frozen
    assert (kw['intent_store_root'] / kw['epoch_id'] / '2' / 'completion.json').read_bytes() == original
    assert json.loads((kw['intent_store_root'] / kw['epoch_id'] / '3' / 'completion.json').read_bytes())['schema_version'] == 2


@pytest.mark.parametrize('kind', ['policy', 'scored_sets', 'missing_reason', 'holding_price'])
def test_partial_c4_semantics_reject_rehashed_changes(partial_continuation, kind):
    published = publish_intent(partial_continuation)
    kw = partial_continuation[1]
    root = kw['intent_store_root'] / kw['epoch_id'] / '2'
    members = {name: (root / name).read_bytes() for name in m.c4.MEMBERS}
    request = json.loads(members['request.json'])
    body = request['body']
    plans = json.loads(members['plans.json'])
    if kind == 'policy':
        body['input_provenance']['coverage_policy'] = 'DROP_UNSCORED_V1'
    elif kind in ('scored_sets', 'missing_reason'):
        ranking = m.c3.replay._ranking_from_csv(members['challenger_ranking.csv'])
        rows = [dict(row) for row in ranking.rows]
        missing = next(r for r in rows if not r['scored'])
        if kind == 'missing_reason':
            missing['coverage_status'] = 'invalid_prediction'
        else:
            scored = next(r for r in rows if r['scored'])
            missing['instrument'], scored['instrument'] = scored['instrument'], missing['instrument']
        from dataclasses import replace
        changed = replace(ranking, rows=tuple(rows))
        members['challenger_ranking.csv'] = changed.to_csv_bytes()
        digest = m.c4._raw(members['challenger_ranking.csv'])
        body['input_provenance']['rankings'][1] = digest
        body['roles'][1]['ranking_digest'] = digest
        plans['roles'][1]['plan']['ranking_digest'] = digest
    else:
        held = published.d1_inputs[0].prior.positions[0].instrument
        prices = plans['roles'][0]['plan']['prices']
        row = next(r for r in prices['rows'] if r['instrument'] == held)
        row['cash_close'] = '999'
        prices['digest'] = m.c4.intent_module._digest({k: v for k, v in prices.items() if k != 'digest'})
        body['roles'][0]['price_digest'] = prices['digest']
    for index, row in enumerate(plans['roles']):
        plan = row['plan']
        plan['result_digest'] = m.c4.intent_module._digest({k: v for k, v in plan.items() if k != 'result_digest'})
        body['roles'][index]['planning_result_digest'] = plan['result_digest']
    members['plans.json'] = m.c4.canonical(plans)
    body['input_digest'] = m.c4._hash(body['input_provenance'])
    body['preparation_digest'] = m.c4._hash(dict(input_digest=body['input_digest'], roles=body['roles']))
    logical = {k: v for k, v in members.items() if k != 'request.json'}
    definitions = json.loads(logical['definitions.json'])
    definitions.pop('request_digest')
    logical['definitions.json'] = m.c4.canonical(definitions)
    body['semantic_members'] = m.c4._inventory(logical)
    digest = m.c4._hash(body)
    request['request_digest'] = digest
    definitions['request_digest'] = digest
    members['definitions.json'] = m.c4.canonical(definitions)
    members['request.json'] = m.c4.canonical(request)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    manifest.update(request_digest=digest, members=m.c4._inventory(members))
    with pytest.raises(Exception):
        m.c4._validate(members, manifest, kw['epoch_id'], digest, continuing=True)


def test_logical_stores_still_require_nonoverlapping_directories(tmp_path):
    left = tmp_path / 'left'
    right = left / 'nested'
    for root in (left, right):
        root.mkdir(mode=0o700)
        (root / 'epoch').mkdir(mode=0o700)
    for intent, settlement in ((left, left), (left, right), (right, left)):
        with pytest.raises(m.c4._Invalid, match='STORE_BINDING_INVALID'):
            m._bindings(intent, settlement, 'epoch', 2, body={})
