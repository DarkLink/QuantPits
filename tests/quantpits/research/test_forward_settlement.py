"""D1 uses a real temporary C4 publisher, B2 observer and B0 accounting."""
import json
import os
import shutil
import struct
from datetime import datetime, timezone

import pytest

from quantpits.research import forward_settlement as m
from tests.quantpits.research.test_forward_definition_evidence import fresh_evidence_workspace
from tests.quantpits.research.test_forward_bootstrap import fresh_bootstrap_workspace
from tests.quantpits.research.test_forward_intent_preparation import prepared_inputs
from tests.quantpits.research.test_forward_intent_publication import publication, _publish as publish_intent


@pytest.fixture
def settlement(publication, tmp_path, monkeypatch):
    result = publish_intent(publication)
    args, kw = publication
    success = tmp_path / 'success.json'
    success.write_text(json.dumps(result.to_safe_summary_dict(), indent=2) + '\n')
    success.chmod(0o600)
    root = tmp_path / 'settlements'
    root.mkdir(mode=0o700)
    provider = args[3]
    (provider / 'calendars/day.txt').write_bytes(b'2026-08-28\n2026-09-04\n2026-09-07\n')
    for directory in (provider / 'features').iterdir():
        for field, value in [('open', 10.), ('factor', 1.)]:
            (directory / (field + '.day.bin')).write_bytes(struct.pack('<ffff', 0., value, value, value))
    monkeypatch.setattr(m, '_clock', lambda: datetime(2026, 9, 7, 2, tzinfo=timezone.utc))
    return dict(intent_store_root=kw['intent_store_root'], epoch_id=kw['epoch_id'],
        expected_intent_request_digest=result.request_digest, publication_success_record_path=success,
        qlib_provider_root=provider, settlement_store_root=root)


def prepare(kw):
    result = m.prepare_first_forward_settlement(**kw)
    assert result.status == 'READY', result.to_safe_summary_dict()
    return result


def publish(kw):
    plan = prepare(kw)
    result = m.publish_first_forward_settlement(**kw, expected_request_digest=plan.request_digest)
    assert result.status == 'COMMITTED', result.to_safe_summary_dict()
    return result


def inspect(kw, digest):
    return m.inspect_first_forward_settlement(kw['settlement_store_root'], kw['epoch_id'], expected_request_digest=digest)


def test_roundtrip(settlement):
    first = prepare(settlement)
    assert prepare(settlement).request_digest == first.request_digest
    assert not first.to_safe_summary_dict()['did_write']
    assert list(settlement['settlement_store_root'].iterdir()) == []
    result = publish(settlement)
    safe = result.to_safe_summary_dict()
    assert safe['accounting_pair_complete'] and safe['state_chain_ready']
    assert not safe['prospective_claim'] and not safe['epoch_started']
    assert [s.role for s in result.after_states] == list(m.ROLES)
    assert all(s.after_state.as_of_date == '2026-09-07' for s in result.after_states)
    assert inspect(settlement, result.request_digest).after_states == result.after_states
    target = settlement['settlement_store_root'] / settlement['epoch_id']
    assert (target / 'publication_success.json').read_bytes() == settlement['publication_success_record_path'].read_bytes()
    rows = json.loads((target / 'settlements.json').read_bytes())['roles']
    for row in rows:
        transition = row['transition']
        assert transition['transition_status'] == 'COMPLETE'
        assert transition['valuation_status'] == 'COMPLETE'
        assert transition['nav_reconciled']
        assert not transition['prospective_claim']
        assert transition['evidence_class'] == 'RETROSPECTIVE_TECHNICAL_REPLAY'
        assert set(transition['quotes']['requested_instruments']) == (
            {r['instrument'] for r in transition['prior_state']['positions']} |
            {r['instrument'] for r in transition['requested_intents']['intents']})


def test_offline_adoption_without_sources(settlement, monkeypatch):
    result = publish(settlement)
    for key in ('intent_store_root', 'qlib_provider_root'):
        shutil.rmtree(settlement[key])
    settlement['publication_success_record_path'].unlink()
    def forbidden(*a, **kw):
        raise AssertionError('live source forbidden')
    monkeypatch.setattr(m, '_clock', forbidden)
    monkeypatch.setattr(m, '_build', forbidden)
    assert inspect(settlement, result.request_digest).after_states == result.after_states
    adopted = m.publish_first_forward_settlement(**settlement, expected_request_digest=result.request_digest)
    assert adopted.status == 'ADOPTED' and not adopted.to_safe_summary_dict()['did_write']
    assert adopted.after_states == result.after_states


@pytest.mark.parametrize('kind', ['preopen', 'calendar_absent', 'all_missing', 'all_invalid'])
def test_waits_without_writing(settlement, monkeypatch, kind):
    provider = settlement['qlib_provider_root']
    if kind == 'preopen':
        monkeypatch.setattr(m, '_clock', lambda: datetime(2026, 9, 7, 1, tzinfo=timezone.utc))
    elif kind == 'calendar_absent':
        (provider / 'calendars/day.txt').write_bytes(b'2026-08-28\n2026-09-04\n')
    else:
        for path in (provider / 'features').glob('*/open.day.bin'):
            if kind == 'all_missing':
                path.unlink()
            else:
                path.write_bytes(struct.pack('<ffff', 0., 10., 10., float('nan')))
    result = m.prepare_first_forward_settlement(**settlement)
    assert result.status == 'WAITING_FOR_DATA', result.to_safe_summary_dict()
    assert not result.to_safe_summary_dict()['did_write']
    assert not list(settlement['settlement_store_root'].iterdir())


@pytest.mark.parametrize('kind', ['prefix', 'intervening', 'extension'])
def test_calendar_continuity(settlement, kind):
    calendar = settlement['qlib_provider_root'] / 'calendars/day.txt'
    data = calendar.read_bytes()
    if kind == 'prefix':
        data = data.replace(b'2026-08-28', b'2026-08-27')
    elif kind == 'intervening':
        data = data.replace(b'2026-09-07', b'2026-09-05\n2026-09-07')
    else:
        data += b'2026-09-08\n'
    calendar.write_bytes(data)
    result = m.prepare_first_forward_settlement(**settlement)
    assert result.status == ('READY' if kind == 'extension' else 'PRECONDITION_BLOCKED')


@pytest.mark.parametrize('field,value', [('status', 'ADOPTED'), ('status', 'VERIFIED'), ('status', 'UNCERTAIN'),
    ('did_write', False), ('manifest_digest', 'f' * 64), ('request_digest', 'f' * 64),
    ('operation_id', '00000000-0000-0000-0000-000000000000'), ('target_binding_digest', 'f' * 64),
    ('trade_date', '2026-09-08'), ('current_cycle_id', '2026-09-03'), ('prospective_claim', False),
    ('epoch_started', False), ('completion_record_verified', False), ('bundle_verified', False)])
def test_original_success_required(settlement, field, value):
    path = settlement['publication_success_record_path']
    doc = json.loads(path.read_bytes())
    doc[field] = value
    path.write_bytes(m.canonical(doc))
    result = m.prepare_first_forward_settlement(**settlement)
    assert result.status == 'PRECONDITION_BLOCKED'
    assert not list(settlement['settlement_store_root'].iterdir())


@pytest.mark.parametrize('kind', ['order', 'budget', 'backward', 'final_deadline', 'data_time', 'duplicate'])
def test_success_time_evidence(settlement, kind):
    path = settlement['publication_success_record_path']
    doc = json.loads(path.read_bytes())
    events = doc['time_observations']
    if kind == 'order':
        events.reverse()
    elif kind == 'budget':
        events[-1]['elapsed_seconds'] = 601
    elif kind == 'backward':
        events[-1]['elapsed_seconds'] = -1
    elif kind == 'final_deadline':
        events[-1]['at_utc'] = '2026-09-07T01:00:00Z'
    elif kind == 'data_time':
        events[3]['at_utc'] = '2026-09-06T00:00:01Z'
    data = m.canonical(doc)
    if kind == 'duplicate':
        data = data.replace(b'"status":"COMMITTED"', b'"status":"COMMITTED","status":"COMMITTED"')
    path.write_bytes(data)
    assert m.prepare_first_forward_settlement(**settlement).status == 'PRECONDITION_BLOCKED'


def test_metadata_stable_price_revision_mismatch(settlement):
    first = prepare(settlement)
    path = next((settlement['qlib_provider_root'] / 'features').glob('*/open.day.bin'))
    os.utime(str(path), None)
    assert prepare(settlement).request_digest == first.request_digest
    for price in (settlement['qlib_provider_root'] / 'features').glob('*/open.day.bin'):
        price.write_bytes(struct.pack('<ffff', 0., 10., 10., 11.))
    result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    assert result.status == 'REQUEST_MISMATCH' and not result.to_safe_summary_dict()['did_write']


@pytest.mark.parametrize('name', m.MEMBERS + ('manifest.json', 'completion.json'))
@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_write_boundary_failure(settlement, monkeypatch, name, error):
    first = prepare(settlement)
    original = m._write_member
    def write(fd, member, data):
        original(fd, member, data)
        if member == name:
            raise error('private-path-and-amount')
    monkeypatch.setattr(m, '_write_member', write)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    else:
        result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
        safe = result.to_safe_summary_dict()
        assert result.status == 'UNCERTAIN' and safe['did_write']
        assert not safe['state_chain_ready'] and result.after_states is None
        assert 'private-path-and-amount' not in json.dumps(safe)
    assert inspect(settlement, first.request_digest).status == ('VERIFIED' if name == 'completion.json' else 'INCOMPLETE')


@pytest.mark.parametrize('name', m.MEMBERS + ('manifest.json', 'completion.json'))
def test_missing_member_no_repair(settlement, name):
    first = publish(settlement)
    target = settlement['settlement_store_root'] / settlement['epoch_id']
    (target / name).unlink()
    result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    assert result.status == 'INCOMPLETE' and not result.to_safe_summary_dict()['did_write']
    assert not (target / name).exists()


@pytest.mark.parametrize('kind', ['source', 'target', 'root'])
def test_mutation_during_writer(settlement, monkeypatch, kind):
    first = prepare(settlement)
    original = m._write_member
    target = settlement['settlement_store_root'] / settlement['epoch_id']
    def write(fd, member, data):
        original(fd, member, data)
        if member == 'settlements.json':
            if kind == 'source':
                path = settlement['publication_success_record_path']
                old = path.read_bytes()
                path.write_bytes(old + b' ')
                path.write_bytes(old)  # Even restored bytes cannot erase an observed write.
            else:
                path = target if kind == 'target' else settlement['settlement_store_root']
                path.rename(path.with_name(path.name + '.displaced'))
                path.mkdir(mode=0o700)
    monkeypatch.setattr(m, '_write_member', write)
    result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
    assert result.after_states is None


def test_single_arm_failure_continues(settlement, monkeypatch):
    original = m.ShadowPortfolioTransition.apply
    seen = []
    def apply(**kwargs):
        seen.append(kwargs['prior'].portfolio_id)
        if len(seen) == 1:
            raise ValueError('private-accounting-detail')
        return original(**kwargs)
    monkeypatch.setattr(m.ShadowPortfolioTransition, 'apply', apply)
    result = m.prepare_first_forward_settlement(**settlement)
    assert result.status == 'PRECONDITION_BLOCKED' and len(seen) == 2
    assert [r['status'] for r in result.to_safe_summary_dict()['roles']] == ['FAILED', 'COMPLETE']
    assert not list(settlement['settlement_store_root'].iterdir())


def test_missing_held_price_partial_valuation(settlement):
    observed = m.c4.inspect_first_forward_intent_pair(settlement['intent_store_root'], settlement['epoch_id'],
        expected_request_digest=settlement['expected_intent_request_digest'])
    held = observed.d1_inputs[0].prior.positions[0].instrument
    (settlement['qlib_provider_root'] / 'features' / held.lower() / 'open.day.bin').unlink()
    result = publish(settlement)
    safe = result.to_safe_summary_dict()
    assert safe['state_chain_ready'] and safe['accounting_pair_complete']
    assert any(r['valuation_status'] == 'PARTIAL' and not r['nav_available'] for r in safe['roles'])


def test_same_slot_conflict(settlement):
    first = publish(settlement)
    result = m.publish_first_forward_settlement(**settlement, expected_request_digest='f' * 64)
    assert result.status == 'CONFLICT' and not result.to_safe_summary_dict()['did_write']
    assert inspect(settlement, first.request_digest).status == 'VERIFIED'


def test_input_guard_close_failure_after_write(settlement, monkeypatch):
    first = prepare(settlement)
    original = m.c4.c3._PreparationGuards.close
    def close(self):
        original(self)
        raise OSError('private-close')
    monkeypatch.setattr(m.c4.c3._PreparationGuards, 'close', close)
    result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
    assert result.after_states is None
    assert inspect(settlement, first.request_digest).status == 'VERIFIED'


def rehash_bundle(kw, mutate):
    """Rebuild every D1 outer digest so tests exercise semantic validation."""
    target = kw['settlement_store_root'] / kw['epoch_id']
    data = {n: (target / n).read_bytes() for n in m.MEMBERS}
    mutate(data)
    request = json.loads(data['request.json'])
    logical = {n: data[n] for n in m.MEMBERS if n != 'request.json'}
    request['body']['semantic_members'] = m.c4._inventory(logical)
    request['body']['publication_success_digest'] = m._raw(data['publication_success.json'])
    request['request_digest'] = m._hash(request['body'])
    data['request.json'] = m.canonical(request)
    manifest = json.loads((target / 'manifest.json').read_bytes())
    manifest.update(request_digest=request['request_digest'], members=m.c4._inventory(data))
    completion = json.loads((target / 'completion.json').read_bytes())
    completion.update(request_digest=request['request_digest'], manifest_digest=m._hash(manifest))
    for n, raw in data.items():
        (target / n).write_bytes(raw)
    (target / 'manifest.json').write_bytes(m.canonical(manifest))
    (target / 'completion.json').write_bytes(m.canonical(completion))
    return request['request_digest']


@pytest.mark.parametrize('kind', ['cash', 'fee', 'terminal', 'warning', 'roles', 'projection', 'numeric_bool',
                                  'price_words', 'price_derivation', 'price_counts', 'prior', 'assumption', 'source_role'])
def test_rehashed_semantic_tampering_rejected(settlement, kind):
    publish(settlement)
    def mutate(data):
        name = 'settlements.json'
        doc = json.loads(data[name])
        transition = doc['roles'][0]['transition']
        if kind == 'cash':
            transition['after_state']['cash'] = '999999'
        elif kind == 'fee':
            transition['total_fee'] = '123.45'
        elif kind == 'terminal':
            transition['terminal_results'] = []
        elif kind == 'warning':
            transition['warnings'] = []
        elif kind == 'roles':
            doc['roles'].reverse()
        elif kind == 'projection':
            transition['quotes']['requested_instruments'] = []
        elif kind == 'numeric_bool':
            transition['prospective_claim'] = 0
        elif kind.startswith('price_'):
            name = 'next_open_prices.json'
            doc = json.loads(data[name])
            receipt = doc['receipt']
            if kind == 'price_words':
                receipt['rows'][0]['numerator']['feature_float32_bits'] = struct.pack('<f', 42.).hex()
            elif kind == 'price_derivation':
                receipt['rows'][0]['cash_price'] = '42'
            else:
                receipt['counts']['observed'] = 0
            receipt['digest'] = m.prices._digest_payload({k: v for k, v in receipt.items() if k != 'digest'})
        else:
            name = 'source_intent.json'
            doc = json.loads(data[name])
            if kind == 'prior':
                doc['roles'][0]['prior']['cash'] = '42'
            elif kind == 'assumption':
                doc['roles'][0]['execution_assumption']['buy_fee_rate'] = '0.9'
            else:
                doc['roles'].reverse()
        data[name] = m.canonical(doc)
    digest = rehash_bundle(settlement, mutate)
    result = inspect(settlement, digest)
    assert result.status in ('CONFLICT', 'UNCERTAIN') and result.after_states is None
    assert not result.to_safe_summary_dict()['state_chain_ready']


def test_concurrent_create_only_single_winner(settlement, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    first = prepare(settlement)
    barrier = Barrier(2)
    original = m._publish
    def writer(*args):
        barrier.wait(timeout=30)
        return original(*args)
    monkeypatch.setattr(m, '_publish', writer)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(m.publish_first_forward_settlement, **settlement,
                              expected_request_digest=first.request_digest) for _ in range(2)]
        results = [future.result(timeout=60) for future in futures]
    assert sum(r.status == 'COMMITTED' for r in results) == 1, [r.to_safe_summary_dict() for r in results]
    assert sum(r.to_safe_summary_dict()['did_write'] is True for r in results) == 1
    assert all(r.status in ('COMMITTED', 'ADOPTED', 'INCOMPLETE', 'UNCERTAIN') for r in results)
    assert inspect(settlement, first.request_digest).status == 'VERIFIED'


@pytest.mark.parametrize('kind', ['empty', 'zero_orders', 'negative', 'funded'])
def test_economic_edges_from_real_c4(fresh_bootstrap_workspace, tmp_path, monkeypatch, kind):
    from decimal import Decimal
    from tests.quantpits.research.test_forward_bootstrap import _rewrite_source_portfolio
    from tests.quantpits.research.test_forward_portfolio_source import _portfolio
    production = fresh_bootstrap_workspace[0]
    portfolio = _portfolio()
    if kind == 'empty':
        portfolio = dict(current_cash='-1.25', current_holding=[])
    elif kind == 'funded':
        portfolio['current_cash'] = '100000'
    _rewrite_source_portfolio(production, portfolio)
    prepared = prepared_inputs.__wrapped__(fresh_bootstrap_workspace, tmp_path, monkeypatch)
    if kind in ('empty', 'zero_orders'):
        for path in (prepared[0][3] / 'features').glob('*/close.day.bin'):
            path.unlink()
    pub = publication.__wrapped__(prepared, tmp_path, monkeypatch)
    kw = settlement.__wrapped__(pub, tmp_path, monkeypatch)
    if kind == 'empty':
        def forbidden(*a, **kw):
            raise AssertionError('empty exposure must not construct observer receipt')
        monkeypatch.setattr(m.prices, 'QlibCashPriceObserver', forbidden)
    result = publish(kw)
    target = kw['settlement_store_root'] / kw['epoch_id']
    transitions = [r['transition'] for r in json.loads((target / 'settlements.json').read_bytes())['roles']]
    if kind in ('empty', 'zero_orders'):
        assert all(not t['requested_intents']['intents'] for t in transitions)
    if kind == 'empty':
        assert json.loads((target / 'next_open_prices.json').read_bytes())['receipt']['status'] == 'NOT_REQUIRED_EMPTY_EXPOSURE'
        assert all(s.after_state.cash == Decimal('-1.25') for s in result.after_states)
    if kind == 'negative':
        assert all(Decimal(t['prior_state']['cash']) < 0 for t in transitions)
        assert all(Decimal(t['after_state']['cash']) >= Decimal(t['prior_state']['cash']) for t in transitions)
    if kind == 'funded':
        assert all(t['all_filled'] and Decimal(t['total_fee']) > 0 for t in transitions)
        for t in transitions:
            assert Decimal(t['after_state']['cash']) == (Decimal(t['prior_state']['cash']) +
                Decimal(t['gross_sell']) - Decimal(t['gross_buy']) - Decimal(t['total_fee']))
        assert transitions[0]['after_state']['portfolio_id'] != transitions[1]['after_state']['portfolio_id']
    assert inspect(kw, result.request_digest).after_states == result.after_states


@pytest.mark.parametrize('phase', ['before_read', 'after_read', 'before_completion', 'after_completion'])
def test_source_guard_covers_full_read_write_window(settlement, monkeypatch, phase):
    first = prepare(settlement)
    original = m.c4.inspect_first_forward_intent_pair
    def observe(*a, **kw):
        path = settlement['intent_store_root'] / settlement['epoch_id'] / 'plans.json'
        def touch():
            raw = path.read_bytes()
            path.write_bytes(raw)
        if phase == 'before_read':
            touch()
        result = original(*a, **kw)
        if phase == 'after_read':
            touch()
        return result
    monkeypatch.setattr(m.c4, 'inspect_first_forward_intent_pair', observe)
    original_write = m._write_member
    def write(fd, name, data):
        original_write(fd, name, data)
        if (phase, name) in [('before_completion', 'manifest.json'), ('after_completion', 'completion.json')]:
            path = settlement['qlib_provider_root'] / 'calendars/day.txt'
            path.write_bytes(path.read_bytes())
    monkeypatch.setattr(m, '_write_member', write)
    result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    post = phase in ('before_completion', 'after_completion')
    assert result.status == ('UNCERTAIN' if post else 'PRECONDITION_BLOCKED')
    assert result.to_safe_summary_dict()['did_write'] is post


@pytest.mark.parametrize('limit', ['MEMBER_LIMIT', 'TOTAL_LIMIT', 'RECORD_LIMIT'])
def test_budget_prewrite(settlement, monkeypatch, limit):
    monkeypatch.setattr(m, limit, 1)
    result = m.prepare_first_forward_settlement(**settlement)
    assert result.status == 'PRECONDITION_BLOCKED'
    assert not list(settlement['settlement_store_root'].iterdir())


def test_different_arm_outcomes_and_exact_union_projection():
    from tests.quantpits.research.test_accounting import assumption_raw
    assumption = m.ExecutionAssumption.from_dict(assumption_raw())
    inputs = []
    for role, cash, instrument in [('CHAMPION', '5000', 'SH600001'), ('CHALLENGER', '-1.25', 'SZ000001')]:
        prior = m.ShadowPortfolioState.from_dict(dict(portfolio_id=role, as_of_date='2026-09-04', cash=cash, positions=[]))
        batch = m.ShadowIntentBatch.from_iterable(portfolio_id=role, cycle_id='2026-09-04', trade_date='2026-09-07',
            rows=[dict(intent_id=role + '.buy', portfolio_id=role, cycle_id='2026-09-04', trade_date='2026-09-07',
                       side='BUY', instrument=instrument, quantity=100)])
        inputs.append(m.c4.FirstIntentAccountingInputs(role, prior, batch, assumption))
    rows = [dict(instrument=i, status='OBSERVED', cash_price='10') for i in ['SH600001', 'SZ000001']]
    transitions, diagnostics = m._calculate(inputs, rows)
    assert [r['status'] for r in diagnostics] == ['COMPLETE', 'COMPLETE']
    assert [t.filled_count for t in transitions] == [1, 0]
    assert [t.no_fill_count for t in transitions] == [0, 1]
    assert [list(t.quotes.requested_instruments) for t in transitions] == [['SH600001'], ['SZ000001']]
    assert str(transitions[1].after_state.cash) == '-1.25'


@pytest.mark.parametrize('error', [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_accounting_interrupt_releases_all_guards(settlement, monkeypatch, error):
    original = m.c4.c3.surface.SourceMutationObserver
    guards = []
    def observer(*args, **kw):
        guard = original(*args, **kw)
        guards.append(guard)
        return guard
    monkeypatch.setattr(m.c4.c3.surface, 'SourceMutationObserver', observer)
    def fail(**kw):
        raise error()
    monkeypatch.setattr(m.ShadowPortfolioTransition, 'apply', fail)
    with pytest.raises(error):
        m.prepare_first_forward_settlement(**settlement)
    assert guards and all(g.fd == -1 for g in guards)


@pytest.mark.parametrize('kind', ['symlink_root', 'symlink_price', 'hardlink_price', 'directory_price'])
def test_physical_input_failures_are_not_waiting(settlement, tmp_path, kind):
    if kind == 'symlink_root':
        alias = tmp_path / 'alias'
        alias.symlink_to(settlement['settlement_store_root'], target_is_directory=True)
        settlement['settlement_store_root'] = alias
    else:
        path = settlement['qlib_provider_root'] / 'features/sh600001/open.day.bin'
        backup = tmp_path / 'open.bin'
        path.rename(backup)
        if kind == 'symlink_price':
            path.symlink_to(backup)
        elif kind == 'hardlink_price':
            os.link(str(backup), str(path))
        else:
            path.mkdir()
    result = m.prepare_first_forward_settlement(**settlement)
    assert result.status == 'PRECONDITION_BLOCKED' and not result.to_safe_summary_dict()['did_write']


@pytest.mark.parametrize('stage', [1, 2, 3, 4])
@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_directory_fsync_boundaries(settlement, monkeypatch, stage, error):
    import stat
    first = prepare(settlement)
    original = m.os.fsync
    calls = []
    def fsync(fd):
        original(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            calls.append(fd)
            if len(calls) == stage:
                raise error('private-fsync')
    monkeypatch.setattr(m.os, 'fsync', fsync)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    else:
        result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
        assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
        assert not result.to_safe_summary_dict()['state_chain_ready']
    assert inspect(settlement, first.request_digest).status == ('INCOMPLETE' if stage <= 2 else 'VERIFIED')


@pytest.mark.parametrize('side_effect', [False, True])
def test_mkdir_failure_does_not_invent_zero_write(settlement, monkeypatch, side_effect):
    first = prepare(settlement)
    original = m.os.mkdir
    def mkdir(path, *a, **kw):
        if path == settlement['epoch_id']:
            if side_effect:
                original(path, *a, **kw)
            raise OSError('unknown-mkdir-outcome')
        return original(path, *a, **kw)
    monkeypatch.setattr(m.os, 'mkdir', mkdir)
    result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    assert result.status == 'UNCERTAIN'
    assert result.to_safe_summary_dict()['did_write'] is None
    assert (settlement['settlement_store_root'] / settlement['epoch_id']).exists() is side_effect


@pytest.mark.parametrize('kind', ['malformed', 'invalid_start', 'invalid_value'])
def test_individual_invalid_price_semantics(settlement, kind):
    path = settlement['qlib_provider_root'] / 'features/sh600001/open.day.bin'
    if kind == 'malformed':
        path.write_bytes(b'broken')
    elif kind == 'invalid_start':
        path.write_bytes(struct.pack('<ffff', 0.5, 10., 10., 10.))
    else:
        path.write_bytes(struct.pack('<ffff', 0., 10., 10., float('nan')))
    result = m.prepare_first_forward_settlement(**settlement)
    assert result.status == ('READY' if kind == 'invalid_value' else 'PRECONDITION_BLOCKED')
    if kind == 'invalid_value':
        assert any(r['valuation_status'] == 'PARTIAL' for r in result.to_safe_summary_dict()['roles'])


@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_descriptor_close_failure_releases_remaining_descriptors(settlement, monkeypatch, error):
    import stat
    first = prepare(settlement)
    original = m.os.close
    closed = []
    target = settlement['settlement_store_root'] / settlement['epoch_id']
    def close(fd):
        is_target = False
        try:
            info = os.fstat(fd)
            is_target = stat.S_ISDIR(info.st_mode) and target.exists() and info.st_ino == target.stat().st_ino
        except OSError:
            pass
        original(fd)
        closed.append(fd)
        if is_target:
            raise error('close-failure')
    monkeypatch.setattr(m.os, 'close', close)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
    else:
        result = m.publish_first_forward_settlement(**settlement, expected_request_digest=first.request_digest)
        assert result.status == 'UNCERTAIN' and result.to_safe_summary_dict()['did_write']
        assert result.after_states is None
    assert closed
    for fd in set(closed):
        with pytest.raises(OSError):
            os.fstat(fd)
