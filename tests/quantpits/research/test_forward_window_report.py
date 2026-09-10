"""D3 formulas and actual C4/D1/D2 persisted window observations."""
import copy
import json
from decimal import Decimal, localcontext
from datetime import datetime, timezone

import pytest

from quantpits.research import forward_window_report as m
from tests.quantpits.research.test_forward_continuation import (
    fresh_evidence_workspace, fresh_bootstrap_workspace, prepared_inputs, publication,
    continuation, publish_intent, settle, _cycle,
)


def formula_rows(posts=((99, 99), (109, 108)), pres=((100, 100), (110, 109))):
    rows = []
    for index, (post, pre) in enumerate(zip(posts, pres), 1):
        arms = []
        for j, role in enumerate(m.ROLES):
            a = m._empty_arm(role)
            a.update(nav_before=str(pre[j]), nav_after=str(post[j]), valuation_status='COMPLETE',
                     cost='1', gross_buy='10', gross_sell='0', reason_codes=[])
            arms.append(a)
        rows.append(dict(cycle_index=index, status='SETTLED', arms=arms))
    return rows


def test_golden_post_to_post_and_baseline_drawdown():
    rows = formula_rows()
    with localcontext() as ctx:
        ctx.prec = 64
        basis, metrics, reasons = m._metrics(rows)
        assert rows[1]['arms'][0]['period_return'] == m._ratio(Decimal(109) / 99 - 1)
    assert not reasons and basis['value'] == '100'
    assert [r['arms'][0]['normalized_nav'] for r in rows] == ['0.99', '1.09']
    assert rows[0]['arms'][0]['drawdown'] == '-0.01'
    assert metrics['arms'][0]['window_return'] == '0.09'
    assert metrics['arms'][0]['max_drawdown'] == '-0.01'
    assert metrics['return_difference'] == '-0.01'
    assert metrics['arms'][0]['total_cost'] == '2'


@pytest.mark.parametrize('pre', [(0, 0), (-1, -1), (100, 101)])
def test_invalid_base(pre):
    rows = formula_rows(pres=(pre, (110, 109)))
    basis, metrics, _ = m._metrics(rows)
    assert basis['status'] == 'INVALID'
    assert all(a['normalized_nav'] is None for r in rows for a in r['arms'])
    assert metrics['arms'][0]['window_return'] is None


def test_negative_nav_and_partial_do_not_hide_risk():
    rows = formula_rows(posts=((-10, 0), (20, 20)))
    _, metrics, _ = m._metrics(rows)
    assert rows[0]['arms'][0]['normalized_nav'] == '-0.1'
    assert metrics['arms'][0]['max_drawdown'] == '-1.1'
    assert rows[1]['arms'][0]['period_return'] is None
    rows = formula_rows()
    rows[0]['arms'][0].update(valuation_status='PARTIAL', nav_before=None, nav_after=None)
    _, metrics, reasons = m._metrics(rows)
    assert 'WINDOW_NAV_INCOMPLETE' in reasons
    assert metrics['arms'][1]['window_return'] is None
    assert m._overlap([], []) == dict(intersection_count=0, union_count=0, jaccard='1', empty_both=True)


@pytest.fixture
def three_window(continuation, monkeypatch):
    args, kw, previous, prepared, tmp = continuation
    request = dict(schema_version=1, first_intent_store_root=str(kw['first_intent_store_root']),
        first_settlement_store_root=str(kw['predecessor_settlement_store_root']),
        continuing_intent_store_root=str(kw['intent_store_root']),
        continuing_settlement_store_root=str(kw['settlement_store_root']), epoch_id=kw['epoch_id'],
        requested_cycles=[dict(cycle_index=1, current_cycle_id='2026-09-04',
            expected_intent_request_digest=kw['expected_first_intent_request_digest'],
            expected_settlement_request_digest=previous.request_digest)])
    for index in (2, 3):
        fixture = (args, kw, previous, prepared, tmp)
        published = publish_intent(fixture)
        settled, _ = settle(fixture, published, monkeypatch)
        request['requested_cycles'].append(dict(cycle_index=index, current_cycle_id=args[4],
            expected_intent_request_digest=published.request_digest, expected_settlement_request_digest=settled.request_digest))
        if index == 2:
            previous = settled
            args = _cycle(args, '2026-09-18', prepared[2], prepared[3], monkeypatch)
            kw = dict(kw, cycle_index=3, predecessor_cycle_index=2,
                predecessor_settlement_store_root=kw['settlement_store_root'], expected_predecessor_request_digest=settled.request_digest,
                decision_deadline_utc='2026-09-21T01:00:00Z', next_open_utc='2026-09-21T01:30:00Z')
            monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 20, tzinfo=timezone.utc), 100.))
    return request


def build(request):
    return m.build_forward_window_report(**{k: v for k, v in request.items() if k != 'schema_version'})


def test_actual_three_periods(three_window):
    report = build(three_window)
    assert report['status'] == 'COMPLETE', [(r['status'], r['reason_codes']) for r in report['rows']]
    assert report['requested_window_chain_verified'] and report['verified_prefix_end_index'] == 3
    assert report['source_forward_records_complete'] and not report['epoch_started']
    assert report['rows'][0]['overlap']['holdings']['jaccard'] != '1'
    assert report['rows'][1]['arms'][0]['nav_before'] != report['rows'][0]['arms'][0]['nav_after']
    with localcontext() as context:
        context.prec = 128
        base = Decimal(report['basis']['value'])
        for index, row in enumerate(report['rows'], 1):
            root = three_window['first_settlement_store_root' if index == 1 else 'continuing_settlement_store_root']
            reader = m.d1.inspect_first_forward_settlement if index == 1 else m.d2.inspect_next_forward_settlement
            args = (root, three_window['epoch_id']) if index == 1 else (root, three_window['epoch_id'], index)
            source = reader(*args, expected_request_digest=row['expected_settlement_request_digest'])
            transitions = m._doc(source, 'settlements.json')['roles']
            assert row['settlement']['reference'] == m._ref(source)
            for arm_index, (arm, saved) in enumerate(zip(row['arms'], transitions)):
                transition = saved['transition']
                for key in ('nav_before', 'nav_after', 'total_fee', 'slippage_cost', 'gross_buy', 'gross_sell', 'filled_count', 'no_fill_count'):
                    assert arm[key] == transition[key]
                assert arm['positions'] == source.after_states[arm_index].after_state.to_dict()['positions']
                assert arm['normalized_nav'] == m._ratio(Decimal(arm['nav_after']) / base)
                denominator = base if index == 1 else Decimal(report['rows'][index - 2]['arms'][arm_index]['nav_after'])
                assert arm['period_return'] == m._ratio(Decimal(arm['nav_after']) / denominator - 1)
            if index > 1:
                previous = report['rows'][index - 2]
                pred = row['source_context']['continuation']['predecessor']
                assert pred['intent'] == previous['intent']['reference']
                assert pred['settlement'] == previous['settlement']['reference']
            assert row['joins'] == dict(source=True, predecessor=True, frozen=True)
    again = build(three_window)
    assert report['semantic_digest'] == again['semantic_digest']


@pytest.mark.parametrize('kind', ['unbound', 'gap', 'tail', 'date', 'digest'])
def test_requested_rows_preserved(three_window, kind):
    request = copy.deepcopy(three_window)
    if kind == 'unbound':
        request['requested_cycles'][1]['expected_intent_request_digest'] = None
    elif kind == 'gap':
        import shutil
        shutil.rmtree(m.Path(request['continuing_settlement_store_root']) / request['epoch_id'] / '2')
    elif kind == 'tail':
        request['requested_cycles'].append(dict(cycle_index=4, current_cycle_id='2026-09-25', expected_intent_request_digest=None, expected_settlement_request_digest=None))
    elif kind == 'date':
        request['requested_cycles'][1]['current_cycle_id'] = '2026-09-12'
    else:
        request['requested_cycles'][1]['expected_intent_request_digest'] = 'a' * 64
    report = build(request)
    assert report['status'] != 'COMPLETE'
    assert len(report['rows']) == len(request['requested_cycles'])
    assert report['metrics']['return_difference'] is None
    assert report['metrics']['arms'][0]['window_return'] is None
    assert report['rows'][2]['settlement']['status'] == 'VERIFIED'
    assert not report['requested_window_chain_verified']
    if kind == 'unbound':
        assert report['rows'][1]['intent']['status'] == 'UNBOUND_PRESENT'
    if kind == 'tail':
        assert report['rows'][-1]['status'] == 'UNSEALED'


def test_defensive_export(three_window):
    r = three_window
    observed = m.c4.inspect_first_forward_intent_pair(r['first_intent_store_root'], r['epoch_id'],
        expected_request_digest=r['requested_cycles'][0]['expected_intent_request_digest'])
    data = observed.verified_report_data
    data['request.json'] = 'bad'
    assert observed.verified_report_data['request.json'] != 'bad'
    settled = m.d1.inspect_first_forward_settlement(r['first_settlement_store_root'], r['epoch_id'],
        expected_request_digest=r['requested_cycles'][0]['expected_settlement_request_digest'])
    data = settled.verified_report_data
    data.clear()
    assert settled.verified_report_data['settlements.json']


@pytest.mark.parametrize('change', ['extra', 'bool_index', 'float_index', 'duplicate_date', 'empty', 'epoch', 'digest'])
def test_request_exact_schema(change, tmp_path):
    from tests.quantpits.scripts.test_report_forward_window import empty_request
    request = empty_request(tmp_path)
    if change == 'extra':
        request['extra'] = True
    elif change == 'bool_index':
        request['requested_cycles'][0]['cycle_index'] = True
    elif change == 'float_index':
        request['requested_cycles'][0]['cycle_index'] = 1.0
    elif change == 'duplicate_date':
        request['requested_cycles'].append(dict(request['requested_cycles'][0], cycle_index=2))
    elif change == 'empty':
        request['requested_cycles'] = []
    elif change == 'epoch':
        request['epoch_id'] = '../escape'
    else:
        request['requested_cycles'][0]['expected_intent_request_digest'] = False
    with pytest.raises(ValueError, match='REQUEST_INVALID'):
        m.validate_request(request)


def test_partial_middle_preserves_local_points_without_bridging():
    rows = formula_rows(posts=((99, 99), (100, 100), (109, 108), (110, 109)),
                        pres=((100, 100), (101, 101), (110, 109), (111, 110)))
    rows[1]['arms'][0].update(valuation_status='PARTIAL', nav_before=None, nav_after=None)
    with localcontext() as ctx:
        ctx.prec = 64
        _, metrics, _ = m._metrics(rows)
    assert metrics['arms'][1]['window_return'] is None
    assert rows[2]['arms'][0]['normalized_nav'] == '1.09'
    assert rows[2]['arms'][0]['period_return'] is None
    assert rows[2]['arms'][0]['drawdown'] is None
    assert rows[3]['arms'][0]['period_return'] is not None


def test_offline_mutation_failure_and_interrupt(three_window, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('live dependency forbidden')
    monkeypatch.setattr(m.c4, '_prepare_pair', forbidden, raising=False)
    monkeypatch.setattr(m.d1, '_build', forbidden)
    monkeypatch.setattr(m.d2, '_intent_run', forbidden)
    original = m.c4.inspect_first_forward_intent_pair
    def failing(*args, **kwargs):
        raise RuntimeError('private source failure')
    monkeypatch.setattr(m.c4, 'inspect_first_forward_intent_pair', failing)
    report = build(three_window)
    assert report['status'] == 'BLOCKED' and report['rows'][2]['settlement']['status'] == 'VERIFIED'
    monkeypatch.setattr(m.c4, 'inspect_first_forward_intent_pair', original)
    assert build(three_window)['status'] == 'COMPLETE'
    def transient(*args, **kwargs):
        observed = original(*args, **kwargs)
        path = m.Path(args[0]) / args[1] / 'plans.json'
        raw = path.read_bytes()
        path.write_bytes(raw)
        return observed
    monkeypatch.setattr(m.c4, 'inspect_first_forward_intent_pair', transient)
    report = build(three_window)
    assert report['status'] == 'BLOCKED'
    assert all(r['status'] == 'UNCERTAIN' for r in report['rows'])
    assert report['metrics']['arms'][0]['window_return'] is None
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(m.c4, 'inspect_first_forward_intent_pair', interrupt)
    with pytest.raises(KeyboardInterrupt):
        build(three_window)


@pytest.mark.parametrize('target_kind', ['intent', 'settlement'])
def test_actual_operation_join_not_just_request(three_window, target_kind):
    # A valid UUID replacement is individually readable but breaks the actual next-period reference.
    import uuid
    root_key = 'first_' + target_kind + '_store_root'
    path = m.Path(three_window[root_key]) / three_window['epoch_id'] / 'completion.json'
    completion = json.loads(path.read_text())
    completion['operation_id'] = str(uuid.uuid4())
    path.write_bytes(m.c4.canonical(completion))
    report = build(three_window)
    assert report['rows'][0][target_kind]['status'] == 'VERIFIED'
    assert report['rows'][1]['status'] == 'CHAIN_BREAK'
    assert report['status'] == 'BLOCKED' and report['metrics']['return_difference'] is None
    if target_kind == 'intent':
        assert report['rows'][0]['joins']['source'] is False


def test_sealed_tail_and_missing_file(three_window):
    import shutil
    root = m.Path(three_window['continuing_settlement_store_root']) / three_window['epoch_id']
    shutil.rmtree(root / '3')
    report = build(three_window)
    assert report['status'] == 'PARTIAL'
    assert report['rows'][2]['status'] == 'SEALED_UNSETTLED'
    assert report['rows'][2]['original_success'] == 'original_success_unchecked'
    assert report['rows'][2]['overlap']['buy'] is not None
    (root / '2' / 'settlements.json').unlink()
    report = build(three_window)
    assert report['rows'][1]['status'] == 'INCOMPLETE'
    assert report['rows'][2]['intent']['status'] == 'VERIFIED'


from tests.quantpits.research.test_forward_settlement import settlement, publish as publish_settlement


def first_request(kw, settled, tmp):
    roots = []
    for name in ('unused_intents', 'unused_settlements'):
        root = tmp / name
        root.mkdir(mode=0o700)
        roots.append(str(root))
    return dict(schema_version=1, first_intent_store_root=str(kw['intent_store_root']),
        first_settlement_store_root=str(kw['settlement_store_root']),
        continuing_intent_store_root=roots[0], continuing_settlement_store_root=roots[1], epoch_id=kw['epoch_id'],
        requested_cycles=[dict(cycle_index=1, current_cycle_id='2026-09-04',
            expected_intent_request_digest=kw['expected_intent_request_digest'], expected_settlement_request_digest=settled.request_digest)])


def test_actual_partial_nav_and_offline_source_removal(settlement, tmp_path):
    import shutil
    intent = m.c4.inspect_first_forward_intent_pair(settlement['intent_store_root'], settlement['epoch_id'],
        expected_request_digest=settlement['expected_intent_request_digest'])
    instrument = intent.d1_inputs[0].prior.positions[0].instrument
    (settlement['qlib_provider_root'] / 'features' / instrument.lower() / 'open.day.bin').unlink()
    result = publish_settlement(settlement)
    request = first_request(settlement, result, tmp_path)
    shutil.rmtree(settlement['qlib_provider_root'])
    settlement['publication_success_record_path'].unlink()
    report = build(request)
    assert report['status'] == 'PARTIAL' and report['requested_window_chain_verified']
    assert report['source_forward_records_complete']
    assert report['basis']['value'] is None
    assert report['metrics']['arms'][0]['window_return'] is None
    assert any(a['valuation_status'] == 'PARTIAL' for a in report['rows'][0]['arms'])
    assert 'PARTIAL_NAV' in m.render_markdown(report)


def test_actual_zero_order_pair(publication, tmp_path, monkeypatch):
    from tests.quantpits.research.test_forward_intent_publication import _publish
    import struct
    args, kw = publication
    for path in (args[3] / 'features').glob('*/close.day.bin'):
        path.unlink()
    intent = _publish(publication)
    assert all(not arm.intents.intents for arm in intent.d1_inputs)
    success = tmp_path / 'zero_success.json'
    success.write_bytes(m.c4.canonical(intent.to_safe_summary_dict()))
    success.chmod(0o600)
    root = tmp_path / 'zero_settlements'
    root.mkdir(mode=0o700)
    (args[3] / 'calendars/day.txt').write_bytes(b'2026-08-28\n2026-09-04\n2026-09-07\n')
    for directory in (args[3] / 'features').iterdir():
        for field, value in [('open', 10.), ('factor', 1.)]:
            (directory / (field + '.day.bin')).write_bytes(struct.pack('<ffff', 0., value, value, value))
    monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 7, 2, tzinfo=timezone.utc))
    params = dict(intent_store_root=kw['intent_store_root'], epoch_id=kw['epoch_id'],
        expected_intent_request_digest=intent.request_digest, publication_success_record_path=success,
        qlib_provider_root=args[3], settlement_store_root=root)
    result = publish_settlement(params)
    report = build(first_request(params, result, tmp_path))
    assert report['status'] == 'COMPLETE'
    assert report['metrics']['return_difference'] == '0'
    for arm in report['rows'][0]['arms']:
        assert arm['filled_count'] == 0 and arm['no_fill_count'] == 0 and arm['cost'] == '0'
    assert report['rows'][0]['overlap']['buy']['empty_both']
    assert report['rows'][0]['overlap']['sell']['empty_both']


def test_changed_manifest_cannot_keep_request_authority(settlement, tmp_path):
    result = publish_settlement(settlement)
    request = first_request(settlement, result, tmp_path)
    target = settlement['intent_store_root'] / settlement['epoch_id']
    path = target / 'manifest.json'
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2))
    completion = json.loads((target / 'completion.json').read_text())
    completion['manifest_digest'] = m.c4._raw(path.read_bytes())
    (target / 'completion.json').write_bytes(m.c4.canonical(completion))
    report = build(request)
    # Canonical serialization is itself part of the reader contract. A byte-distinct
    # manifest cannot become a new valid identity merely by updating completion.
    assert report['rows'][0]['intent']['status'] != 'VERIFIED'
    assert report['rows'][0]['settlement']['status'] == 'VERIFIED'
    assert report['rows'][0]['joins']['source'] is not True
    assert report['status'] == 'BLOCKED'


def test_physical_store_binding_cannot_be_spliced(settlement, tmp_path):
    import shutil
    result = publish_settlement(settlement)
    request = first_request(settlement, result, tmp_path)
    copied = tmp_path / 'copied_intents'
    shutil.copytree(settlement['intent_store_root'], copied)
    request['first_intent_store_root'] = str(copied)
    report = build(request)
    assert report['status'] == 'BLOCKED'
    assert report['rows'][0]['intent']['status'] != 'VERIFIED'
    assert report['rows'][0]['settlement']['status'] == 'VERIFIED'
    assert report['metrics']['arms'][0]['window_return'] is None


@pytest.mark.parametrize('failure', [None, RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit])
def test_guard_close_attempts_all_and_preserves_interruption(tmp_path, monkeypatch, failure):
    from tests.quantpits.scripts.test_report_forward_window import empty_request
    request = empty_request(tmp_path)
    original = m.SourceMutationObserver.close
    closed = []
    def close(guard):
        closed.append(guard)
        original(guard)
        if len(closed) == 1:
            raise OSError('/private/guard-close-secret')
    monkeypatch.setattr(m.SourceMutationObserver, 'close', close)
    interruption = failure('original failure') if failure else None
    if failure:
        def diagnostics(*args):
            raise interruption
        monkeypatch.setattr(m, '_diagnostics', diagnostics)
    if failure in (KeyboardInterrupt, SystemExit, GeneratorExit):
        with pytest.raises(failure) as caught:
            build(request)
        assert caught.value is interruption
    else:
        report = build(request)
        assert report['status'] == 'BLOCKED'
        assert len(report['rows']) == report['requested_count'] == 1
        assert report['rows'][0]['status'] == 'UNCERTAIN'
        assert 'SOURCE_GUARD_CLOSE_FAILED' in report['reason_codes']
        if failure:
            assert 'REPORT_PAYLOAD_INVALID' in report['reason_codes']
        assert not report['requested_window_chain_verified']
        assert not report['source_forward_records_complete']
        assert report['metrics']['return_difference'] is None
        assert 'secret' not in json.dumps(m.safe_summary(report))
    assert len(closed) == 2 and len({id(g) for g in closed}) == 2
    assert all(g.fd == -1 for g in closed)
