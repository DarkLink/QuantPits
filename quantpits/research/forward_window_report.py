"""Read-only, explicitly bound common-window reports for forward pairs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from pathlib import Path

from quantpits.evidence.inspection import SourceMutationObserver, strict_json_object
from quantpits.research import forward_intent_publication as c4
from quantpits.research import forward_settlement as d1
from quantpits.research import forward_continuation as d2

RULE = 'FIRST_NEXT_OPEN_PRETRADE_BASE_TO_POSTTRADE_SAMPLES_V1'
ROLES = ('CHAMPION', 'CHALLENGER')
ROOTS = ('first_intent_store_root', 'first_settlement_store_root',
         'continuing_intent_store_root', 'continuing_settlement_store_root')
CYCLE_KEYS = ('cycle_index', 'current_cycle_id', 'expected_intent_request_digest',
              'expected_settlement_request_digest')
ARM_VALUES = ('portfolio_id valuation_status nav_before nav_after cash_after total_fee slippage_cost '
              'gross_buy gross_sell filled_count no_fill_count normalized_nav period_return drawdown '
              'cost cost_rate gross_turnover').split()
CSV_FIELDS = ('cycle_index current_cycle_id trade_date next_open_utc status row_reason_codes role'.split()
              + ARM_VALUES + ['reason_codes'])
WARNINGS = ['LOCAL_OBSERVATIONS_NO_EXTERNAL_TRUSTED_TIMESTAMP', 'SYNTHETIC_EXECUTION_ASSUMPTION',
            'CORPORATE_ACTION_UNMODELED', 'NO_EXTERNAL_CASHFLOW', 'DISCRETE_NEXT_OPEN_DRAWDOWN',
            'NAV_ALREADY_INCLUDES_COST', 'BOOTSTRAP_TO_FIRST_OPEN_EXCLUDED']


def _hash(value):
    return hashlib.sha256(c4.canonical(value)).hexdigest()


def parse_request(data):
    value = strict_json_object(data)
    if value is None:
        raise ValueError('REQUEST_INVALID')
    return validate_request(value)


def validate_request(request):
    if type(request) is not dict or set(request) != set(ROOTS + ('schema_version', 'epoch_id', 'requested_cycles')):
        raise ValueError('REQUEST_INVALID')
    if type(request['schema_version']) is not int or request['schema_version'] != 1:
        raise ValueError('REQUEST_INVALID')
    for key in ROOTS:
        if type(request[key]) is not str or not request[key] or '\x00' in request[key]:
            raise ValueError('REQUEST_INVALID')
    try:
        c4.c3.surface._identifier(request['epoch_id'], 'epoch_id')
        rows = request['requested_cycles']
        if type(rows) is not list or not rows:
            raise ValueError()
        previous = ''
        for index, row in enumerate(rows, 1):
            if type(row) is not dict or set(row) != set(CYCLE_KEYS):
                raise ValueError()
            if type(row['cycle_index']) is not int or row['cycle_index'] != index:
                raise ValueError()
            date = row['current_cycle_id']
            c4.c3.surface._date(date, 'current_cycle_id')
            if date <= previous:
                raise ValueError()
            previous = date
            for key in CYCLE_KEYS[2:]:
                if row[key] is not None:
                    c4._digest(row[key])
    except c4._PROCESS_CONTROL:
        raise
    except Exception:
        raise ValueError('REQUEST_INVALID') from None
    return json.loads(json.dumps(request))


def _text(value):
    if value is None:
        return None
    value = Decimal(value)
    if value == 0:
        return '0'
    return format(value, 'f').rstrip('0').rstrip('.') if '.' in format(value, 'f') else format(value, 'f')


def _ratio(value):
    return None if value is None else _text(value.quantize(Decimal('1e-24'), rounding=ROUND_HALF_EVEN))


def _overlap(left, right):
    left, right = set(left), set(right)
    return dict(intersection_count=len(left & right), union_count=len(left | right),
                jaccard=_ratio(Decimal(len(left & right)) / len(left | right)) if left | right else '1',
                empty_both=not bool(left | right))


def _ref(observation):
    safe = observation.to_safe_summary_dict()
    return {key: safe[key] for key in ('request_digest', 'manifest_digest', 'operation_id')}


def _empty_arm(role):
    return dict(dict.fromkeys(ARM_VALUES), role=role, reason_codes=['SETTLEMENT_UNAVAILABLE'],
                positions=None, price_provenance=None)


def _observe(root, epoch, index, expected, kind, guards):
    summary = dict(status='MISSING', reason_codes=[], reference=None, raw_digests=None)
    try:
        # Watch exact requested slots, including currently absent slots, before inspect.
        root = Path(root).absolute()
        c4._directory(root)
        logical = epoch if index == 1 else epoch + '/' + str(index)
        guard = SourceMutationObserver(root, ())
        guards.append(guard)
        guard.add_paths((logical,))
        if not guard.supported:
            raise ValueError('OBSERVER_UNSUPPORTED')
        target = root / logical
        if not os.path.lexists(str(target)):
            return summary, None
        if expected is None:
            summary.update(status='UNBOUND_PRESENT', reason_codes=['EXPECTED_IDENTITY_UNBOUND'])
            return summary, None
        reader = ((c4.inspect_first_forward_intent_pair if kind == 'intent' else d1.inspect_first_forward_settlement)
                  if index == 1 else (d2.inspect_next_forward_intent_pair if kind == 'intent' else d2.inspect_next_forward_settlement))
        args = (root, epoch) if index == 1 else (root, epoch, index)
        observation = reader(*args, expected_request_digest=expected)
        safe = observation.to_safe_summary_dict()
        summary.update(status=safe['status'], reason_codes=safe['reason_codes'])
        if safe['status'] == 'VERIFIED':
            raw = observation.verified_report_data
            summary.update(reference=_ref(observation), raw_digests={k: hashlib.sha256(v.encode()).hexdigest() for k, v in raw.items()})
            return summary, observation
        return summary, None
    except c4._PROCESS_CONTROL:
        raise
    except Exception:
        summary.update(status='UNCERTAIN', reason_codes=['SOURCE_OBSERVATION_FAILED'])
        return summary, None


def _doc(observation, name):
    return json.loads(observation.verified_report_data[name])


def _intent_body(observation):
    return _doc(observation, 'request.json')['body']


def _join(row, intent, settlement, previous, first):
    body = _intent_body(intent)
    row['source_context'] = dict(time_policy=body['time_policy'], input_provenance=body['input_provenance'],
        continuation=body.get('continuation'), execution_assumption=intent.d1_inputs[0].execution_assumption.to_dict(),
        price_observation=None)
    row.update(date_verified=body['current_cycle_id'] == row['current_cycle_id'],
               trade_date=body['trade_date'], next_open_utc=body['time_policy']['next_open_utc'])
    if not row['date_verified']:
        row['reason_codes'].append('REQUEST_CYCLE_MISMATCH')
    if settlement is None:
        return
    source = _doc(settlement, 'source_intent.json')
    source_body = json.loads(source['metadata']['request.json'])['body']
    completion = json.loads(source['metadata']['completion.json'])
    transitions = _doc(settlement, 'settlements.json')['roles']
    source_ok = (source_body == body and completion == _doc(intent, 'completion.json')
                 and {k: completion[k] for k in _ref(intent)} == _ref(intent)
                 and all(t['role'] == i.role and t['transition']['prior_state'] == i.prior.to_dict()
                         and t['transition']['requested_intents'] == i.intents.to_dict()
                         and t['transition']['assumption'] == i.execution_assumption.to_dict()
                         for t, i in zip(transitions, intent.d1_inputs)))
    row['joins']['source'] = source_ok
    row['original_success'] = 'ORIGINAL_SUCCESS_VERIFIED' if source_ok else 'SOURCE_MISMATCH'
    if not source_ok:
        row['reason_codes'].append('SOURCE_INTENT_JOIN_INVALID')
    if row['cycle_index'] == 1:
        row['joins'].update(predecessor=True, frozen=True)
    else:
        continuation = body['continuation']
        pred = continuation['predecessor']
        pi, ps = previous
        predecessor_ok = (pi is not None and ps is not None and pred['intent'] == _ref(pi)
                          and pred['settlement'] == _ref(ps)
                          and pred['current_cycle_id'] == _intent_body(pi)['current_cycle_id']
                          and pred['trade_date'] == _intent_body(pi)['trade_date']
                          and all(i.role == s.role and i.prior.to_dict() == s.after_state.to_dict()
                                  for i, s in zip(intent.d1_inputs, ps.after_states)))
        frozen_ok = False
        if first is not None:
            initial = _intent_body(first)
            selectors = initial['input_provenance']['selectors']
            frozen = {k: v for k, v in selectors.items() if k not in ('current_cycle_id', 'signal_capsule_id')}
            frozen_ok = (continuation['first_intent'] == _ref(first)
                         and continuation['frozen_selectors'] == frozen
                         and continuation['definition_request_digest'] == initial['input_provenance']['definition']
                         and continuation['schedule']['first_anchor'] == initial['current_cycle_id']
                         and body['time_policy']['market_timezone'] == initial['time_policy']['market_timezone'])
            if pi is not None and row['cycle_index'] > 2:
                frozen_ok = frozen_ok and continuation['schedule'] == _intent_body(pi)['continuation']['schedule']
        row['joins'].update(predecessor=bool(predecessor_ok), frozen=bool(frozen_ok))
        if not predecessor_ok:
            row['reason_codes'].append('PREDECESSOR_JOIN_INVALID')
        if not frozen_ok:
            row['reason_codes'].append('FROZEN_JOIN_INVALID')
    if row['reason_codes']:
        row['status'] = 'CHAIN_BREAK'


def _diagnostics(row, intent, settlement):
    if intent is not None:
        data = intent.verified_report_data
        plans = json.loads(data['plans.json'])['roles']
        k = plans[0]['plan']['definition']['topk']
        rankings = [c4.c3.replay._ranking_from_csv(data[r.lower() + '_ranking.csv'].encode()) for r in ROLES]
        batches = intent.d1_inputs
        row['overlap'] = dict(ranking_top_k=dict(_overlap(
            [r['instrument'] for r in rankings[0].rows[:k]], [r['instrument'] for r in rankings[1].rows[:k]]), k=k),
            buy=_overlap(*[[i.instrument for i in b.intents.intents if i.side == 'BUY'] for b in batches]),
            sell=_overlap(*[[i.instrument for i in b.intents.intents if i.side == 'SELL'] for b in batches]),
            holdings=None, holding_quantities=None)
    if settlement is None:
        return
    transitions = _doc(settlement, 'settlements.json')['roles']
    if row['source_context'] is not None:
        row['source_context']['price_observation'] = _doc(settlement, 'next_open_prices.json')
    with localcontext() as context:
        context.prec = max(128, 4 * max(len(str(entry['transition'][k])) for entry in transitions
            for k in ('nav_before', 'nav_after', 'gross_buy', 'gross_sell', 'total_fee', 'slippage_cost')) + 64)
        for arm, entry in zip(row['arms'], transitions):
            t = entry['transition']
            arm.update({k: t[k] for k in ('valuation_status nav_before nav_after total_fee slippage_cost gross_buy gross_sell filled_count no_fill_count').split()})
            arm.update(portfolio_id=t['after_state']['portfolio_id'], cash_after=t['after_state']['cash'],
                       positions=t['after_state']['positions'], price_provenance=t['quotes'], reason_codes=[])
            arm['cost'] = _text(Decimal(t['total_fee']) + Decimal(t['slippage_cost']))
            if t['valuation_status'] != 'COMPLETE':
                arm['reason_codes'].append('PARTIAL_NAV')
            if t['nav_after'] is not None and Decimal(t['nav_after']) <= 0:
                arm['reason_codes'].append('NONPOSITIVE_NAV')
            pre = None if t['nav_before'] is None else Decimal(t['nav_before'])
            arm['gross_turnover'] = _ratio((Decimal(t['gross_buy']) + Decimal(t['gross_sell'])) / pre) if pre is not None and pre > 0 else None
            if arm['gross_turnover'] is None:
                arm['reason_codes'].append('TURNOVER_DENOMINATOR_INVALID')
    if row['overlap'] is not None:
        holdings = [{p['instrument']: p['quantity'] for p in a['positions']} for a in row['arms']]
        row['overlap']['holdings'] = _overlap(*holdings)
        row['overlap']['holding_quantities'] = [dict(instrument=i, champion=holdings[0].get(i, 0), challenger=holdings[1].get(i, 0)) for i in sorted(set(holdings[0]) | set(holdings[1]))]


def _metrics(rows):
    """Pure Decimal formulas; rows are internal observations, never authority inputs."""
    basis = dict(rule=RULE, status='INVALID', value=None, normalized_start=None, reason_codes=[])
    initial = rows[0]
    values = [a['nav_before'] for a in initial['arms']]
    if (initial['status'] == 'SETTLED' and all(a['valuation_status'] == 'COMPLETE' for a in initial['arms'])
            and None not in values and Decimal(values[0]) == Decimal(values[1]) and Decimal(values[0]) > 0):
        base = Decimal(values[0])
        basis.update(status='VALID', value=_text(base), normalized_start='1')
    else:
        base = None
        basis['reason_codes'].append('COMMON_BASE_INVALID')
    full_chain = all(r['status'] == 'SETTLED' for r in rows)
    full_nav = all(a['valuation_status'] == 'COMPLETE' for r in rows for a in r['arms'])
    reasons = ([] if full_chain else ['REQUESTED_WINDOW_INCOMPLETE']) + ([] if full_nav else ['WINDOW_NAV_INCOMPLETE']) + basis['reason_codes']
    complete = not reasons
    metrics = dict(arms=[], return_difference=None)
    exact = [[{} for _ in ROLES] for _ in rows]
    for index, role in enumerate(ROLES):
        previous, peak, minimum, connected = base, Decimal(1), Decimal(0), True
        samples_complete = True
        total_cost, turnover = Decimal(0), Decimal(0)
        costs_ok, turnover_ok = full_chain, full_chain
        last = None
        for row_index, row in enumerate(rows):
            arm = row['arms'][index]
            connected = connected and row['status'] == 'SETTLED'
            post = None if arm['nav_after'] is None or arm['valuation_status'] != 'COMPLETE' else Decimal(arm['nav_after'])
            if base is not None and connected and post is not None:
                normalized = post / base
                peak = max(peak, normalized)
                drawdown = normalized / peak - 1
                minimum = min(minimum, drawdown)
                arm.update(normalized_nav=_ratio(normalized), drawdown=_ratio(drawdown) if samples_complete else None)
                if not samples_complete:
                    arm['reason_codes'].append('DRAWDOWN_HISTORY_INCOMPLETE')
                exact[row_index][index]['normalized_nav'] = normalized
                if previous is not None and previous > 0:
                    arm['period_return'] = _ratio(post / previous - 1)
                    exact[row_index][index]['period_return'] = post / previous - 1
                else:
                    arm['reason_codes'].append('ADJACENT_DENOMINATOR_INVALID')
                last = post / base - 1
            else:
                arm['reason_codes'].append('NORMALIZATION_UNAVAILABLE')
            previous = post if connected else None
            # A partial valuation breaks the drawdown sample history, but not the state chain.
            if post is None:
                samples_complete = False
            if arm['cost'] is None:
                costs_ok = False
            else:
                total_cost += Decimal(arm['cost'])
                arm['cost_rate'] = _ratio(Decimal(arm['cost']) / base) if base is not None else None
            pre = None if arm['nav_before'] is None else Decimal(arm['nav_before'])
            if pre is None or pre <= 0:
                turnover_ok = False
            else:
                turnover += (Decimal(arm['gross_buy']) + Decimal(arm['gross_sell'])) / pre
        metrics['arms'].append(dict(role=role, window_return=_ratio(last) if complete else None,
            max_drawdown=_ratio(minimum) if complete else None,
            total_cost=_text(total_cost) if costs_ok else None,
            cost_rate=_ratio(total_cost / base) if costs_ok and base is not None else None,
            cumulative_gross_turnover=_ratio(turnover) if turnover_ok else None,
            reason_codes=list(dict.fromkeys(reasons + ([] if costs_ok else ['COST_COVERAGE_INCOMPLETE'])
                + ([] if turnover_ok else ['TURNOVER_COVERAGE_OR_DENOMINATOR_INVALID'])))))
    if complete:
        metrics['return_difference'] = _ratio((Decimal(rows[-1]['arms'][1]['nav_after']) - Decimal(rows[-1]['arms'][0]['nav_after'])) / base)
    for row_index, row in enumerate(rows):
        for key, source in (('period_return_difference', 'period_return'), ('normalized_difference', 'normalized_nav')):
            a, b = [arm.get(source) for arm in exact[row_index]]
            row[key] = _ratio(b - a) if a is not None and b is not None else None
    return basis, metrics, reasons


def build_forward_window_report(first_intent_store_root, first_settlement_store_root,
                                continuing_intent_store_root, continuing_settlement_store_root,
                                epoch_id, requested_cycles):
    request = dict(zip(ROOTS, map(str, (first_intent_store_root, first_settlement_store_root,
                                      continuing_intent_store_root, continuing_settlement_store_root))))
    request.update(schema_version=1, epoch_id=epoch_id, requested_cycles=requested_cycles)
    request = validate_request(request)
    with localcontext() as context:
        context.prec = 128
        return _build(request)


def _build(request):
    guards, rows, observations = [], [], []
    try:
        for requested in request['requested_cycles']:
            index = requested['cycle_index']
            row = dict(requested, status='UNSEALED', reason_codes=[], intent=None, settlement=None,
                joins=dict(source=None, predecessor=None, frozen=None), date_verified=False,
                trade_date=None, next_open_utc=None, original_success='original_success_unchecked',
                arms=[_empty_arm(r) for r in ROLES], overlap=None, source_context=None,
                period_return_difference=None, normalized_difference=None)
            rows.append(row)
            pair = []
            for kind, rootkey in zip(('intent', 'settlement'), ROOTS[:2] if index == 1 else ROOTS[2:]):
                safe, observed = _observe(request[rootkey], request['epoch_id'], index,
                    requested['expected_' + kind + '_request_digest'], kind, guards)
                row[kind] = safe
                pair.append(observed)
            intent, settlement = pair
            observations.append(pair)
            statuses = [row[k]['status'] for k in ('intent', 'settlement')]
            for status in ('UNCERTAIN', 'CONFLICT', 'UNBOUND_PRESENT', 'INCOMPLETE'):
                if status in statuses:
                    row['status'] = status
                    break
            else:
                row['status'] = ('SETTLED' if intent and settlement else 'SEALED_UNSETTLED' if intent else
                                 'CHAIN_BREAK' if settlement else 'UNSEALED')
            row['reason_codes'] = list(dict.fromkeys(code for k in ('intent', 'settlement') for code in row[k]['reason_codes']))
            try:
                if intent:
                    _join(row, intent, settlement, observations[-2] if index > 1 else (None, None), observations[0][0])
                    if not row['date_verified']:
                        row['status'] = 'CHAIN_BREAK'
                if settlement and not intent:
                    row['reason_codes'].append('ACTUAL_INTENT_UNAVAILABLE')
                _diagnostics(row, intent, settlement)
            except c4._PROCESS_CONTROL:
                raise
            except Exception:
                row['status'] = 'CONFLICT'
                row['reason_codes'].append('REPORT_PAYLOAD_INVALID')
        if any(not guard.supported or guard.mutated() for guard in guards):
            for row in rows:
                row['status'] = 'UNCERTAIN'
                row['reason_codes'].append('SOURCE_CHANGED_DURING_WINDOW')
    finally:
        for guard in guards:
            guard.close()
    # Use sufficient precision for the persisted accounting values and ratio quantization.
    with localcontext() as context:
        context.prec = max(128, 4 * max(len(str(a[k])) for r in rows for a in r['arms']
            for k in ('nav_before', 'nav_after', 'gross_buy', 'gross_sell', 'cost')) + 64)
        basis, metrics, reasons = _metrics(rows)
    prefix = 0
    for row in rows:
        if row['status'] != 'SETTLED':
            break
        prefix += 1
    blocked = any(row['status'] in ('CONFLICT', 'UNCERTAIN', 'CHAIN_BREAK') for row in rows)
    report = dict(schema_version=1, rule=RULE, status='BLOCKED' if blocked else 'PARTIAL' if reasons else 'COMPLETE',
        request_digest=_hash({k: v for k, v in request.items() if k not in ROOTS}),
        implementation={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (Path(__file__), Path(c4.__file__), Path(d1.__file__), Path(d2.__file__))},
        report_generated_at=datetime.now(timezone.utc).isoformat(), private_bindings={k: request[k] for k in ROOTS},
        epoch_id=request['epoch_id'], requested_count=len(rows), settled_count=sum(r['status'] == 'SETTLED' for r in rows),
        verified_prefix_end_index=prefix, requested_window_chain_verified=prefix == len(rows),
        source_forward_records_complete=prefix == len(rows), prospective_claim=False, epoch_started=False,
        promotion_capability=False, time_evidence_scope='LOCAL_OBSERVATIONS_ONLY',
        reason_codes=list(dict.fromkeys(reasons + [c for r in rows for c in r['reason_codes']])), basis=basis,
        window=dict(start=rows[0]['next_open_utc'], end=rows[-1]['next_open_utc'], sampling='PRETRADE_BASE_THEN_POSTTRADE_NEXT_OPEN'),
        metrics=metrics, rows=rows, warnings=WARNINGS[:])
    report['semantic_digest'] = _hash({k: v for k, v in report.items() if k not in ('private_bindings', 'report_generated_at')})
    return report


def safe_summary(report):
    return {key: report[key] for key in ('status', 'requested_count', 'settled_count', 'verified_prefix_end_index',
        'requested_window_chain_verified', 'source_forward_records_complete', 'prospective_claim',
        'epoch_started', 'promotion_capability', 'reason_codes')}


def render_csv(report):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator='\n')
    writer.writeheader()
    for row in report['rows']:
        for arm in row['arms']:
            value = {k: row[k] for k in CSV_FIELDS[:5]}
            value.update(row_reason_codes='|'.join(row['reason_codes']), role=arm['role'],
                         reason_codes='|'.join(arm['reason_codes']))
            value.update({k: arm[k] for k in ARM_VALUES})
            writer.writerow(value)
    return stream.getvalue()


def render_markdown(report):
    def cell(value):
        return 'null' if value is None else str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['# Private forward common-window report', '',
        'Status: %s; requested: %s; settled: %s; verified prefix: %s.' % tuple(report[k] for k in
            ('status', 'requested_count', 'settled_count', 'verified_prefix_end_index')),
        'Epoch: ' + cell(report['epoch_id']), 'Rule: ' + RULE,
        'Window: %s → %s; first next-open pretrade to final next-open posttrade.' % (cell(report['window']['start']), cell(report['window']['end'])),
        'Common base: %s; normalized start: %s.' % (cell(report['basis']['value']), cell(report['basis']['normalized_start'])),
        'Reasons: ' + ', '.join(report['reason_codes']), '',
        '| Role | Window return | Sampled max drawdown | Total cost | Cost rate | Gross turnover |',
        '| --- | --- | --- | --- | --- | --- |']
    for arm in report['metrics']['arms']:
        lines.append('| ' + ' | '.join(cell(arm[k]) for k in ('role', 'window_return', 'max_drawdown', 'total_cost', 'cost_rate', 'cumulative_gross_turnover')) + ' |')
    lines += ['', 'Challenger − champion return: ' + cell(report['metrics']['return_difference']), '',
              '| Index | Cycle | Trade | Next open UTC | Status | Role | NAV before | NAV after | Normalized | Period return | Drawdown | Cost | Reasons |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in report['rows']:
        for arm in row['arms']:
            values = [row[k] for k in ('cycle_index', 'current_cycle_id', 'trade_date', 'next_open_utc', 'status')]
            values += [arm[k] for k in ('role', 'nav_before', 'nav_after', 'normalized_nav', 'period_return', 'drawdown', 'cost')]
            values += [', '.join(row['reason_codes'] + arm['reason_codes'])]
            lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    lines += ['', *report['warnings'], 'Original records only support local observations; this report grants no prospective, epoch-start or promotion capability.', '']
    return '\n'.join(lines)


class ReportOutputError(ValueError):
    def __init__(self, written, attempted):
        super().__init__('OUTPUT_FAILED')
        self.written_files = list(written)
        self.attempted_files = list(attempted)


def write_report(report, output_dir):
    """Create private derived files; no overwrite, no rollback of uncertain output."""
    written, attempted = [], []
    fd = None
    try:
        target = Path(output_dir).absolute()
        for part in (target,) + tuple(target.parents):
            if part.is_symlink():
                raise ValueError()
        parent = target.parent.resolve(strict=True)
        target = parent / target.name
        for source in report['private_bindings'].values():
            source = Path(source).resolve()
            if target == source or source in target.parents or target in source.parents:
                raise ValueError()
        if target.exists():
            c4._directory(target)
            if any(target.iterdir()):
                raise ValueError()
        else:
            target.mkdir(mode=0o700)
        identity = c4._directory(target)
        fd = os.open(str(target), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if (os.fstat(fd).st_dev, os.fstat(fd).st_ino, os.fstat(fd).st_mode) != identity:
            raise ValueError()
        payloads = [('report.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n'),
                    ('cycles.csv', render_csv(report)), ('report.md', render_markdown(report))]
        for name, content in payloads:
            if c4._directory(target) != identity or target.parent.resolve(strict=True) != parent:
                raise ValueError()
            attempted.append(name)
            c4._write_member(fd, name, content.encode('utf-8'))
            raw, _ = c4.c3.surface._read_regular(target / name, maximum=len(content.encode()) + 1, private=True)
            if raw != content.encode('utf-8') or c4._directory(target) != identity:
                raise ValueError()
            written.append(name)
        os.fsync(fd)
        if c4._directory(target) != identity or {p.name for p in target.iterdir()} != set(written):
            raise ValueError()
        for name, content in payloads:
            raw, _ = c4.c3.surface._read_regular(target / name, maximum=len(content.encode()) + 1, private=True)
            if raw != content.encode('utf-8'):
                raise ValueError()
        return written
    except c4._PROCESS_CONTROL:
        raise
    except Exception:
        raise ReportOutputError(written, attempted) from None
    finally:
        if fd is not None:
            os.close(fd)
