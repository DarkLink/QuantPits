"""Forward common coverage is narrow; historical replay remains complete-only."""
import io

import pandas as pd
import pytest

from quantpits.research import replay

ANCHOR = '2026-09-18'
UNIVERSE = tuple('SH%06d' % i for i in range(246))


def prediction(ids, values=None, anchor=ANCHOR):
    frame = pd.DataFrame({'score': list(range(len(ids))) if values is None else values,
                          'label': float('nan')}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp(anchor), i) for i in ids], names=('datetime', 'instrument')))
    stream = io.BytesIO()
    frame.to_pickle(stream)
    return stream.getvalue()


def rank(inputs, universe=UNIVERSE):
    return replay.rank_common_anchor(prediction_bytes_by_member=inputs,
        member_order=tuple('abcd'), anchor_date=ANCHOR, eligible_instruments=universe)


def test_246_243_and_complete_parity():
    inputs = {n: prediction(UNIVERSE[:-3]) for n in 'abcd'}
    observed = rank(inputs)
    assert (observed.eligible_count, observed.scored_count, observed.missing_count) == (246, 243, 3)
    assert [r['instrument'] for r in observed.rows if not r['scored']] == list(UNIVERSE[-3:])
    assert all(r['coverage_status'] == 'missing_prediction' for r in observed.rows if not r['scored'])
    with pytest.raises(replay.ReplayInputError):
        replay.rank_complete_anchor(prediction_bytes_by_member=inputs,
            member_order=tuple('abcd'), anchor_date=ANCHOR, eligible_instruments=UNIVERSE)
    inputs = {n: prediction(UNIVERSE) for n in 'abcd'}
    assert rank(inputs).to_csv_bytes() == replay.rank_complete_anchor(
        prediction_bytes_by_member=inputs, member_order=tuple('abcd'),
        anchor_date=ANCHOR, eligible_instruments=UNIVERSE).to_csv_bytes()


@pytest.mark.parametrize('kind,reason', [
    ('different', 'COVERAGE_DIFFERENT'), ('foreign', 'INDEX_FOREIGN'),
    ('anchor', 'ANCHOR_MISSING'), ('duplicate', 'INDEX_DUPLICATE'),
    ('nan', 'SCORE_NONFINITE'), ('inf', 'SCORE_NONFINITE'), ('missing', 'SOURCE_MISSING')])
def test_bad_observations_not_intersected(kind, reason):
    ids = UNIVERSE[:-3]
    inputs = {n: prediction(ids) for n in 'abcd'}
    if kind == 'different':
        inputs['d'] = prediction(ids[:-1])
    elif kind == 'foreign':
        inputs['d'] = prediction(ids + ('FOREIGN',))
    elif kind == 'anchor':
        inputs['d'] = prediction(ids, anchor='2026-09-17')
    elif kind == 'duplicate':
        inputs['d'] = prediction(ids + (ids[0],))
    elif kind == 'missing':
        del inputs['d']
    else:
        inputs['d'] = prediction(ids, [float(kind)] + list(range(len(ids)-1)))
    with pytest.raises(replay.ReplayInputError, match=reason):
        rank(inputs)


@pytest.mark.parametrize('size', [1, 7])
def test_complete_ties_and_member_order_match_legacy(size):
    universe = UNIVERSE[:size]
    inputs = {n: prediction(universe, [(i * (j + 1)) % 3 for i in range(size)])
              for j, n in enumerate('abcd')}
    legacy = replay.rank_complete_anchor(prediction_bytes_by_member=inputs,
        member_order=tuple('abcd'), anchor_date=ANCHOR, eligible_instruments=universe)
    assert rank(inputs, universe).to_csv_bytes() == legacy.to_csv_bytes()
