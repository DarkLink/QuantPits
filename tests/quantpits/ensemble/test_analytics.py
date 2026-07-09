import json

import pandas as pd
import pytest

from quantpits.ensemble.analytics import (
    CorrelationAnalysisRequest,
    build_model_contribution_payload,
    calculate_loo_contribution,
    compute_prediction_correlation,
    run_correlation_analysis,
    save_model_contribution_snapshot,
    summarize_correlation_matrix,
)


def _prediction_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]), ["AAA"]],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {
            "M1": [1.0, 2.0, 3.0],
            "M2": [2.0, 4.0, 6.0],
            "M3": [3.0, 2.0, 1.0],
        },
        index=index,
    )


def test_compute_prediction_correlation_matches_pandas():
    frame = _prediction_frame()

    result = compute_prediction_correlation(frame)

    pd.testing.assert_frame_equal(result, frame.corr())


def test_run_correlation_analysis_writes_legacy_combo_path(tmp_path):
    frame = _prediction_frame()

    result = run_correlation_analysis(
        CorrelationAnalysisRequest(
            norm_df=frame,
            output_dir=tmp_path,
            anchor_date="2026-01-07",
            combo_name="combo_a",
        ),
        verbose=False,
    )

    assert result.path == tmp_path / "correlation_matrix_combo_a_2026-01-07.csv"
    assert result.path.exists()
    assert result.average == summarize_correlation_matrix(frame.corr())["average"]

    saved = pd.read_csv(result.path, index_col=0)
    pd.testing.assert_frame_equal(saved, frame.corr())


def test_run_correlation_analysis_writes_default_path_without_extra_suffix(tmp_path):
    result = run_correlation_analysis(
        CorrelationAnalysisRequest(
            norm_df=_prediction_frame()[["M1"]],
            output_dir=tmp_path,
            anchor_date="2026-01-07",
        ),
        verbose=False,
    )

    assert result.path == tmp_path / "correlation_matrix_2026-01-07.csv"
    assert result.average is None
    assert result.maximum is None
    assert result.minimum is None


def test_summarize_correlation_matrix_uses_upper_triangle_only():
    corr = pd.DataFrame(
        [[1.0, 0.2, 0.8], [0.2, 1.0, -0.4], [0.8, -0.4, 1.0]],
        index=["M1", "M2", "M3"],
        columns=["M1", "M2", "M3"],
    )

    summary = summarize_correlation_matrix(corr)

    assert summary["average"] == pytest.approx(0.2)
    assert summary["maximum"] == pytest.approx(0.8)
    assert summary["minimum"] == pytest.approx(-0.4)


def test_calculate_loo_contribution_single_model_returns_empty():
    frame = _prediction_frame()[["M1"]]
    final_score = frame["M1"]

    assert calculate_loo_contribution(frame, final_score) == {}


def test_calculate_loo_contribution_preserves_legacy_rounding():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]), ["AAA"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        {
            "M1": [1.0, 2.0, 3.0, 4.0],
            "M2": [1.0, 3.0, 2.0, 5.0],
            "M3": [4.0, 1.0, 3.0, 2.0],
        },
        index=index,
    )
    final_score = pd.Series([1.0, 2.0, 3.0, 4.0], index=frame.index)

    result = calculate_loo_contribution(frame, final_score)

    assert set(result) == {"M1", "M2", "M3"}
    full_ic = float(frame.mean(axis=1).corr(final_score))
    for model_name, metrics in result.items():
        other_models = [model for model in frame.columns if model != model_name]
        loo_ic = float(frame[other_models].mean(axis=1).corr(final_score))
        assert metrics == {
            "loo_ic": round(loo_ic, 6),
            "full_ic": round(full_ic, 6),
            "delta": round(full_ic - loo_ic, 6),
        }


def test_build_model_contribution_payload_preserves_legacy_schema():
    contributions = {"M1": {"loo_ic": 0.1, "full_ic": 0.2, "delta": 0.1}}

    payload = build_model_contribution_payload(
        combo_name=None,
        anchor_date="2026-01-07",
        contributions=contributions,
    )

    assert payload == {
        "combo": "default",
        "anchor_date": "2026-01-07",
        "method": "loo_ic_proxy",
        "contributions": contributions,
    }


def test_save_model_contribution_snapshot_preserves_unicode(tmp_path):
    contributions = {"模型一": {"loo_ic": 0.1, "full_ic": 0.2, "delta": 0.1}}

    result = save_model_contribution_snapshot(
        output_dir=tmp_path,
        anchor_date="2026-01-07",
        combo_name="组合A",
        contributions=contributions,
    )

    assert result.path == tmp_path / "model_contribution_组合A_2026-01-07.json"
    assert result.path.exists()
    raw = result.path.read_text(encoding="utf-8")
    assert "组合A" in raw
    assert "模型一" in raw
    assert "\\u7ec4" not in raw
    assert json.loads(raw) == result.payload
