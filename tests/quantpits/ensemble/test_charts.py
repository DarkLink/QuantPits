from unittest.mock import patch

import pandas as pd


def _report(values=(0.01, 0.02), bench=(0.005, 0.006)):
    return pd.DataFrame(
        {"return": list(values), "bench": list(bench)},
        index=pd.to_datetime(["2026-01-07", "2026-01-14"]),
    )


def test_chart_path_helpers_preserve_legacy_names(tmp_path):
    from quantpits.ensemble.charts import (
        ensemble_nav_chart_path,
        ensemble_weights_chart_path,
    )

    assert (
        ensemble_nav_chart_path(tmp_path, "2026-01-07").name
        == "ensemble_nav_2026-01-07.png"
    )
    assert (
        ensemble_weights_chart_path(tmp_path, "2026-01-07").name
        == "ensemble_weights_2026-01-07.png"
    )
    assert (
        ensemble_nav_chart_path(tmp_path, "2026-01-07", combo_name="combo_a").name
        == "ensemble_nav_combo_a_2026-01-07.png"
    )
    assert (
        ensemble_weights_chart_path(tmp_path, "2026-01-07", combo_name="combo_a").name
        == "ensemble_weights_combo_a_2026-01-07.png"
    )


def test_build_chart_outputs_writes_nav_chart(tmp_path):
    from quantpits.ensemble.charts import ChartGenerationRequest, build_chart_outputs

    report_df = _report()
    result = build_chart_outputs(
        ChartGenerationRequest(
            all_reports={"Ensemble": report_df, "M1": _report((0.0, 0.01))},
            report_df=report_df,
            final_weights=None,
            is_dynamic=False,
            freq="day",
            output_dir=tmp_path,
            anchor_date="2026-01-07",
            combo_name="combo_a",
        ),
        verbose=False,
    )

    assert result.nav_chart_path == tmp_path / "ensemble_nav_combo_a_2026-01-07.png"
    assert result.nav_chart_path.exists()
    assert result.weights_chart_path is None


def test_build_chart_outputs_skips_reports_without_return(tmp_path):
    from quantpits.ensemble.charts import ChartGenerationRequest, build_chart_outputs

    result = build_chart_outputs(
        ChartGenerationRequest(
            all_reports={"Invalid": pd.DataFrame({"bench": [0.0]})},
            report_df=None,
            final_weights=None,
            is_dynamic=False,
            freq="day",
            output_dir=tmp_path,
            anchor_date="2026-01-07",
        ),
        verbose=False,
    )

    assert result.nav_chart_path == tmp_path / "ensemble_nav_2026-01-07.png"
    assert result.nav_chart_path.exists()


def test_build_chart_outputs_writes_dynamic_weights_chart(tmp_path):
    from quantpits.ensemble.charts import ChartGenerationRequest, build_chart_outputs

    final_weights = pd.DataFrame(
        {"M1": [0.6, 0.5], "M2": [0.4, 0.5]},
        index=pd.to_datetime(["2026-01-07", "2026-01-14"]),
    )
    result = build_chart_outputs(
        ChartGenerationRequest(
            all_reports={},
            report_df=None,
            final_weights=final_weights,
            is_dynamic=True,
            freq="day",
            output_dir=tmp_path,
            anchor_date="2026-01-07",
        ),
        verbose=False,
    )

    assert result.nav_chart_path is None
    assert result.weights_chart_path == tmp_path / "ensemble_weights_2026-01-07.png"
    assert result.weights_chart_path.exists()


def test_build_chart_outputs_static_mode_skips_weights_chart(tmp_path):
    from quantpits.ensemble.charts import ChartGenerationRequest, build_chart_outputs

    final_weights = pd.DataFrame({"M1": [1.0]}, index=pd.to_datetime(["2026-01-07"]))
    result = build_chart_outputs(
        ChartGenerationRequest(
            all_reports={},
            report_df=None,
            final_weights=final_weights,
            is_dynamic=False,
            freq="day",
            output_dir=tmp_path,
            anchor_date="2026-01-07",
        ),
        verbose=False,
    )

    assert result.nav_chart_path is None
    assert result.weights_chart_path is None
    assert tmp_path.exists()


def test_script_generate_charts_wrapper_delegates_to_ensemble_module():
    from quantpits.scripts import ensemble_fusion as ef

    with patch("quantpits.ensemble.charts.generate_charts") as mock_generate:
        ef.generate_charts(
            {"Ensemble": _report()},
            _report(),
            None,
            False,
            "day",
            "2026-01-07",
            "output/ensemble",
            combo_name="combo_a",
        )

    mock_generate.assert_called_once()
    assert mock_generate.call_args.kwargs["combo_name"] == "combo_a"
