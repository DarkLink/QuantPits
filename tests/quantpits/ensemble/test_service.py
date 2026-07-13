import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from quantpits.ensemble import (
    EmptyPredictionWindowError,
    EnsembleExecutionHooks,
    EnsembleFusionService,
    EnsembleRunConfig,
    EnsembleRunOptions,
    NoRequiredModelsError,
    prepare_ensemble_run,
    prepared_plan_json,
    render_prepared_plan,
)
from quantpits.utils.workspace import WorkspaceContext


def _run_config():
    return EnsembleRunConfig(
        train_records={
            "anchor_date": "2026-07-09",
            "experiment_name": "DemoExp",
            "models": {"m1@static": "rid1"},
        },
        model_config={"freq": "week"},
        ensemble_config={
            "combos": {
                "default_combo": {
                    "models": ["m1"],
                    "method": "equal",
                    "default": True,
                }
            }
        },
    )


def _workspace(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    return WorkspaceContext.from_root(root)


def _norm_df():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-07-09"), "AAA")],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"m1@static": [0.5]}, index=idx)


def _result():
    return {
        "name": "default_combo",
        "models": ["m1@static"],
        "method": "equal",
        "is_default": True,
        "pred_file": "recorder-id",
        "report_df": None,
    }


def _prepared(ctx, *, options=None):
    options = options or EnsembleRunOptions(from_config=True, run_id="service-run")
    return prepare_ensemble_run(
        ctx=ctx,
        options=options,
        cli_args=("--from-config", "--run-id", options.run_id or "service-run"),
        run_config=_run_config(),
        validate=False,
    )


def test_prepare_and_render_plan_do_not_import_script(tmp_path):
    _mod = sys.modules.pop("quantpits.scripts.ensemble_fusion", None)
    try:
        ctx = _workspace(tmp_path)

        prepared = _prepared(ctx)
        payload = prepared_plan_json(prepared)
        rendered = render_prepared_plan(prepared)

        assert "quantpits.scripts.ensemble_fusion" not in sys.modules
        assert payload["plan"]["command"] == "ensemble_fusion"
        assert payload["plan_fingerprint"]
        assert "raw_config" not in json.dumps(payload)
        assert "--- Execution Plan (dry run) ---" in rendered
        assert "Plan fingerprint:" in rendered
    finally:
        if _mod is not None:
            sys.modules["quantpits.scripts.ensemble_fusion"] = _mod


def test_service_execute_success_writes_manifest_and_operator_log(tmp_path):
    ctx = _workspace(tmp_path)
    prepared = _prepared(ctx)
    calls = []

    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: calls.append("init"),
        load_selected_predictions=lambda *a, **k: (calls.append("load") or (_norm_df(), {"m1@static": 0.1}, ["m1@static"])),
        filter_norm_df_by_args=lambda norm_df, args: calls.append("filter") or norm_df,
        run_single_combo=lambda **kwargs: calls.append("combo") or _result(),
        compare_combos=lambda *a, **k: calls.append("compare"),
    )

    summary = EnsembleFusionService(hooks).execute(prepared)

    assert calls == ["init", "load", "filter", "combo"]
    assert summary.manifest_path == "output/manifests/ensemble_fusion/service-run.json"
    manifest = json.loads((ctx.root / summary.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["records"]["n_combos"] == 1
    log_entry = json.loads(ctx.data_path("operator_log.jsonl").read_text(encoding="utf-8").strip())
    assert log_entry["run_id"] == "service-run"
    assert log_entry["manifest_path"] == summary.manifest_path
    assert log_entry["plan_fingerprint"]


def test_service_execute_passes_workspace_absolute_paths_to_hooks(tmp_path):
    ctx = _workspace(tmp_path)
    prepared = _prepared(
        ctx,
        options=EnsembleRunOptions(
            from_config=True,
            run_id="path-run",
            output_dir="output/ensemble",
            prediction_dir="output/predictions",
        ),
    )
    seen = {}

    def run_combo(**kwargs):
        args = kwargs["args"]
        seen["output_dir"] = args.output_dir
        seen["prediction_dir"] = args.prediction_dir
        seen["workspace_root"] = args.workspace_root
        seen["cli_args"] = args.cli_args
        return _result()

    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *a, **k: (_norm_df(), {"m1@static": 0.1}, ["m1@static"]),
        filter_norm_df_by_args=lambda norm_df, args: norm_df,
        run_single_combo=run_combo,
        compare_combos=lambda *a, **k: None,
    )

    EnsembleFusionService(hooks).execute(prepared)

    assert seen["output_dir"] == (ctx.root / "output" / "ensemble").as_posix()
    assert seen["prediction_dir"] == (ctx.root / "output" / "predictions").as_posix()
    assert seen["workspace_root"] == ctx.root.as_posix()
    assert seen["cli_args"] == ["--from-config", "--run-id", "path-run"]


def test_service_execute_respects_no_manifest(tmp_path):
    ctx = _workspace(tmp_path)
    options = EnsembleRunOptions(from_config=True, run_id="no-manifest-run", no_manifest=True)
    prepared = _prepared(ctx, options=options)

    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *a, **k: (_norm_df(), {}, ["m1@static"]),
        filter_norm_df_by_args=lambda norm_df, args: norm_df,
        run_single_combo=lambda **kwargs: _result(),
        compare_combos=lambda *a, **k: None,
    )

    summary = EnsembleFusionService(hooks).execute(prepared)

    assert summary.manifest_path is None
    assert not ctx.output_path("manifests").exists()
    log_entry = json.loads(ctx.data_path("operator_log.jsonl").read_text(encoding="utf-8").strip())
    assert log_entry["manifest_path"] is None


def test_service_execute_failure_writes_failed_manifest_and_reraises(tmp_path):
    ctx = _workspace(tmp_path)
    prepared = _prepared(ctx, options=EnsembleRunOptions(from_config=True, run_id="failed-run"))

    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("recorder missing")),
        filter_norm_df_by_args=lambda norm_df, args: norm_df,
        run_single_combo=lambda **kwargs: _result(),
        compare_combos=lambda *a, **k: None,
    )

    with pytest.raises(RuntimeError, match="recorder missing"):
        EnsembleFusionService(hooks).execute(prepared)

    manifest = json.loads(
        ctx.output_path("manifests", "ensemble_fusion", "failed-run.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error"] == {"type": "RuntimeError", "message": "recorder missing"}
    log_entry = json.loads(ctx.data_path("operator_log.jsonl").read_text(encoding="utf-8").strip())
    assert log_entry["manifest_path"] == "output/manifests/ensemble_fusion/failed-run.json"


def test_service_execute_failure_after_partial_combo_records_count(tmp_path):
    ctx = _workspace(tmp_path)
    config = _run_config()
    config = replace(
        config,
        ensemble_config={
            "combos": {
                "combo_a": {"models": ["m1"], "method": "equal", "default": True},
                "combo_b": {"models": ["m1"], "method": "equal", "default": False},
            }
        },
    )
    options = EnsembleRunOptions(from_config_all=True, run_id="partial-failed-run")
    prepared = prepare_ensemble_run(
        ctx=ctx,
        options=options,
        cli_args=("--from-config-all",),
        run_config=config,
        validate=False,
    )
    seen = {"count": 0}

    def run_combo(**kwargs):
        seen["count"] += 1
        if seen["count"] == 2:
            raise RuntimeError("combo failed")
        result = _result()
        result["name"] = "combo_a"
        return result

    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *a, **k: (_norm_df(), {}, ["m1@static"]),
        filter_norm_df_by_args=lambda norm_df, args: norm_df,
        run_single_combo=run_combo,
        compare_combos=lambda *a, **k: None,
    )

    with pytest.raises(RuntimeError, match="combo failed"):
        EnsembleFusionService(hooks).execute(prepared)

    manifest = json.loads(
        ctx.output_path("manifests", "ensemble_fusion", "partial-failed-run.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["records"]["n_combos"] == 1


def test_service_execute_multi_combo_calls_comparison(tmp_path):
    ctx = _workspace(tmp_path)
    config = replace(
        _run_config(),
        ensemble_config={
            "combos": {
                "combo_a": {"models": ["m1"], "method": "equal", "default": True},
                "combo_b": {"models": ["m1"], "method": "equal", "default": False},
            }
        },
    )
    prepared = prepare_ensemble_run(
        ctx=ctx,
        options=EnsembleRunOptions(from_config_all=True, run_id="compare-run"),
        cli_args=("--from-config-all",),
        run_config=config,
        validate=False,
    )
    calls = []

    def run_combo(**kwargs):
        result = _result()
        result["name"] = kwargs["combo_name"]
        return result

    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *a, **k: (_norm_df(), {}, ["m1@static"]),
        filter_norm_df_by_args=lambda norm_df, args: norm_df,
        run_single_combo=run_combo,
        compare_combos=lambda *a, **k: calls.append(a),
    )

    summary = EnsembleFusionService(hooks).execute(prepared)

    assert len(summary.combo_results) == 2
    assert len(calls) == 1
    assert calls[0][1] == "2026-07-09"


def test_service_execute_no_required_models_raises_typed_error_without_failed_manifest(tmp_path):
    ctx = _workspace(tmp_path)
    prepared = replace(
        _prepared(ctx, options=EnsembleRunOptions(from_config=True, run_id="no-model-run")),
        combos=(
            SimpleNamespace(
                name="empty_combo",
                models=(),
                method="equal",
                manual_weights=None,
                is_default=True,
                warnings=(),
            ),
        ),
    )
    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *a, **k: (_norm_df(), {}, []),
        filter_norm_df_by_args=lambda norm_df, args: norm_df,
        run_single_combo=lambda **kwargs: _result(),
        compare_combos=lambda *a, **k: None,
    )

    with pytest.raises(NoRequiredModelsError, match="没有有效的模型"):
        EnsembleFusionService(hooks).execute(prepared)

    assert not ctx.output_path("manifests").exists()
    log_entry = json.loads(ctx.data_path("operator_log.jsonl").read_text(encoding="utf-8").strip())
    assert log_entry["exception"]["type"] == "NoRequiredModelsError"


def test_service_execute_empty_filtered_predictions_writes_failed_manifest(tmp_path):
    ctx = _workspace(tmp_path)
    prepared = _prepared(ctx, options=EnsembleRunOptions(from_config=True, run_id="empty-filter-run"))
    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *a, **k: (_norm_df(), {}, ["m1@static"]),
        filter_norm_df_by_args=lambda norm_df, args: norm_df.iloc[0:0],
        run_single_combo=lambda **kwargs: _result(),
        compare_combos=lambda *a, **k: None,
    )

    with pytest.raises(EmptyPredictionWindowError, match="过滤后没有预测数据"):
        EnsembleFusionService(hooks).execute(prepared)

    manifest = ctx.output_path("manifests", "ensemble_fusion", "empty-filter-run.json")
    assert json.loads(manifest.read_text(encoding="utf-8"))["error"]["type"] == "EmptyPredictionWindowError"
    log_entry = json.loads(ctx.data_path("operator_log.jsonl").read_text(encoding="utf-8").strip())
    assert log_entry["exception"]["type"] == "EmptyPredictionWindowError"
