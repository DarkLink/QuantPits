"""Deterministic external model/Qlib boundary for subprocess cycle acceptance.

All command adapters, training publication, fusion, post-trade transactions,
order persistence, paper journal and comparison remain the production services.
"""
import json
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

from quantpits.operations.process import ProcessRunner
from quantpits.research.paper_market import PaperMarket
from quantpits.utils.workspace import WorkspaceContext


def record(ctx, rid, experiment, frame):
    root = ctx.root / "mlruns" / experiment
    artifact = root / rid / "artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    (root / "meta.yaml").write_text(yaml.safe_dump({"experiment_id": experiment, "name": experiment,
        "lifecycle_stage": "active", "artifact_location": root.as_uri()}))
    (artifact.parent / "meta.yaml").write_text(yaml.safe_dump({"run_id": rid, "experiment_id": experiment,
        "status": 3, "lifecycle_stage": "active", "artifact_uri": artifact.as_uri()}))
    frame.to_pickle(artifact / "pred.pkl")
    return artifact


class DeterministicRunner(ProcessRunner):
    def run(self, ctx, module, arguments, log):
        if module.endswith("compare_accounts") and (log.parents[3] / "fail-report").exists():
            log.write_text("injected report process failure\n")
            return 1
        if module.rsplit(".", 1)[-1] in ("static_train", "ensemble_fusion", "order_gen", "prod_post_trade"):
            return super().run(ctx, "tests.quantpits.operations.cycle_worker", [module.rsplit(".", 1)[-1]] + list(arguments), log)
        return super().run(ctx, module, arguments, log)


def native(ctx, name, args):
    from quantpits.utils import env
    env.safeguard = lambda *a, **kw: None  # isolated temporary accounts need no UI delay
    if name == "static_train":
        from quantpits.scripts import static_train
        from quantpits.training import service
        from quantpits.training.records import ModelRecordEntry
        from quantpits.training.runners import TrainingTargetResult
        options = static_train.parse_args(args)
        def runner(request):
            if (ctx.root / "fail-model").exists():
                raise ValueError("injected model boundary failure")
            run, target = request.resolved_run, request.target
            index = pd.MultiIndex.from_product([[pd.Timestamp(run.anchor_date)], ["SH600001", "SH600002"]], names=["datetime", "instrument"])
            frame = pd.DataFrame({"score": [2., 1.]}, index=index)
            rid = run.prepared.plan.run_id + "-" + target.target.model_name
            artifact = record(ctx, rid, run.output_experiment_name, frame)
            source = target.source_entry
            entry = ModelRecordEntry(target.key, target.target.model_name, "static", "predict_only", "ready", rid,
                run.output_experiment_name, experiment_id=run.output_experiment_name,
                artifact_path=str(artifact.relative_to(ctx.root)), requested_anchor=run.anchor_date,
                prediction_start=run.anchor_date, prediction_end=run.anchor_date, prediction_rows=2,
                source_recorder_id=source.recorder_id, source_experiment_name=source.experiment_name, source_operation=source.operation)
            return TrainingTargetResult(target.key, target.operation, "success", entry, {"IC_Mean": .1})
        def fault(point, detail=None):
            marker = ctx.root / "interrupt-training"
            if marker.exists() and point == "after_receipt_write":
                marker.unlink()
                raise KeyboardInterrupt("injected unfinished native publication")
        service.default_execution_hooks = lambda **kw: service.TrainingExecutionHooks(
            activate_workspace=lambda root: None, init_qlib=lambda: None,
            calculate_dates=lambda: {"anchor_date": options.anchor, "test_end_time": options.anchor, "freq": "week"},
            run_static_target=runner, fault_hook=fault)
        return static_train.main(args)
    if name == "ensemble_fusion":
        from quantpits.scripts import ensemble_fusion
        from quantpits.ensemble.input_integrity import load_strict_prediction_bundle
        from quantpits.ensemble.persistence import PredictionSaveRequest, save_ensemble_predictions
        from quantpits.ensemble.types import EnsembleExecutionHooks
        from quantpits.utils import predict_utils
        def get_recorder(recorder_id, experiment_name):
            class Recorder:
                info = {"id": recorder_id, "experiment_id": experiment_name,
                        "artifact_uri": (ctx.root / "mlruns" / experiment_name / recorder_id / "artifacts").as_uri()}
                def load_object(self, name):
                    return pd.read_pickle(Path(self.info["artifact_uri"][7:]) / name)
                def list_metrics(self):
                    return {"ICIR": .1}
            return Recorder()
        def save(**kw):
            rid = kw["tags"]["run_id"] + "-ensemble"
            record(ctx, rid, "Ensemble_Fusion", kw["pred"])
            return rid
        predict_utils.save_predictions_to_recorder = save
        def combo(**kw):
            models = tuple(kw["selected_models"])
            saved = save_ensemble_predictions(PredictionSaveRequest(
                final_score=kw["norm_df"].mean(axis=1), anchor_date=kw["anchor_date"], experiment_name=kw["experiment_name"],
                method=kw["method"], model_names=models, model_metrics=kw["model_metrics"], static_weights=None,
                is_dynamic=False, output_dir=kw["args"].output_dir, combo_name=kw["combo_name"],
                is_default=kw["is_default"], workspace_root=ctx.root, prediction_dir=kw["args"].prediction_dir, save_csv=True,
                source_recorders={k: kw["train_records"]["models"][k] for k in models},
                source_anchors={k: kw["anchor_date"] for k in models}, run_id=kw["args"].run_id),
                output_inspector=lambda rid, **kw: {"recorder_id": rid, "experiment_id": "Ensemble_Fusion", "contained": True,
                    "artifact_path": "mlruns/Ensemble_Fusion/%s/artifacts" % rid})
            return {"name": kw["combo_name"], "models": list(models), "method": kw["method"],
                    "is_default": kw["is_default"], "pred_file": saved.returned_ref, "recorder_id": saved.recorder_id,
                    "output_evidence": saved.output_evidence, "report_df": None}
        ensemble_fusion.default_ensemble_hooks = lambda **kw: EnsembleExecutionHooks(
            init_qlib=lambda: None, load_selected_predictions=lambda *a, **k: None,
            load_prediction_bundle=lambda records, selected, **kw: load_strict_prediction_bundle(records, selected, recorder_getter=get_recorder, **kw),
            filter_norm_df_by_args=lambda frame, args: frame, run_single_combo=combo, compare_combos=lambda *a: None)
        return ensemble_fusion.main(args)
    if name == "order_gen":
        from quantpits.scripts import order_gen
        from quantpits.order.execution import OrderExecutionHooks, UniverseSnapshot, load_resolved_prediction
        from quantpits.order.persistence import persist_order_artifacts
        from quantpits.utils import strategy
        def prices(anchor, market, instruments):
            return pd.DataFrame({"current_close": [10.] * len(instruments), "possible_min": [9.] * len(instruments),
                                 "possible_max": [11.] * len(instruments)}, index=pd.Index(instruments, name="instrument"))
        def validate(anchor, selected):
            market = PaperMarket(ctx.qlib_data_dir)
            assert market.check_date(anchor, actual=True) == market.check_date(selected) == "VERIFIED"
        order_gen.default_order_execution_hooks = lambda: OrderExecutionHooks(
            init_qlib=lambda: None, get_anchor_date=lambda: "1900-01-01", get_next_trade_date=lambda a: "1900-01-02",
            load_predictions=load_resolved_prediction, get_price_data=prices, create_order_generator=strategy.create_order_generator,
            get_strategy_params=strategy.get_strategy_params, build_model_opinions=lambda request: None,
            persist_artifacts=persist_order_artifacts, validate_dates=validate,
            resolve_universe=lambda m, a: UniverseSnapshot.observe(m, a, PaperMarket(ctx.qlib_data_dir).universe(m, a)[0]))
        return order_gen.main(args)
    if name == "prod_post_trade":
        from quantpits.scripts import prod_post_trade
        from quantpits.post_trade import valuation
        from quantpits.post_trade import transaction
        original = transaction.PostTradeTransactionManager
        class InterruptibleManager(original):
            def __init__(self, ctx, event_hook=None):
                def fault(event, artifact):
                    marker = ctx.root / "interrupt-post"
                    if marker.exists() and event == "after_target_write_before_verification" and artifact.role == "prod_config_cursor":
                        marker.unlink()
                        raise KeyboardInterrupt("injected native post-trade interruption")
                super().__init__(ctx, event_hook=event_hook or fault)
        transaction.PostTradeTransactionManager = InterruptibleManager
        from quantpits.post_trade import service as post_service
        post_service.PostTradeTransactionManager = InterruptibleManager
        market = PaperMarket(ctx.qlib_data_dir)
        prod_post_trade.init_qlib = lambda: None
        prod_post_trade.get_trade_dates = lambda start, end, **kw: [d for d in market.calendar if start <= d <= end]
        valuation.QlibValuationProvider = lambda: valuation.MappingValuationProvider(
            {d: {"SH600001": 10, "SH600002": 10} for d in market.calendar}, {d: 100 for d in market.calendar})
        prod_post_trade.main()
        return 0
    raise ValueError(name)


if __name__ == "__main__":
    name, arguments = sys.argv[1], sys.argv[2:]
    sys.argv = [sys.argv[0]] + arguments
    if name == "cycle":
        from quantpits.scripts.run_cycle import main
        def fault(point, step):
            marker = Path(arguments[arguments.index("--project") + 1]).parent / "interrupt-after"
            if marker.exists() and marker.read_text() == step["id"]:
                marker.unlink()
                raise KeyboardInterrupt("injected native commit/project progress gap")
        raise SystemExit(main(arguments, runner=DeterministicRunner(), fault=fault))
    ctx = WorkspaceContext.from_root(os.environ["QLIB_WORKSPACE_DIR"], mlflow_uri=os.environ["MLFLOW_TRACKING_URI"],
                                     qlib_data_dir=os.environ["QLIB_DATA_DIR"], qlib_region=os.environ["QLIB_REGION"])
    raise SystemExit(native(ctx, name, arguments) or 0)
