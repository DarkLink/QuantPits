"""Mechanical public-boundary drivers with deterministic external fakes."""

from dataclasses import dataclass
import pandas as pd

from quantpits.ensemble.command import EnsembleCommandDependencies
from quantpits.ensemble.config import load_ensemble_run_config
from quantpits.ensemble.execution import EnsembleExecutionError
from quantpits.ensemble.input_integrity import load_strict_prediction_bundle
from quantpits.ensemble.persistence import PredictionSaveRequest, save_ensemble_predictions
from quantpits.ensemble.service import EnsembleFusionService
from quantpits.ensemble.types import EnsembleExecutionHooks
from quantpits.order.command import OrderRunOptions, load_order_run_config, prepare_order_run
from quantpits.order.execution import LoadedOrderPrediction, OrderExecutionHooks
from quantpits.order.persistence import persist_order_artifacts
from quantpits.order.service import OrderGenerationService


def prediction_frame(anchor="2026-07-16", values=(0.9, 0.1)):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(anchor), "AAA"), (pd.Timestamp(anchor), "BBB")],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"score": list(values)}, index=index)


@dataclass
class FakeRecorder:
    recorder_id: str
    artifact_uri: str
    prediction: object
    experiment_id: str = "sanitized-experiment-id"

    @property
    def info(self):
        return {
            "id": self.recorder_id,
            "artifact_uri": self.artifact_uri,
            "experiment_id": self.experiment_id,
        }

    def load_object(self, name):
        if name != "pred.pkl" or isinstance(self.prediction, BaseException):
            raise self.prediction if isinstance(self.prediction, BaseException) else KeyError(name)
        return self.prediction

    def list_metrics(self):
        return {"ICIR": 0.1}


def recorder_inventory(workspace, *, foreign_key=None, missing_key=None):
    records = workspace.read_json("latest_train_records.json")["models"]
    inventory = {}
    for key, recorder_id in records.items():
        if key == missing_key:
            continue
        artifact = workspace.root / "mlruns" / recorder_id / "artifacts"
        if key == foreign_key:
            artifact = workspace.root.parent / "outside" / recorder_id / "artifacts"
        artifact.mkdir(parents=True, exist_ok=True)
        inventory[recorder_id] = FakeRecorder(recorder_id, artifact.as_uri(), prediction_frame())
    return inventory


def ensemble_command_dependencies(
    workspace, inventory, *, output_recorder="ensemble-sentinel", monkeypatch=None,
    publication_error=False,
):
    if monkeypatch is None:
        raise ValueError("semantic ensemble driver requires pytest monkeypatch")

    def save_fake_recorder(**kwargs):
        artifact = workspace.root / "mlruns" / output_recorder / "artifacts" / "pred.pkl"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(kwargs["pred"].to_csv().encode("utf-8"))
        if publication_error:
            raise EnsembleExecutionError("semantic publication failed after recorder bytes")
        return output_recorder

    monkeypatch.setattr(
        "quantpits.utils.predict_utils.save_predictions_to_recorder",
        save_fake_recorder,
    )
    run_config = load_ensemble_run_config(workspace.ctx)

    def getter(recorder_id, experiment_name):
        del experiment_name
        if recorder_id not in inventory:
            raise KeyError(recorder_id)
        return inventory[recorder_id]

    def load_bundle(records, selected, **kwargs):
        return load_strict_prediction_bundle(records, selected, recorder_getter=getter, **kwargs)

    def run_combo(**kwargs):
        models = tuple(kwargs["selected_models"])
        source_recorders = {
            key: run_config.train_records["models"][key]
            for key in models
        }
        saved = save_ensemble_predictions(
            PredictionSaveRequest(
                final_score=kwargs["norm_df"].mean(axis=1),
                anchor_date=kwargs["anchor_date"],
                experiment_name=kwargs["experiment_name"],
                method=kwargs["method"],
                model_names=models,
                model_metrics=kwargs["model_metrics"],
                static_weights={key: 1.0 / len(models) for key in models},
                is_dynamic=False,
                output_dir=kwargs["args"].output_dir,
                combo_name=kwargs["combo_name"],
                is_default=kwargs["is_default"],
                workspace_root=workspace.root,
                source_recorders=source_recorders,
                source_anchors={key: kwargs["anchor_date"] for key in models},
                run_id=kwargs["args"].run_id,
                plan_fingerprint=kwargs["args"].plan_fingerprint,
            ),
            output_inspector=lambda *args, **kw: {
                "recorder_id": output_recorder,
                "experiment_id": "ensemble-experiment-id",
                "artifact_path": "mlruns/ensemble-sentinel/artifacts",
            },
        )
        return {
            "name": kwargs["combo_name"], "models": list(models), "method": kwargs["method"],
            "is_default": kwargs["is_default"], "pred_file": saved.returned_ref,
            "recorder_id": saved.recorder_id, "report_df": None,
        }

    hooks = EnsembleExecutionHooks(
        init_qlib=lambda: None,
        load_selected_predictions=lambda *args, **kwargs: None,
        filter_norm_df_by_args=lambda frame, args: frame,
        run_single_combo=run_combo,
        compare_combos=lambda *args, **kwargs: None,
        load_prediction_bundle=load_bundle,
    )
    service = EnsembleFusionService(hooks)
    dependencies = EnsembleCommandDependencies(
        get_workspace_context=lambda: workspace.ctx,
        load_run_config=lambda ctx, record_file: run_config,
        safeguard=lambda label: None,
        service_factory=lambda: service,
    )
    return dependencies, service


class RecordingOrderGenerator:
    def __init__(self):
        self.cash_seen = None
        self.holdings_seen = None

    def analyze_positions(self, predictions, prices, holdings):
        self.holdings_seen = tuple(dict(item) for item in holdings)
        frame = pd.DataFrame({"score": [0.9], "current_close": [10.0]}, index=["AAA"])
        return frame.iloc[0:0], frame.iloc[0:0], frame, frame, 1

    def generate_sell_orders(self, candidates, holdings, trade_date):
        return [], 0.0

    def generate_buy_orders(self, candidates, count, cash, trade_date):
        self.cash_seen = cash
        return [{"instrument": "AAA", "estimated_amount": min(float(cash), 100.0)}]


def execute_order(workspace, *, run_id="order-semantic"):
    options = OrderRunOptions(output_dir="output", run_id=run_id)
    config = load_order_run_config(workspace.ctx, options)
    prepared = prepare_order_run(
        ctx=workspace.ctx, options=options, cli_args=("--run-id", run_id), run_config=config,
    )
    generator = RecordingOrderGenerator()
    loaded = LoadedOrderPrediction(prediction_frame(), prepared.source, "sanitized ensemble source")
    hooks = OrderExecutionHooks(
        init_qlib=lambda: None,
        get_anchor_date=lambda: "2026-07-16",
        get_next_trade_date=lambda anchor: "2026-07-17",
        load_predictions=lambda source: loaded,
        get_price_data=lambda *args, **kwargs: pd.DataFrame({"current_close": [10.0]}, index=["AAA"]),
        create_order_generator=lambda config: generator,
        get_strategy_params=lambda config: {"topk": 1, "n_drop": 0, "buy_suggestion_factor": 1},
        build_model_opinions=lambda request: None,
        persist_artifacts=persist_order_artifacts,
    )
    summary = OrderGenerationService(hooks).execute(prepared)
    return prepared, summary, generator
