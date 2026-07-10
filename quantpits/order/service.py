"""Order generation execution lifecycle and run-manifest integration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from quantpits.order.command import OrderRunSummary, PreparedOrderRun
from quantpits.order.execution import (
    LoadedOrderPrediction,
    OrderCalculationResult,
    OrderExecutionError,
    OrderExecutionHooks,
    normalize_prediction_data,
)
from quantpits.order.opinions import ModelOpinionsRequest
from quantpits.order.persistence import (
    OrderArtifactLedger,
    OrderPersistenceError,
    OrderPersistenceRequest,
    display_path,
)
from quantpits.runtime import CommandResult, OutputRef, manifest_from_result, manifest_path, write_run_manifest
from quantpits.utils.operator_log import OperatorLog


def _cashflow_today(config: dict, anchor_date: str) -> float:
    cashflows = config.get("cashflows", {})
    if isinstance(cashflows, dict) and anchor_date in cashflows:
        return float(cashflows[anchor_date])
    return float(config.get("cash_flow_today", 0))


def _focus_instruments(*frames: Any) -> tuple[str, ...]:
    values: list[str] = []
    for frame in frames:
        if len(frame) == 0:
            continue
        index = frame.index.get_level_values("instrument") if "instrument" in frame.index.names else frame.index
        values.extend(str(item) for item in index.tolist())
    return tuple(dict.fromkeys(values))


def _estimated_buy_range(orders: tuple[dict, ...], target_count: int) -> tuple[float | None, float | None]:
    if not orders or target_count <= 0:
        return None, None
    amounts = sorted(float(item["estimated_amount"]) for item in orders)
    return sum(amounts[:target_count]), sum(amounts[-target_count:])


def _manifest_self_ref(prepared: PreparedOrderRun) -> OutputRef:
    path = manifest_path(prepared.ctx, "order_gen", prepared.plan.run_id)
    return OutputRef(display_path(prepared.ctx, path), kind="manifest", description="run manifest", overwrite=True)


def _records(prepared: PreparedOrderRun, calculation: OrderCalculationResult, ledger: OrderArtifactLedger) -> dict:
    return {
        "anchor_date": calculation.anchor_date,
        "trade_date": calculation.trade_date,
        "source": {
            "mode": prepared.source.mode,
            "requested_name": prepared.source.requested_name,
            "resolved_name": prepared.source.resolved_name,
            "record_id": prepared.source.record_id,
            "experiment_name": prepared.source.experiment_name,
        },
        "holding_count": calculation.holding_count,
        "sell_count": len(calculation.sell_orders),
        "target_buy_count": calculation.target_buy_count,
        "buy_suggestion_count": len(calculation.buy_orders),
        "estimated_sell_amount": calculation.estimated_sell_amount,
        "estimated_buy_min": calculation.estimated_buy_min,
        "estimated_buy_max": calculation.estimated_buy_max,
        "opinion_source_count": len(calculation.opinions.source_summaries) if calculation.opinions else 0,
        "actual_output_count": len(ledger.outputs),
    }


class OrderGenerationService:
    def __init__(self, hooks: OrderExecutionHooks):
        self.hooks = hooks

    def _calculate(self, prepared: PreparedOrderRun) -> OrderCalculationResult:
        import pandas as pd

        options = prepared.execution_options
        config = prepared.config.merged_config
        print(f"\n{'#' * 60}\n# Order Generation — 订单生成\n{'#' * 60}")
        self.hooks.init_qlib()
        anchor_date = self.hooks.get_anchor_date()
        generator = self.hooks.create_order_generator(prepared.config.strategy_config)
        params = self.hooks.get_strategy_params(prepared.config.strategy_config)
        top_k = params.get("topk", 20)
        drop_n = params.get("n_drop", 3)
        factor = params.get("buy_suggestion_factor", 2)
        current_holding = config.get("current_holding", [])
        current_cash = float(config.get("current_cash", 0))
        print("\nStage 0: 配置加载")
        print(f"当前持仓   : {len(current_holding)} 个")

        loaded: LoadedOrderPrediction = self.hooks.load_predictions(prepared.source)
        predictions = normalize_prediction_data(loaded.data)
        print("\nStage 1: 加载预测数据")
        print(f"预测来源   : {loaded.description}")
        latest = predictions.index.get_level_values("datetime").max()
        instruments = predictions.xs(latest, level="datetime").index.tolist()
        prices = self.hooks.get_price_data(anchor_date, config.get("market", "csi300"), instruments=instruments)
        trade_date = self.hooks.get_next_trade_date(anchor_date)
        print("\nStage 2: 获取价格数据")
        print(f"下一交易日 : {trade_date}")
        hold, sell_candidates, buy_candidates, sorted_frame, target_buy_count = generator.analyze_positions(
            predictions, prices, current_holding
        )
        print("\nStage 3: 排序与持仓分析")
        print(f"继续持有   : {len(hold)} 个")
        print(f"计划卖出   : {len(sell_candidates)} 个")
        if options.verbose:
            print("\n--- 继续持有 ---")
            if len(hold):
                print(hold.to_string())
            print("\n--- 卖出候选 ---")
            if len(sell_candidates):
                print(sell_candidates.to_string())
            print("\n--- 买入候选 ---")
            if len(buy_candidates):
                print(buy_candidates.to_string())

        opinions = self.hooks.build_model_opinions(
            ModelOpinionsRequest(
                focus_instruments=_focus_instruments(hold, sell_candidates, buy_candidates),
                current_holding_instruments=tuple(item["instrument"] for item in current_holding),
                top_k=top_k,
                drop_n=drop_n,
                buy_suggestion_factor=factor,
                sorted_predictions=sorted_frame,
                trade_date=trade_date,
                run_config=prepared.config,
                load_prediction=self._load_opinion_prediction,
            )
        )
        if opinions is not None:
            for warning in opinions.warnings:
                print(f"⚠️  Opinion source warning: {warning}")
        sell_orders, sell_amount = generator.generate_sell_orders(sell_candidates, current_holding, trade_date)
        cash = current_cash + float(sell_amount) + _cashflow_today(prepared.config.cashflow_config, anchor_date)
        buy_orders = generator.generate_buy_orders(buy_candidates, target_buy_count, cash, trade_date)
        sell_tuple = tuple(dict(item) for item in sell_orders)
        buy_tuple = tuple(dict(item) for item in buy_orders)
        buy_min, buy_max = _estimated_buy_range(buy_tuple, target_buy_count)

        if options.verbose and opinions is not None and not opinions.dataframe.empty:
            print(opinions.dataframe.to_string())
        print("\nStage 4: 生成卖出订单")
        print(f"卖出订单   : {len(sell_tuple)}")
        if sell_tuple:
            print(pd.DataFrame(sell_tuple).to_string(index=False))
        print("\nStage 5: 生成买入订单")
        print(f"买入备选   : {len(buy_tuple)}")
        if buy_tuple:
            print(pd.DataFrame(buy_tuple).to_string(index=False))
        return OrderCalculationResult(
            anchor_date=anchor_date,
            trade_date=trade_date,
            source_label=prepared.source.source_label,
            source_description=loaded.description.split("\n")[0],
            holding_count=len(hold),
            target_buy_count=target_buy_count,
            sell_orders=sell_tuple,
            buy_orders=buy_tuple,
            estimated_sell_amount=float(sell_amount),
            estimated_buy_min=buy_min,
            estimated_buy_max=buy_max,
            opinions=opinions,
        )

    @staticmethod
    def _load_opinion_prediction(record_id: str, experiment_name: str) -> Any:
        from qlib.workflow import R

        return R.get_recorder(recorder_id=record_id, experiment_name=experiment_name).load_object("pred.pkl")

    def _persist(self, prepared: PreparedOrderRun, calculation: OrderCalculationResult) -> OrderArtifactLedger:
        return self.hooks.persist_artifacts(
            OrderPersistenceRequest(
                ctx=prepared.ctx,
                output_dir=Path(prepared.execution_options.output_dir),
                trade_date=calculation.trade_date,
                source_label=calculation.source_label,
                sell_orders=calculation.sell_orders,
                buy_orders=calculation.buy_orders,
                opinions=calculation.opinions,
            )
        )

    def _summary(
        self,
        prepared: PreparedOrderRun,
        calculation: OrderCalculationResult,
        ledger: OrderArtifactLedger,
        manifest_file: str | None,
    ) -> OrderRunSummary:
        return OrderRunSummary(
            anchor_date=calculation.anchor_date,
            trade_date=calculation.trade_date,
            source_label=calculation.source_label,
            source_description=calculation.source_description,
            holding_count=calculation.holding_count,
            sell_count=len(calculation.sell_orders),
            buy_count=len(calculation.buy_orders),
            sell_file=ledger.sell_csv,
            buy_file=ledger.buy_csv,
            dry_run=prepared.execution_options.dry_run,
            run_id=prepared.plan.run_id,
            manifest_path=manifest_file,
            opinion_csv_file=ledger.opinion_csv,
            opinion_json_file=ledger.opinion_json,
            target_buy_count=calculation.target_buy_count,
            estimated_sell_amount=calculation.estimated_sell_amount,
            estimated_buy_min=calculation.estimated_buy_min,
            estimated_buy_max=calculation.estimated_buy_max,
            actual_outputs=tuple(item.path for item in ledger.outputs),
        )

    def _write_success_manifest(
        self, prepared: PreparedOrderRun, started_at: str, calculation: OrderCalculationResult, ledger: OrderArtifactLedger
    ) -> str | None:
        if prepared.execution_options.no_manifest:
            return None
        outputs = ledger.outputs + (_manifest_self_ref(prepared),)
        result = CommandResult(
            plan=prepared.plan,
            status="success",
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
            outputs=outputs,
            records=_records(prepared, calculation, ledger),
            warnings=calculation.opinions.warnings if calculation.opinions else (),
        )
        try:
            return display_path(prepared.ctx, write_run_manifest(prepared.ctx, manifest_from_result(result)))
        except Exception as exc:
            print(f"⚠️  RunManifest Warning: Could not write manifest: {exc}")
            return None

    def _write_failed_manifest(
        self, prepared: PreparedOrderRun, started_at: str, exc: BaseException, outputs: tuple[OutputRef, ...]
    ) -> None:
        if prepared.execution_options.no_manifest:
            return
        refs = outputs + (_manifest_self_ref(prepared),)
        result = CommandResult(
            plan=prepared.plan,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now().isoformat(),
            outputs=refs,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        try:
            write_run_manifest(prepared.ctx, manifest_from_result(result))
        except Exception as manifest_exc:
            print(f"⚠️  RunManifest Warning: Could not write failed manifest: {manifest_exc}")

    def execute(self, prepared: PreparedOrderRun) -> OrderRunSummary:
        started_at = datetime.now().isoformat()
        if prepared.execution_options.dry_run:
            print("\n⚠️  DRY-RUN 模式: 不会写入任何文件")
            calculation = self._calculate(prepared)
            from quantpits.order.persistence import build_order_artifact_paths

            preview = build_order_artifact_paths(
                prepared.execution_options.output_dir,
                calculation.trade_date,
                calculation.source_label,
            )
            print("\n[DRY-RUN] 以下文件不会被写入:")
            for path in (preview.opinion_csv, preview.opinion_json, preview.sell_csv, preview.buy_csv):
                print(f"  {path}")
            print("\n# ✅ 订单生成完成!")
            print(f"📊 预测来源 : {calculation.source_description}")
            return self._summary(prepared, calculation, OrderArtifactLedger(), None)

        log_file = prepared.ctx.data_path("operator_log.jsonl").as_posix()
        with OperatorLog(
            "order_gen",
            args=list(prepared.cli_args),
            log_file=log_file,
            run_id=prepared.plan.run_id,
            plan_fingerprint=prepared.plan_fingerprint,
        ) as oplog:
            committed: tuple[OutputRef, ...] = ()
            try:
                calculation = self._calculate(prepared)
                ledger = self._persist(prepared, calculation)
                committed = ledger.outputs
                print("\nStage 6: 保存订单")
                for output in ledger.outputs:
                    print(f"  {output.path}")
                manifest_file = self._write_success_manifest(prepared, started_at, calculation, ledger)
                oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_file)
                oplog.set_result(_records(prepared, calculation, ledger))
                print("\n# ✅ 订单生成完成!")
                print(f"📊 预测来源 : {calculation.source_description}")
                return self._summary(prepared, calculation, ledger, manifest_file)
            except OrderExecutionError:
                raise
            except OrderPersistenceError as exc:
                self._write_failed_manifest(prepared, started_at, exc, exc.committed_outputs)
                raise
            except Exception as exc:
                self._write_failed_manifest(prepared, started_at, exc, committed)
                raise
