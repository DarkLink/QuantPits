"""Typed command boundary for ensemble fusion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from quantpits.ensemble.service import (
    EnsembleFusionService,
    prepare_ensemble_run,
    prepared_plan_json,
    render_prepared_plan,
)
from quantpits.ensemble.types import (
    EnsembleRunConfig,
    EnsembleRunSummary,
    PreparedEnsembleRun,
    options_from_namespace,
)
from quantpits.utils.workspace import WorkspaceContext


class EnsembleCommandUsageError(ValueError):
    """Raised when parsed arguments do not select an ensemble mode."""


@dataclass(frozen=True)
class EnsembleCommandDependencies:
    get_workspace_context: Callable[[], WorkspaceContext]
    load_run_config: Callable[[WorkspaceContext, str], EnsembleRunConfig]
    safeguard: Callable[[str], None]
    service_factory: Callable[[], EnsembleFusionService]


@dataclass(frozen=True)
class EnsembleCommandRequest:
    args: argparse.Namespace
    cli_args: tuple[str, ...]


@dataclass(frozen=True)
class EnsembleCommandOutcome:
    mode: Literal["json-plan", "explain-plan", "execute"]
    prepared: PreparedEnsembleRun
    rendered_output: str | None = None
    summary: EnsembleRunSummary | None = None


def build_ensemble_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensemble Fusion - 对选定模型组合进行融合预测、回测和风险分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 等权融合
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158,alstm_Alpha158

  # ICIR 加权
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method icir_weighted

  # 从 ensemble_config.json 读取 default combo
  python quantpits/scripts/ensemble_fusion.py --from-config

  # 运行指定 combo
  python quantpits/scripts/ensemble_fusion.py --combo combo_A

  # 运行所有 combo 并生成对比
  python quantpits/scripts/ensemble_fusion.py --from-config-all

  # 手动权重
  python quantpits/scripts/ensemble_fusion.py --models gru,linear_Alpha158 --method manual --weights "gru:0.6,linear_Alpha158:0.4"
""",
    )
    parser.add_argument("--models", type=str, help="逗号分隔的模型名列表（直接指定）")
    parser.add_argument("--from-config", action="store_true", help="从 ensemble_config.json 读取 default combo")
    parser.add_argument("--from-config-all", action="store_true", help="运行 ensemble_config.json 中所有 combo")
    parser.add_argument("--combo", type=str, help="运行指定名称的 combo")
    parser.add_argument(
        "--method",
        type=str,
        default="equal",
        choices=["equal", "icir_weighted", "manual", "dynamic"],
        help="权重模式 (默认 equal，--models 模式下使用)",
    )
    parser.add_argument("--weights", type=str, help='手动权重, 如 "gru:0.6,linear_Alpha158:0.4"')
    parser.add_argument(
        "--freq", type=str, default=None, choices=["day", "week"], help="回测频率 (默认从 model_config 读取)"
    )
    parser.add_argument(
        "--record-file", type=str, default="latest_train_records.json", help="训练记录文件 (默认 latest_train_records.json)"
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default=None,
        choices=["static", "rolling"],
        help="训练模式过滤 (默认 None=自动解析，static 或 rolling)",
    )
    parser.add_argument("--output-dir", type=str, default="output/ensemble", help="输出目录 (默认 output/ensemble)")
    parser.add_argument("--prediction-dir", type=str, default=None, help="预测 CSV 输出目录 (默认 output/predictions)")
    parser.add_argument("--no-backtest", action="store_true", help="跳过回测")
    parser.add_argument("--no-charts", action="store_true", help="跳过图表生成")
    parser.add_argument("--start-date", type=str, default=None, help="预测数据过滤的开始日期 YYYY-MM-DD (包含该日)")
    parser.add_argument("--end-date", type=str, default=None, help="预测数据过滤的结束日期 YYYY-MM-DD (包含该日)")
    parser.add_argument("--only-last-years", type=int, default=0, help="仅使用最后 N 年的预测数据 (作为 OOS 测试集)")
    parser.add_argument("--only-last-months", type=int, default=0, help="仅使用最后 N 个月的预测数据 (作为 OOS 测试集)")
    parser.add_argument("--detailed-analysis", action="store_true", help="生成详尽的回测分析报告（类似实盘分析）")
    parser.add_argument("--verbose-backtest", action="store_true", help="开启 Qlib 回测的详细日志模式")
    parser.add_argument("--save-csv", action="store_true", help="除了保存为 Qlib Recorder 外，同时输出 predictions csv")
    parser.add_argument(
        "--norm-method",
        type=str,
        default="rank",
        choices=["zscore", "rank"],
        help="截面归一化方法 (默认: zscore)",
    )
    parser.add_argument("--explain-plan", action="store_true", help="仅打印执行计划，不初始化 Qlib，不写文件")
    parser.add_argument("--json-plan", action="store_true", help="以 JSON 输出执行计划；隐含 dry-run，不写文件")
    parser.add_argument("--run-id", type=str, default=None, help="显式指定运行 ID（用于 dry-run/manifest 对齐）")
    parser.add_argument(
        "--no-manifest", action="store_true", help="真实执行时不写 output/manifests/ensemble_fusion/<run_id>.json"
    )
    return parser


def _validate_selector(args: argparse.Namespace) -> None:
    if not args.models and not args.from_config and not args.from_config_all and not args.combo:
        raise EnsembleCommandUsageError("必须指定 --models、--from-config、--from-config-all 或 --combo")


def run_ensemble_command(
    request: EnsembleCommandRequest,
    dependencies: EnsembleCommandDependencies,
) -> EnsembleCommandOutcome:
    """Prepare and route an ensemble command without owning process semantics."""

    args = request.args
    _validate_selector(args)

    ctx = dependencies.get_workspace_context()
    options = options_from_namespace(args)
    if options.json_plan and not options.explain_plan:
        options = replace(options, explain_plan=True)
    run_config = dependencies.load_run_config(ctx, options.record_file)
    prepared = prepare_ensemble_run(
        ctx=ctx,
        options=options,
        cli_args=request.cli_args,
        run_config=run_config,
    )

    if options.json_plan:
        return EnsembleCommandOutcome(
            mode="json-plan",
            prepared=prepared,
            rendered_output=json.dumps(
                prepared_plan_json(prepared),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
        )
    if options.explain_plan:
        return EnsembleCommandOutcome(
            mode="explain-plan",
            prepared=prepared,
            rendered_output=render_prepared_plan(prepared),
        )

    dependencies.safeguard("Ensemble Fusion")
    summary = dependencies.service_factory().execute(prepared)
    return EnsembleCommandOutcome(mode="execute", prepared=prepared, summary=summary)
