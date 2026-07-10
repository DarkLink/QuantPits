import argparse
import json
from dataclasses import replace

import pytest

from quantpits.order.command import (
    OrderCommandDependencies,
    OrderCommandRequest,
    OrderRunConfig,
    OrderRunOptions,
    OrderRunSummary,
    build_order_arg_parser,
    build_order_command_plan,
    prepare_order_run,
    resolve_order_source,
    run_order_command,
)
from quantpits.runtime import fingerprint_command_plan
from quantpits.utils.workspace import WorkspaceContext


@pytest.fixture
def ctx(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "output").mkdir()
    return WorkspaceContext.from_root(root)


@pytest.fixture
def run_config():
    return OrderRunConfig(
        merged_config={
            "market": "csi300",
            "current_cash": 1000,
            "current_holding": [{"instrument": "DEMO", "value": 100}],
        },
        cashflow_config={"cashflows": {"2026-01-01": 100}},
        strategy_config={
            "strategy": {
                "name": "topk_dropout",
                "params": {"topk": 20, "n_drop": 3, "buy_suggestion_factor": 2},
            }
        },
        ensemble_config={},
        ensemble_records={
            "default_combo": "demo_combo",
            "combos": {"demo_combo": {"record_id": "ensemble-record"}},
        },
        train_records={
            "experiment_name": "demo_train",
            "models": {"demo_model": "model-record"},
        },
    )


def _namespace(**overrides):
    values = vars(build_order_arg_parser().parse_args([]))
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parser_preserves_defaults_and_adds_plan_flags():
    args = build_order_arg_parser().parse_args([])
    assert args.output_dir == "output"
    assert args.record_file is None
    assert not args.dry_run
    assert not args.explain_plan
    assert not args.json_plan


def test_model_source_takes_precedence_over_combo(run_config):
    source = resolve_order_source(OrderRunOptions(model="demo_model", combo="demo_combo"), run_config)
    assert source.mode == "model"
    assert source.record_id == "model-record"
    assert any("takes precedence" in warning for warning in source.warnings)


def test_ensemble_source_uses_default_and_supports_record_dict(run_config):
    source = resolve_order_source(OrderRunOptions(), run_config)
    assert source.resolved_name == "demo_combo"
    assert source.record_id == "ensemble-record"


def test_missing_source_is_a_plan_warning(run_config):
    source = resolve_order_source(OrderRunOptions(model="missing"), run_config)
    assert source.record_id is None
    assert any("not found" in warning for warning in source.warnings)


def test_relative_execution_paths_are_workspace_bound(monkeypatch, ctx, run_config):
    monkeypatch.setattr("quantpits.order.command.validate_workspace", lambda *args, **kwargs: None)
    prepared = prepare_order_run(
        ctx=ctx,
        options=OrderRunOptions(output_dir="reports", record_file="records.json"),
        cli_args=(),
        run_config=run_config,
    )
    assert prepared.options.output_dir == "reports"
    assert prepared.execution_options.output_dir == (ctx.root / "reports").as_posix()
    assert prepared.execution_options.record_file == (ctx.root / "records.json").as_posix()


def test_plan_uses_legacy_config_record_fallback(ctx, run_config):
    fallback = ctx.config_path("latest_train_records.json")
    fallback.write_text('{"models": {}}', encoding="utf-8")
    source = resolve_order_source(OrderRunOptions(), run_config)
    plan = build_order_command_plan(
        ctx=ctx,
        options=OrderRunOptions(),
        cli_args=(),
        config=run_config,
        source=source,
        validation_result=None,
    )
    assert plan.metadata["record_file"] == "config/latest_train_records.json"


def test_dry_run_plan_has_preview_paths_but_no_outputs(ctx, run_config):
    options = OrderRunOptions(dry_run=True)
    source = resolve_order_source(options, run_config)
    plan = build_order_command_plan(
        ctx=ctx,
        options=options,
        cli_args=("--dry-run",),
        config=run_config,
        source=source,
        validation_result=None,
    )
    assert plan.outputs == ()
    assert len(plan.metadata["preview_output_paths"]) == 4
    assert all(step.can_skip for step in plan.steps[-2:])


def test_plan_fingerprint_ignores_run_id(ctx, run_config):
    options = OrderRunOptions()
    source = resolve_order_source(options, run_config)
    first = build_order_command_plan(
        ctx=ctx, options=replace(options, run_id="run-a"), cli_args=(), config=run_config,
        source=source, validation_result=None,
    )
    second = build_order_command_plan(
        ctx=ctx, options=replace(options, run_id="run-b"), cli_args=(), config=run_config,
        source=source, validation_result=None,
    )
    assert fingerprint_command_plan(first) == fingerprint_command_plan(second)


def test_json_plan_is_lightweight_and_does_not_mutate_namespace(monkeypatch, ctx, run_config):
    monkeypatch.setattr("quantpits.order.command.validate_workspace", lambda *args, **kwargs: None)
    args = _namespace(json_plan=True)
    calls = []
    deps = OrderCommandDependencies(
        get_workspace_context=lambda: ctx,
        load_run_config=lambda _ctx, _options: run_config,
        safeguard=lambda name: calls.append(("safeguard", name)),
        execute=lambda prepared: calls.append(("execute", prepared)),
    )
    outcome = run_order_command(OrderCommandRequest(args=args, cli_args=("--json-plan",)), deps)
    payload = json.loads(outcome.rendered_output)
    assert payload["schema_version"] == 1
    assert payload["command"] == "order_gen"
    assert calls == []
    assert args.explain_plan is False
    assert outcome.prepared.options.explain_plan is True


def test_execute_calls_safeguard_and_callback_once(monkeypatch, ctx, run_config):
    monkeypatch.setattr("quantpits.order.command.validate_workspace", lambda *args, **kwargs: None)
    calls = []
    summary = OrderRunSummary("a", "b", "ensemble", "demo", 1, 0, 0, None, None, False)
    deps = OrderCommandDependencies(
        get_workspace_context=lambda: ctx,
        load_run_config=lambda _ctx, _options: run_config,
        safeguard=lambda name: calls.append(("safeguard", name)),
        execute=lambda prepared: calls.append(("execute", prepared.plan.command)) or summary,
    )
    outcome = run_order_command(OrderCommandRequest(args=_namespace(), cli_args=()), deps)
    assert calls == [("safeguard", "Order Generation"), ("execute", "order_gen")]
    assert outcome.summary is summary
