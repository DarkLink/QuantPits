"""Workspace-safe command planning for unified post-trade intake."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import pandas as pd

from quantpits.post_trade.contracts import (
    ParsedPostTradeInput, PostTradeInputCatalog, PostTradeInputError,
    PostTradeIntakeIssue, PostTradePlanError,
    PostTradeScope,
)
from quantpits.post_trade.contracts import PostTradeInputMissingError, PostTradeSourceRef
from quantpits.post_trade.intake import (
    discover_inputs, load_ingestion_receipts, parse_pending_sources_with_catalog,
    validate_cross_stream, verify_parsed_sources, verify_settlement_bundle,
)
from quantpits.post_trade.ingestion import IngestionResult, ingest_execution_evidence
from quantpits.runtime import CommandPlan, CommandStep, InputRef, OutputRef, StateRef
from quantpits.runtime import generate_run_id
from quantpits.runtime.command import fingerprint_command_plan
from quantpits.runtime.render import render_command_plan
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file


@dataclass(frozen=True)
class PostTradeRunOptions:
    scope: PostTradeScope = "all"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    broker: Optional[str] = None
    dry_run: bool = False
    explain_plan: bool = False
    json_plan: bool = False
    allow_missing_settlement: bool = False
    settlement_bundle: Optional[str] = None
    verbose: bool = False
    run_id: Optional[str] = None
    no_manifest: bool = False


@dataclass(frozen=True)
class PostTradeRunConfig:
    prod_config: dict
    cashflow_config: dict
    broker_name: str
    state_cursor: str
    legacy_execution_cursor: Optional[str]
    prod_state_document: Optional[dict] = None

    @property
    def runtime_config(self):
        return self.prod_config

    @property
    def raw_prod_config(self):
        return self.prod_state_document if self.prod_state_document is not None else self.prod_config


@dataclass(frozen=True)
class PreparedPostTradeRun:
    ctx: WorkspaceContext
    options: PostTradeRunOptions
    execution_options: PostTradeRunOptions
    cli_args: Tuple[str, ...]
    config: PostTradeRunConfig
    catalog: PostTradeInputCatalog
    plan: CommandPlan
    plan_fingerprint: str


@dataclass(frozen=True)
class PostTradeRunSummary:
    prepared: PreparedPostTradeRun
    ingestion: Optional[IngestionResult] = None
    state_result: Any = None
    parsed_inputs: Optional[dict] = None


@dataclass(frozen=True)
class PostTradeCommandDependencies:
    """Injectable command boundary for script adapters and tests.

    Preparation only consumes ``load_run_config`` and ``discover_inputs``;
    the remaining hooks document the process-boundary ownership used by the
    two compatibility CLIs without importing either script here.
    """

    get_workspace_context: Optional[Callable[[], WorkspaceContext]] = None
    load_run_config: Optional[Callable[[WorkspaceContext, PostTradeRunOptions], PostTradeRunConfig]] = None
    discover_inputs: Optional[Callable[..., PostTradeInputCatalog]] = None
    safeguard: Optional[Callable[[str], None]] = None
    execute: Optional[Callable[[PreparedPostTradeRun], PostTradeRunSummary]] = None


def add_post_trade_arguments(parser: argparse.ArgumentParser, *, default_scope: str) -> argparse.ArgumentParser:
    parser.add_argument("--scope", choices=("all", "state", "execution"), default=default_scope)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--broker", default=None)
    parser.add_argument("--allow-missing-settlement", action="store_true")
    parser.add_argument(
        "--settlement-bundle", action="append", metavar="PATH",
        help="Explicit YYYY-MM-DD-YYYY-MM-DD-table.xlsx settlement source",
    )
    parser.add_argument("--dry-run", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--explain-plan", action="store_true")
    modes.add_argument("--json-plan", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--transaction-status", action="store_true", help="Show recoverable transaction status without writing")
    parser.add_argument("--retry-classification", metavar="TRANSACTION_ID", help="Retry classification for a committed transaction")
    parser.add_argument("--verbose", action="store_true")
    return parser


def options_from_namespace(args: argparse.Namespace) -> PostTradeRunOptions:
    if args.dry_run and (args.explain_plan or args.json_plan):
        raise PostTradePlanError("--dry-run cannot be combined with plan-only modes")
    values = {name: getattr(args, name) for name in PostTradeRunOptions.__dataclass_fields__}
    bundles = values.get("settlement_bundle")
    if bundles is not None:
        if not isinstance(bundles, list):
            bundles = [bundles]
        if len(bundles) != 1:
            raise PostTradePlanError("--settlement-bundle may be provided exactly once")
        if not isinstance(bundles[0], str) or not bundles[0].strip():
            raise PostTradePlanError("--settlement-bundle requires a non-empty path")
        values["settlement_bundle"] = bundles[0]
    return PostTradeRunOptions(**values)


def _read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_config(ctx: WorkspaceContext, options: PostTradeRunOptions) -> PostTradeRunConfig:
    prod = _read_json(ctx.config_path("prod_config.json"), {})
    cashflow = _read_json(ctx.config_path("cashflow.json"), {})
    state_cursor = prod.get("last_processed_date", prod.get("current_date"))
    if not state_cursor:
        raise PostTradePlanError("prod_config.json does not define current_date/last_processed_date")
    legacy = _read_json(ctx.data_path(".order_trade_state.json"), {}).get("last_processed_date")
    return PostTradeRunConfig(prod, cashflow, options.broker or prod.get("broker", "gtja"), state_cursor, legacy)


def _ref(ctx: WorkspaceContext, path: Path, *, kind: str = "data", required: bool = False) -> InputRef:
    try:
        display = path.relative_to(ctx.root).as_posix()
    except ValueError:
        display = path.as_posix()
    return InputRef(display, kind=kind, fingerprint=fingerprint_file(path) if path.exists() else None, required=required)


def _build_plan(ctx: WorkspaceContext, options: PostTradeRunOptions, config: PostTradeRunConfig, catalog: PostTradeInputCatalog, cli_args: Tuple[str, ...]) -> CommandPlan:
    inputs = [_ref(ctx, ctx.config_path("prod_config.json"), kind="config", required=True), _ref(ctx, ctx.config_path("cashflow.json"), kind="config")]
    inputs.extend([
        _ref(ctx, ctx.data_path("post_trade_ingestion_state.json"), kind="state"),
        _ref(ctx, ctx.data_path(".order_trade_state.json"), kind="state"),
    ])
    if catalog.settlement_bundle is not None:
        bundle = catalog.settlement_bundle
        inputs.append(InputRef(
            bundle.display_path, kind="data", fingerprint=bundle.fingerprint,
            required=True, description="settlement_bundle",
        ))
    else:
        inputs.extend(InputRef(
            x.display_path, kind="data", fingerprint=x.fingerprint,
            required=False, description="settlement",
        ) for x in catalog.settlement_sources)
    for stream in ("order", "trade"):
        inputs.extend(InputRef(x.display_path, kind="data", fingerprint=x.fingerprint, required=False, description=stream) for x in catalog.sources_for(stream))
    states = []
    outputs = []
    states.append(StateRef("data/.post_trade.lock", "write", "real-run process lock"))
    if options.scope in {"all", "state"}:
        states.append(StateRef("data/.post_trade_transactions/", "read_write", "recoverable account-state transaction"))
        states.append(StateRef("config/cashflow.json", "read"))
        for name in ("trade_log_full.csv", "holding_log_full.csv", "daily_amount_log_full.csv", "trade_classification.csv"):
            states.append(StateRef("data/%s" % name, "read_write"))
            outputs.append(OutputRef("data/%s" % name, kind="data", description="conditional legacy state output", overwrite=True))
        states.append(StateRef("config/prod_config.json", "read_write"))
    if options.scope in {"all", "execution"}:
        for name in ("raw_order_log_full.csv", "raw_trade_log_full.csv", "post_trade_ingestion_state.json"):
            states.append(StateRef("data/%s" % name, "read_write"))
            outputs.append(OutputRef("data/%s" % name, kind="data" if name.endswith(".csv") else "state", description="conditional execution-ingestion output", overwrite=True))
        states.append(StateRef("data/.order_trade_state.json", "write", "legacy compatibility mirror"))
    steps = [CommandStep("load-config-snapshot", "Load post-trade configuration"), CommandStep("discover-broker-sources", "Discover settlement, order, and trade evidence")]
    if options.scope in {"all", "state"}: steps.append(CommandStep("resolve-trading-calendar", "Resolve exact Qlib trading calendar", expensive=True))
    steps += [CommandStep("strict-parse-sources", "Open and validate pending broker exports", expensive=True), CommandStep("validate-cross-stream-completeness", "Check authority and execution-evidence consistency")]
    if options.scope in {"all", "execution"}:
        steps.append(CommandStep("ingest-execution-evidence", "Atomically merge raw order/trade evidence"))
    if options.scope in {"all", "state"}:
        steps.extend([
            CommandStep("inspect-unfinished-transactions", "Recover an unfinished account-state transaction before new calculation"),
            CommandStep("reconcile-execution-quantities", "Reconcile filled quantities before state update", expensive=True),
            CommandStep("load-valuation-snapshots", "Load exact holding and benchmark valuations", expensive=True),
            CommandStep("calculate-state-change-set", "Calculate deterministic multi-day account state", expensive=True),
            CommandStep("validate-state-invariants", "Validate cash, position, and valuation invariants"),
            CommandStep("stage-transaction-payloads", "Stage exact account-state and cashflow target bytes"),
            CommandStep("persist-derived-state-outputs", "Recoverably replace derived state outputs"),
            CommandStep("commit-cashflow-consumption", "Consume cashflow entries for processed dates"),
            CommandStep("commit-account-state-cursor", "Commit prod_config cursor as the last mandatory write"),
            CommandStep("run-trade-classification", "Update settlement trade classification", expensive=True),
        ])
    if not options.no_manifest:
        outputs.append(OutputRef("output/manifests/post-trade/<run-id>.json", kind="manifest", description="actual-output run audit", overwrite=True))
    if options.scope in {"all", "execution"}:
        steps.append(CommandStep("update-legacy-execution-cursor", "Best-effort compatibility cursor mirror"))
    warnings = [issue.code + ": " + issue.message for issue in catalog.issues]
    if options.scope == "state": warnings.append("state_scope_skips_execution: execution evidence is not processed")
    if options.scope == "execution": warnings.append("execution_scope_skips_state: account state is not changed")
    metadata = {
        "scope": options.scope, "broker": config.broker_name,
        "requested_date_from": catalog.date_from, "requested_date_to": catalog.date_to,
        "state_cursor": config.state_cursor, "legacy_execution_cursor": config.legacy_execution_cursor,
        "source_counts": {
            "settlement": 1 if catalog.settlement_bundle is not None else len(catalog.settlement_sources),
            "order": len(catalog.order_sources), "trade": len(catalog.trade_sources),
        },
        "source_status_counts": {
            stream: {
                status: (
                    int(status == "present")
                    if stream == "settlement" and catalog.settlement_bundle is not None
                    else sum(item.status == status for item in catalog.sources_for(stream))
                )
                for status in ("present", "missing", "assumed_empty", "already_ingested", "changed")
            }
            for stream in ("settlement", "order", "trade")
        },
        "allow_missing_settlement": options.allow_missing_settlement,
        "settlement_source_mode": "bundle" if catalog.settlement_bundle is not None else "daily",
        "settlement_bundle": ({
            "path": catalog.settlement_bundle.display_path,
            "coverage_start": catalog.settlement_bundle.coverage_start,
            "coverage_end": catalog.settlement_bundle.coverage_end,
        } if catalog.settlement_bundle is not None else None),
        "settlement_logical_partitions": {
            item.trade_date: {
                "status": (
                    "assumed_empty" if item.status == "assumed_empty" else "observed"
                ),
                "row_count": item.row_count,
                "fingerprint": item.fingerprint,
            }
            for item in catalog.settlement_sources
            if catalog.settlement_bundle is not None
        },
        "calendar_resolution": "deferred" if options.scope in {"all", "state"} else "not_required",
    }
    return CommandPlan(command="post-trade", workspace=ctx.root.name, run_id=options.run_id or generate_run_id("post_trade"), mode=options.scope, args=cli_args, inputs=tuple(inputs), outputs=tuple(outputs), states=tuple(states), steps=tuple(steps), warnings=tuple(warnings), metadata=metadata)


def prepare_post_trade_run(
    ctx: WorkspaceContext, options: PostTradeRunOptions, *, cli_args: Sequence[str] = (),
    dependencies: Optional[PostTradeCommandDependencies] = None,
) -> PreparedPostTradeRun:
    if options.run_id is None:
        options = replace(options, run_id=generate_run_id("post_trade"))
    deps = dependencies or PostTradeCommandDependencies()
    config_loader = deps.load_run_config or load_run_config
    discovery = deps.discover_inputs or discover_inputs
    config = config_loader(ctx, options)
    state_start = (datetime.strptime(config.state_cursor, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    if options.scope in {"all", "state"} and options.start_date and options.start_date < state_start:
        raise PostTradePlanError(
            "start date %s precedes the next state date %s; historical state replay is not supported"
            % (options.start_date, state_start)
        )
    # Execution discovery bootstraps all existing receipts unless the operator
    # explicitly narrows it. Settlement discovery always starts at the state
    # boundary, so normal all-scope runs never reparse historical settlements.
    execution_start = options.start_date or "1900-01-01"
    settlement_start = options.start_date or state_start
    start = execution_start if options.scope == "execution" else settlement_start
    end = options.end_date or datetime.now().strftime("%Y-%m-%d")
    if start > end: raise PostTradePlanError("start date must not be after end date")
    if options.settlement_bundle is not None and options.scope == "execution":
        raise PostTradePlanError("--settlement-bundle requires state or all scope")
    stream_starts = {
        "settlement": settlement_start,
        "order": execution_start if options.scope in {"all", "execution"} else settlement_start,
        "trade": execution_start if options.scope in {"all", "execution"} else settlement_start,
    }
    discovery_kwargs = {"stream_date_from": stream_starts}
    if options.settlement_bundle is not None:
        discovery_kwargs["settlement_bundle"] = options.settlement_bundle
    catalog = discovery(ctx, min(stream_starts.values()), end, **discovery_kwargs)
    if options.scope == "state":
        catalog = replace(catalog, order_sources=(), trade_sources=())
    elif options.scope == "execution":
        catalog = replace(catalog, settlement_sources=(), settlement_bundle=None)
    # date_from is the operator-visible requested/state boundary. Historical
    # execution bootstrap is represented by the individual source refs.
    catalog = replace(catalog, date_from=start)
    plan = _build_plan(ctx, options, config, catalog, tuple(cli_args))
    semantic_args = []
    skip_next = False
    for arg in plan.args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--run-id":
            skip_next = True
            continue
        if arg.startswith("--run-id="):
            continue
        semantic_args.append(arg)
    fingerprint = fingerprint_command_plan(replace(plan, args=tuple(semantic_args)))
    plan = replace(plan, metadata=dict(plan.metadata, plan_fingerprint=fingerprint))
    execution_options = replace(options, start_date=start, end_date=end)
    return PreparedPostTradeRun(ctx, options, execution_options, tuple(cli_args), config, catalog, plan, fingerprint)


def render_prepared(prepared: PreparedPostTradeRun) -> str:
    if prepared.options.json_plan:
        return json.dumps(prepared.plan.to_public_dict(), ensure_ascii=False, sort_keys=True)
    return render_command_plan(prepared.plan)


def execute_prepared(
    prepared: PreparedPostTradeRun, adapter: object, *,
    init_qlib: Optional[Callable[[], None]] = None,
    resolve_trade_dates: Optional[Callable[[str, str], Sequence[str]]] = None,
    state_callback: Optional[Callable[[PreparedPostTradeRun, dict], Any]] = None,
) -> PostTradeRunSummary:
    # Light discovery tolerates a damaged receipt ledger so explain-plan can
    # report sources. Strict/dry execution fails closed before any writer.
    load_ingestion_receipts(prepared.ctx.data_path("post_trade_ingestion_state.json"), strict=True)
    if prepared.options.scope in {"all", "state"} and init_qlib:
        init_qlib()
    catalog = prepared.catalog
    settlement_dates: Tuple[str, ...] = ()
    if prepared.options.scope in {"all", "state"} and resolve_trade_dates:
        state_start = prepared.options.start_date or (
            datetime.strptime(prepared.config.state_cursor, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        dates = tuple(resolve_trade_dates(state_start, catalog.date_to))
        settlement_dates = dates
        if catalog.settlement_bundle is not None:
            verify_settlement_bundle(prepared.ctx, catalog.settlement_bundle)
    parsed, catalog = parse_pending_sources_with_catalog(
        catalog, adapter,
        reparse_execution_dates=settlement_dates if prepared.options.scope == "all" else (),
    )
    if prepared.options.scope in {"all", "state"} and resolve_trade_dates:
        known = {item.trade_date for item in catalog.settlement_sources}
        if catalog.settlement_bundle is not None:
            unexpected = sorted(known - set(dates))
            if unexpected:
                raise PostTradeInputError(
                    "Settlement bundle contains dates outside the requested trading calendar: %s"
                    % ", ".join(unexpected)
                )
        missing = [date for date in dates if date not in known]
        if missing and not prepared.options.allow_missing_settlement:
            raise PostTradeInputMissingError("Missing settlement evidence for: %s" % ", ".join(missing))
        assumed = tuple(PostTradeSourceRef(
            stream="settlement", trade_date=date,
            path=(
                catalog.settlement_bundle.path if catalog.settlement_bundle is not None
                else prepared.ctx.data_path("%s-table.xlsx" % date)
            ),
            display_path=(
                catalog.settlement_bundle.display_path if catalog.settlement_bundle is not None
                else "data/%s-table.xlsx" % date
            ),
            status="assumed_empty",
            source_kind="assumed_empty", row_count=0,
        ) for date in missing)
        assumed_issues = tuple(PostTradeIntakeIssue(
            code="assumed_empty_settlement", severity="warning",
            message="Missing settlement explicitly treated as no activity.",
            stream="settlement", trade_date=date,
        ) for date in missing)
        catalog = replace(
            catalog,
            settlement_sources=tuple(sorted(
                tuple(catalog.settlement_sources) + assumed,
                key=lambda source: source.trade_date,
            )),
            issues=tuple(catalog.issues) + assumed_issues,
        )
        for source in assumed:
            parsed[("settlement", source.trade_date)] = ParsedPostTradeInput(
                source, pd.DataFrame(), 0,
            )
    validate_cross_stream(
        parsed, scope=prepared.options.scope,
        settlement_required_dates=settlement_dates,
    )
    effective_prepared = prepared
    if catalog is not prepared.catalog:
        effective_plan = _build_plan(
            prepared.ctx, prepared.options, prepared.config, catalog, prepared.cli_args,
        )
        effective_plan = replace(
            effective_plan,
            metadata=dict(effective_plan.metadata, plan_fingerprint=prepared.plan_fingerprint),
        )
        effective_prepared = replace(prepared, catalog=catalog, plan=effective_plan)
    verify_parsed_sources(parsed)
    if catalog.settlement_bundle is not None:
        verify_settlement_bundle(prepared.ctx, catalog.settlement_bundle)
    if prepared.options.dry_run:
        state_result = None
        if prepared.options.scope in {"all", "state"} and state_callback:
            state_result = state_callback(effective_prepared, parsed)
        return PostTradeRunSummary(effective_prepared, state_result=state_result, parsed_inputs=parsed)
    ingestion = None
    if prepared.options.scope in {"all", "execution"}:
        ingestion = ingest_execution_evidence(prepared.ctx, parsed, run_id=prepared.options.run_id or prepared.plan_fingerprint)
    state_result = None
    if prepared.options.scope in {"all", "state"} and state_callback:
        try:
            state_result = state_callback(effective_prepared, parsed)
        except Exception as exc:
            from quantpits.post_trade.contracts import PostTradePartialExecutionError
            partial = PostTradeRunSummary(effective_prepared, ingestion, None, parsed)
            raise PostTradePartialExecutionError(
                "Post-trade state execution failed after intake: %s" % exc,
                summary=partial,
                cause=exc,
            ) from exc
    return PostTradeRunSummary(effective_prepared, ingestion, state_result, parsed)
