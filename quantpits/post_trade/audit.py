"""Privacy-safe post-trade RunManifest helpers."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping, Tuple

from quantpits.runtime import OutputRef, RunManifest, manifest_path, write_run_manifest
from quantpits.post_trade.contracts import settlement_logical_partitions


def legacy_assumed_empty_settlement_inputs(manifest: Mapping[str, Any]) -> Tuple[str, ...]:
    """Interpret the old committed state audit, without consulting live state.

    Old daily audits included resolved empty days as optional, unfingerprinted
    settlement inputs and omitted their logical partitions. Optionality alone
    is insufficient: require the successful state-only processing scope and
    the entire original daily source range to agree.
    """
    records = manifest.get("records")
    args = manifest.get("args")
    inputs = manifest.get("inputs")
    if (
        type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
        or manifest.get("command") not in {"post-trade", "prod_post_trade"}
        or manifest.get("status") != "success"
        or type(records) is not dict or type(args) is not list
        or any(type(arg) is not str for arg in args)
        or "--allow-missing-settlement" not in args
        or "--dry-run" in args
        or any(arg == "--settlement-bundle" or arg.startswith("--settlement-bundle=") for arg in args)
        or type(inputs) is not list or any(type(row) is not dict for row in inputs)
        or records.get("scope") != "state" or records.get("state_committed") is not True
        or records.get("transaction_status") not in {"state_committed", "completed"}
        or records.get("settlement_source_mode") != "daily"
        or records.get("settlement_bundle_coverage") is not None
        or records.get("settlement_logical_partitions") != {}
        or "settlement_logical_day_count" in records
        or records.get("execution_ingestion_status") != "not_run"
        or any(type(records.get(key)) is not int or records[key] != 0
               for key in ("order_source_count", "trade_source_count", "ingested_source_count"))
        or any(row.get("description") in {"order", "trade", "settlement_bundle"} for row in inputs)
    ):
        return ()
    refs = [row for row in inputs if row.get("description") == "settlement"]
    dates, empty_paths = [], []
    for row in refs:
        path = row.get("path")
        match = re.fullmatch(r"data/(\d{4}-\d{2}-\d{2})-table\.xlsx", path) if type(path) is str else None
        if match is None or row.get("kind") != "data" or row.get("required") is not False:
            return ()
        day = match.group(1)
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            return ()
        dates.append(day)
        fingerprint = row.get("fingerprint")
        if fingerprint is None:
            empty_paths.append(path)
        elif type(fingerprint) is not str or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None:
            return ()
    if (
        not empty_paths or len(set(dates)) != len(dates)
        or any(type(records.get(key)) is not int or records[key] != len(dates)
               for key in ("processed_date_count", "settlement_source_count"))
        or records.get("processed_date_from") != min(dates)
        or records.get("processed_date_to") != max(dates)
    ):
        return ()
    outputs = manifest.get("outputs", [])
    if type(outputs) is not list or any(type(row) is not dict for row in outputs):
        return ()
    for path in empty_paths:
        if (sum(row.get("path") == path for row in inputs) != 1
                or any(row.get("path") == path for row in outputs)):
            return ()
    return tuple(empty_paths)


def display_path(ctx, path):
    return path.resolve().relative_to(ctx.root).as_posix()


def redact_error(ctx, exc):
    """Return a compact error record without workspace-local absolute paths."""
    message = str(exc).replace(ctx.root.as_posix(), "<workspace>")
    return {"type": type(exc).__name__, "message": message[:1000]}


def write_post_trade_manifest(
    prepared, summary, *, started_at, status="success", error=None, warnings=(),
    journal=None, linked_transaction_id=None, actual_state_paths=None,
):
    outputs = []
    actual_state_set = set(actual_state_paths) if actual_state_paths is not None else None
    ingestion = summary.ingestion
    if ingestion is not None:
        for path in getattr(ingestion, "outputs", ()):
            shown = path if isinstance(path, str) else display_path(prepared.ctx, path)
            outputs.append(OutputRef(shown, kind="data", overwrite=True))
    state = summary.state_result
    if state is not None and state.artifacts is not None:
        if state.artifacts.transaction_id:
            from quantpits.post_trade.transaction import PostTradeTransactionManager
            journal = PostTradeTransactionManager(prepared.ctx).load(state.artifacts.transaction_id)
        for path in state.artifacts.outputs:
            shown = display_path(prepared.ctx, path)
            outputs.append(OutputRef(shown, kind="state", overwrite=True))
    elif journal is not None:
        allowed = actual_state_set
        for artifact in journal.artifacts:
            if allowed is None or artifact.path in allowed:
                outputs.append(OutputRef(artifact.path, kind="state", overwrite=True))
    if journal is not None:
        existing = {item.path for item in outputs}
        for path in journal.classification.output_paths:
            if path not in existing:
                outputs.append(OutputRef(path, kind="data", description="trade classification", overwrite=True))
                existing.add(path)
    manifest_file = manifest_path(prepared.ctx, "post-trade", prepared.plan.run_id)
    outputs.append(OutputRef(display_path(prepared.ctx, manifest_file), kind="manifest", overwrite=True))
    records = {
        "scope": prepared.options.scope,
        "light_plan_fingerprint": prepared.plan_fingerprint,
        "transaction_id": journal.transaction_id if journal else None,
        "linked_transaction_id": linked_transaction_id,
        "transaction_status": journal.status if journal else None,
        "resolved_execution_fingerprint": journal.resolved_execution_fingerprint if journal else None,
        "processed_date_count": len(journal.processed_dates) if journal else 0,
        "processed_date_from": journal.processed_dates[0] if journal and journal.processed_dates else None,
        "processed_date_to": journal.processed_dates[-1] if journal and journal.processed_dates else None,
        "settlement_source_count": (
            1 if prepared.catalog.settlement_bundle is not None
            else sum(item.status != "assumed_empty" for item in prepared.catalog.settlement_sources)
        ),
        "settlement_logical_day_count": len(prepared.catalog.settlement_sources),
        "settlement_source_mode": (
            "bundle" if prepared.catalog.settlement_bundle is not None else "daily"
        ),
        "settlement_bundle_coverage": ({
            "path": prepared.catalog.settlement_bundle.display_path,
            "start": prepared.catalog.settlement_bundle.coverage_start,
            "end": prepared.catalog.settlement_bundle.coverage_end,
        } if prepared.catalog.settlement_bundle is not None else None),
        "settlement_logical_partitions": settlement_logical_partitions(prepared.catalog),
        "order_source_count": len(prepared.catalog.order_sources),
        "trade_source_count": len(prepared.catalog.trade_sources),
        "ingested_source_count": len(getattr(ingestion, "ingested_sources", ())) if ingestion else 0,
        "state_artifact_count": len(journal.artifacts) if journal else 0,
        "consumed_cashflow_date_count": len(journal.consumed_cashflow_dates) if journal else 0,
        "execution_ingestion_status": "committed" if ingestion is not None else "not_run",
        "state_committed": bool(journal and journal.status in {"state_committed", "completed"}),
        "classification_status": journal.classification.status if journal else "not_applicable",
        "classification_attempts": journal.classification.attempts if journal else 0,
        "output_fingerprints": {
            artifact.path: artifact.target_sha256
            for artifact in journal.artifacts
            if actual_state_set is None or artifact.path in actual_state_set
        } if journal else {},
        "classification_output_fingerprints": (
            dict(journal.classification.output_fingerprints) if journal else {}
        ),
        "actual_output_count": len(outputs) - 1,
    }
    manifest = RunManifest(
        run_id=prepared.plan.run_id, command="post-trade", workspace=prepared.ctx.root.name,
        started_at=started_at, finished_at=datetime.now().isoformat(), status=status,
        args=prepared.plan.args, inputs=prepared.plan.inputs, outputs=tuple(outputs),
        states=prepared.plan.states, steps=prepared.plan.steps,
        config_fingerprints=prepared.plan.config_fingerprints, records=records,
        warnings=tuple(prepared.plan.warnings) + tuple(warnings), error=error,
    )
    return write_run_manifest(prepared.ctx, manifest), journal
