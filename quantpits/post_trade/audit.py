"""Privacy-safe post-trade RunManifest helpers."""

from __future__ import annotations

from datetime import datetime

from quantpits.runtime import OutputRef, RunManifest, manifest_path, write_run_manifest


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
        for path in journal.classification.output_paths:
            outputs.append(OutputRef(path, kind="data", overwrite=True))
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
        "settlement_source_count": len(prepared.catalog.settlement_sources),
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
