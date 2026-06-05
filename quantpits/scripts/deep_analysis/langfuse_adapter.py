"""
Langfuse Adapter for LLM Trace Logger.

Pushes LLMTraceRecord objects to a Langfuse server as Observations
(generations) nested inside Traces + Spans.

This module has a *soft* dependency on ``langfuse``.  When the package is
not installed, every method is a no-op and a warning is logged exactly once.

Langfuse hierarchy mapping:
    Run (run_id)         → Langfuse Trace
    Session (session_id) → Langfuse Span  (inside the Trace)
    Trace (trace_id)     → Langfuse Generation (inside the Span)

Usage::

    adapter = LangfuseAdapter.from_config(llm_config, ws_config)
    trace_logger = LLMTraceLogger(..., langfuse_adapter=adapter)
    # … then use trace_logger normally; adapter is called automatically

Configuration in llm_config.json::

    {
        "trace": {
            "langfuse": {
                "enabled": true,
                "host": "http://localhost:3000",
                "public_key_env": "LANGFUSE_PUBLIC_KEY",
                "secret_key_env": "LANGFUSE_SECRET_KEY"
            }
        }
    }
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_trace import LLMTraceRecord, SessionContext

logger = logging.getLogger(__name__)

# Sentinel so we warn only once per process if langfuse is absent
_LANGFUSE_IMPORT_WARNED = False


def _try_import_langfuse():
    """Return the ``langfuse`` module or None if not installed."""
    global _LANGFUSE_IMPORT_WARNED
    try:
        import langfuse as _lf
        return _lf
    except ImportError:
        if not _LANGFUSE_IMPORT_WARNED:
            logger.warning(
                "langfuse package not installed — Langfuse adapter is disabled. "
                "Install with: pip install langfuse"
            )
            _LANGFUSE_IMPORT_WARNED = True
        return None


class LangfuseAdapter:
    """Pushes LLM trace records to a Langfuse server.

    Thread-safe: the Langfuse SDK handles its own async queue internally.
    All public methods silently degrade to no-ops if langfuse is unavailable
    or if ``enabled=False``.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "http://localhost:3000",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._client: Optional[Any] = None
        # Keyed by run_id → Langfuse Trace object
        self._traces: dict = {}
        # Keyed by session_id → Langfuse Span object
        self._spans: dict = {}

        if not enabled:
            return

        lf = _try_import_langfuse()
        if lf is None:
            self._enabled = False
            return

        if not public_key or not secret_key:
            logger.warning("LangfuseAdapter: public_key or secret_key missing — adapter disabled.")
            self._enabled = False
            return

        try:
            self._client = lf.Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("LangfuseAdapter: connected to %s", host)
        except Exception as e:
            logger.warning("LangfuseAdapter: failed to initialise client: %s — adapter disabled.", e)
            self._enabled = False

    # -----------------------------------------------------------------------
    # Public API (called by LLMTraceLogger)
    # -----------------------------------------------------------------------

    def on_trace(
        self,
        record: "LLMTraceRecord",
        session_ctx: Optional["SessionContext"],
    ) -> None:
        """Push one API call to Langfuse as a Generation observation."""
        if not self._enabled or self._client is None:
            return
        try:
            self._push_generation(record, session_ctx)
        except Exception as e:
            logger.debug("LangfuseAdapter.on_trace failed: %s", e)

    def on_session_end(self, session_ctx: "SessionContext") -> None:
        """Close the Langfuse Span for this session."""
        if not self._enabled or self._client is None:
            return
        try:
            span = self._spans.get(session_ctx.session_id)
            if span is not None:
                span.end()
        except Exception as e:
            logger.debug("LangfuseAdapter.on_session_end failed: %s", e)

    def flush(self) -> None:
        """Flush all pending async Langfuse events."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception as e:
            logger.debug("LangfuseAdapter.flush failed: %s", e)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _ensure_trace(self, run_id: str, run_date: str, workspace: str) -> Any:
        """Get or create the Langfuse Trace for a run."""
        if run_id not in self._traces:
            self._traces[run_id] = self._client.trace(
                id=run_id,
                name=f"deep_analysis_{run_date}",
                metadata={"workspace": workspace, "run_date": run_date},
            )
        return self._traces[run_id]

    def _ensure_span(
        self,
        session_ctx: "SessionContext",
        lf_trace: Any,
    ) -> Any:
        """Get or create the Langfuse Span for a session."""
        sid = session_ctx.session_id
        if sid not in self._spans:
            self._spans[sid] = lf_trace.span(
                id=sid,
                name=session_ctx.label or session_ctx.session_type,
                metadata={
                    "session_type": session_ctx.session_type,
                    "target": session_ctx.target,
                },
            )
        return self._spans[sid]

    def _push_generation(
        self,
        record: "LLMTraceRecord",
        session_ctx: Optional["SessionContext"],
    ) -> None:
        lf_trace = self._ensure_trace(
            record.run_id or "unknown_run",
            record.run_date or "",
            record.workspace or "",
        )

        parent = None
        if session_ctx is not None:
            parent = self._ensure_span(session_ctx, lf_trace)

        # Build input / output for Langfuse
        # Input = messages list (prompt); output = response_content
        lf_input = record.messages
        lf_output = record.response_content
        if record.reasoning_content:
            # Surface reasoning as metadata so it's visible in the Langfuse UI
            lf_output = {
                "content": record.response_content,
                "reasoning": record.reasoning_content,
            }

        kwargs = dict(
            id=record.trace_id,
            name=record.label or record.session_type,
            model=record.model_responded or record.model_requested,
            model_parameters={
                "temperature": record.temperature,
                "max_tokens": record.max_tokens,
                **record.other_params,
            },
            input=lf_input,
            output=lf_output,
            usage={
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
            },
            metadata={
                "session_type": record.session_type,
                "round_number": record.round_number,
                "round_phase": record.round_phase,
                "participant_id": record.participant_id,
                "participant_role": record.participant_role,
                "finish_reason": record.finish_reason,
                "duration_ms": record.duration_ms,
                "reasoning_tokens": record.reasoning_tokens,
                "base_url": record.base_url,
                "pipeline_stage": record.pipeline_stage,
                "success": record.success,
                "error": record.error,
                "retry_attempt": record.retry_attempt,
                "system_prompt_hash": record.system_prompt_hash,
            },
            level="ERROR" if not record.success else "DEFAULT",
        )

        if parent is not None:
            parent.generation(**kwargs)
        else:
            lf_trace.generation(**kwargs)

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        llm_config: dict,
        ws_config: Optional[dict] = None,
    ) -> Optional["LangfuseAdapter"]:
        """Create from parsed llm_config.json.  Returns None if not configured.

        Args:
            llm_config: Parsed llm_config.json dict.
            ws_config: Unused; kept for API consistency with other adapters.
        """
        trace_cfg = llm_config.get("trace", {})
        lf_cfg = trace_cfg.get("langfuse", {})
        if not lf_cfg.get("enabled", False):
            return None

        host = lf_cfg.get("host", "http://localhost:3000")

        pub_env = lf_cfg.get("public_key_env", "LANGFUSE_PUBLIC_KEY")
        sec_env = lf_cfg.get("secret_key_env", "LANGFUSE_SECRET_KEY")
        public_key = os.environ.get(pub_env)
        secret_key = os.environ.get(sec_env)

        if not public_key or not secret_key:
            logger.warning(
                "LangfuseAdapter: env vars %s / %s not set — adapter disabled.",
                pub_env, sec_env,
            )
            return None

        return cls(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            enabled=True,
        )
