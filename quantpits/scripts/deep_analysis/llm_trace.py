"""
LLM Trace Logger for the MAS Deep Analysis System.

Records every LLM API call as a self-contained JSON file under a dated run
directory, with a manifest.json index generated at the end of each run.

Directory layout:
    output/deep_analysis/llm_traces/
      {date}_run_{run_id_short}/
        manifest.json
        triage/
          001_triage_{model_short}.json
        per_model/
          {model_name}/
            002_r1_{participant}_{phase}.json   # multi-LLM
          002_alstm_Alpha158_{model_short}.json  # single-LLM
        per_combo/
          003_combo_v2_{model_short}.json
        execution_risk/
          004_execution_risk_{model_short}.json
        synthesizer/
          005_synthesizer_{model_short}.json
        summary/
          006_summary_{model_short}.json

Hierarchy (all 4 levels — current single-LLM uses only first 3):
    Run        — one full deep_analysis execution
    Session    — one logical operation (triage, per_model:X, synthesizer, …)
    Round      — one dialogue round (single-LLM: always round 1)
    Trace      — one API call (one model's request + response)

Multi-LLM debate mode is supported via round_number / round_phase /
participant_id / input_trace_ids fields — no structural changes needed when
that mode is added later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LLMTraceRecord:
    """Complete record of a single LLM API call.

    All content fields are stored verbatim (no truncation).  The caller is
    responsible for passing the raw API objects; this class only serialises.
    """

    # --- Hierarchy identifiers ---
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    session_id: str = ""
    session_type: str = ""   # "triage" | "per_model_critic" | "per_combo_critic" |
                             # "execution_risk" | "synthesizer" | "summary" |
                             # "experiment_advisor" | "critic"
    operation: str = ""

    # --- Multi-LLM fields (single-LLM defaults) ---
    round_number: int = 1
    round_phase: str = "sole"             # "sole" | "independent_opinion" | "debate" |
                                          # "rebuttal" | "consensus" | "arbitration"
    participant_id: str = "primary"       # single-LLM: "primary"
    participant_role: str = "sole"        # "critic" | "challenger" | "arbiter" | "sole"
    input_trace_ids: List[str] = field(default_factory=list)  # DAG edges

    # --- Timing ---
    timestamp: str = ""           # ISO 8601 set by logger
    duration_ms: int = 0

    # --- Human-readable label ---
    label: str = ""               # e.g. "Per-Model(alstm_Alpha158)"

    # --- Model / endpoint ---
    model_requested: str = ""
    model_responded: str = ""     # from response.model (may differ)
    base_url: str = ""

    # --- Request parameters ---
    temperature: float = 0.3
    max_tokens: int = 0
    other_params: Dict[str, Any] = field(default_factory=dict)

    # --- Request content (verbatim) ---
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt_hash: str = ""  # SHA-256 of system message content

    # --- Response content (verbatim, no truncation) ---
    response_content: str = ""
    reasoning_content: str = ""   # DeepSeek / Claude extended thinking
    finish_reason: str = ""

    # --- Token usage ---
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0     # separate reasoning budget (DeepSeek)
    cached_tokens: int = 0        # prompt cache hit tokens

    # --- Status ---
    success: bool = True
    error: str = ""
    retry_attempt: int = 0

    # --- Run context ---
    workspace: str = ""
    run_date: str = ""
    pipeline_stage: str = "single_stage"  # "single_stage" | "layered"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def compute_system_prompt_hash(self) -> None:
        """Compute and set system_prompt_hash from the first system message."""
        for msg in self.messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    self.system_prompt_hash = self._sha256(content)
                    return
        self.system_prompt_hash = ""


# ---------------------------------------------------------------------------
# Session context
# ---------------------------------------------------------------------------

class SessionContext:
    """Tracks a logical session (one LLM operation) across rounds.

    Thread-safe: a single session may be used from one thread only in the
    current single-LLM mode; the class is safe to share across threads for
    future multi-LLM use.
    """

    def __init__(
        self,
        session_id: str,
        session_type: str,
        label: str,
        target: str = "",
        run_id: str = "",
    ) -> None:
        self.session_id = session_id
        self.session_type = session_type
        self.label = label
        self.target = target          # e.g. model name for per_model
        self.run_id = run_id

        self._lock = threading.Lock()
        self._current_round: int = 0
        self._current_phase: str = "sole"
        self._current_participant: str = "primary"
        self._current_role: str = "sole"

        self.started_at: float = time.monotonic()
        self.trace_files: List[str] = []   # absolute paths, appended by logger
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_reasoning_tokens: int = 0
        self.total_duration_ms: int = 0
        self.trace_count: int = 0

    # -- Round / participant management (multi-LLM) --

    def new_round(self, phase: str = "sole") -> int:
        """Start a new round.  Returns the 1-based round number."""
        with self._lock:
            self._current_round += 1
            self._current_phase = phase
            self._current_participant = "primary"
            self._current_role = "sole"
            return self._current_round

    def set_participant(self, participant_id: str, role: str = "sole") -> None:
        with self._lock:
            self._current_participant = participant_id
            self._current_role = role

    @property
    def current_round(self) -> int:
        with self._lock:
            return max(self._current_round, 1)

    @property
    def current_phase(self) -> str:
        with self._lock:
            return self._current_phase

    @property
    def current_participant(self) -> str:
        with self._lock:
            return self._current_participant

    @property
    def current_role(self) -> str:
        with self._lock:
            return self._current_role

    def _record_trace(self, record: LLMTraceRecord, path: str) -> None:
        """Called by LLMTraceLogger after writing a trace file."""
        with self._lock:
            self.trace_files.append(path)
            self.total_prompt_tokens += record.prompt_tokens
            self.total_completion_tokens += record.completion_tokens
            self.total_reasoning_tokens += record.reasoning_tokens
            self.total_duration_ms += record.duration_ms
            self.trace_count += 1

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

#: Map from session_type to a short subdirectory name.
_SESSION_SUBDIR: Dict[str, str] = {
    "triage": "triage",
    "per_model_critic": "per_model",
    "per_combo_critic": "per_combo",
    "execution_risk": "execution_risk",
    "synthesizer": "synthesizer",
    "summary": "summary",
    "experiment_advisor": "experiment_advisor",
    "critic": "critic",
}


def _model_short(model_name: str, max_len: int = 16) -> str:
    """Abbreviate a model name for use in filenames."""
    # Strip common provider prefixes
    for prefix in ("deepseek-", "gpt-", "claude-", "gemini-"):
        if model_name.lower().startswith(prefix):
            model_name = model_name[len(prefix):]
            break
    # Replace dots/slashes/spaces with underscores
    model_name = model_name.replace("/", "_").replace(".", "_").replace(" ", "_")
    return model_name[:max_len]


class LLMTraceLogger:
    """Records every LLM API call as an individual JSON file inside a
    dated run directory.

    Usage (single-LLM mode — the session context manager handles round 1
    automatically)::

        trace_logger = LLMTraceLogger(
            output_dir="output/deep_analysis/llm_traces",
            run_date="2026-06-05",
            workspace="Example_Workspace",
            pipeline_stage="layered",
        )

        with trace_logger.session("per_model_critic", label="Per-Model(foo)", target="foo") as sess:
            # … build prompts …
            # call _call_llm(), which internally calls log_trace()
            pass

        trace_logger.finalize()   # writes manifest.json

    Thread safety: individual trace writes are protected by a lock.  Multiple
    threads may call log_trace() concurrently (as happens during the parallel
    per-model phase).
    """

    def __init__(
        self,
        output_dir: str,
        run_id: Optional[str] = None,
        run_date: Optional[str] = None,
        workspace: str = "",
        pipeline_stage: str = "single_stage",
        run_label: str = "",
        langfuse_adapter: Optional[Any] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            output_dir: Base directory for all traces (absolute or relative to cwd).
            run_id: Unique run identifier; auto-generated (8 hex chars) if not given.
            run_date: Date string for run directory prefix (YYYY-MM-DD); today if not given.
            workspace: Workspace name embedded in each trace record.
            pipeline_stage: "single_stage" | "layered".
            run_label: Optional human-readable label injected into run_dir name.
            langfuse_adapter: Optional LangfuseAdapter; no-op when None.
            enabled: Set to False to disable all logging (no-op mode).
        """
        self.enabled = enabled
        self.workspace = workspace
        self.pipeline_stage = pipeline_stage
        self.run_label = run_label
        self._langfuse = langfuse_adapter

        if not enabled:
            self.run_id = run_id or str(uuid.uuid4()).replace("-", "")[:8]
            self.run_date = run_date or datetime.now().strftime("%Y-%m-%d")
            self.run_dir = ""
            self._sessions: List[SessionContext] = []
            self._lock = threading.Lock()
            self._seq = 0
            return

        self.run_id = run_id or str(uuid.uuid4()).replace("-", "")[:8]
        self.run_date = run_date or datetime.now().strftime("%Y-%m-%d")
        _label_segment = self.run_date
        if self.run_label:
            _label_segment += f"_{self.run_label}"
        self.run_dir = os.path.join(
            output_dir,
            f"{_label_segment}_run_{self.run_id}",
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self._sessions: List[SessionContext] = []
        self._lock = threading.Lock()   # guards _seq and _sessions
        self._seq: int = 0              # global per-run sequence counter

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    @contextmanager
    def session(
        self,
        session_type: str,
        label: str = "",
        target: str = "",
    ) -> Generator[SessionContext, None, None]:
        """Context manager that creates and registers a SessionContext.

        The session is yielded, and its elapsed time is recorded on exit.
        One round (round=1, phase="sole") is started automatically so
        single-LLM callers don't need to call new_round() explicitly.

        Example::

            with trace_logger.session("per_model_critic", target="alstm") as sess:
                result = self._call_llm(..., session_ctx=sess)
        """
        if not self.enabled:
            # Yield a dummy context so callers don't need guards
            sess = SessionContext(
                session_id=str(uuid.uuid4()),
                session_type=session_type,
                label=label or session_type,
                target=target,
                run_id=self.run_id,
            )
            sess.new_round("sole")
            yield sess
            return

        sess = SessionContext(
            session_id=str(uuid.uuid4()),
            session_type=session_type,
            label=label or session_type,
            target=target,
            run_id=self.run_id,
        )
        # Pre-start round 1 for single-LLM callers
        sess.new_round("sole")

        with self._lock:
            self._sessions.append(sess)

        try:
            yield sess
        finally:
            # Notify Langfuse adapter that this session is done
            if self._langfuse is not None:
                try:
                    self._langfuse.on_session_end(sess)
                except Exception:
                    pass

    # -----------------------------------------------------------------------
    # Trace writing
    # -----------------------------------------------------------------------

    def log_trace(
        self,
        record: LLMTraceRecord,
        session_ctx: Optional[SessionContext] = None,
    ) -> str:
        """Write one trace record to a JSON file.

        Returns the absolute path of the written file, or "" when disabled.
        Thread-safe.
        """
        if not self.enabled:
            return ""

        # Fill in run-level fields if not already set
        record.run_id = record.run_id or self.run_id
        record.run_date = record.run_date or self.run_date
        record.workspace = record.workspace or self.workspace
        record.pipeline_stage = record.pipeline_stage or self.pipeline_stage

        if session_ctx is not None:
            record.session_id = session_ctx.session_id
            record.session_type = session_ctx.session_type
            if not record.label:
                record.label = session_ctx.label
            if not record.round_number or record.round_number < 1:
                record.round_number = session_ctx.current_round
            if not record.round_phase or record.round_phase == "sole":
                record.round_phase = session_ctx.current_phase
            if not record.participant_id or record.participant_id == "primary":
                record.participant_id = session_ctx.current_participant
            if not record.participant_role or record.participant_role == "sole":
                record.participant_role = session_ctx.current_role

        # Compute system prompt hash
        record.compute_system_prompt_hash()

        # Determine subdirectory
        subdir_name = _SESSION_SUBDIR.get(record.session_type, "other")
        # For per-model, nest under the target name when available
        if record.session_type == "per_model_critic" and session_ctx and session_ctx.target:
            subdir = os.path.join(self.run_dir, subdir_name, session_ctx.target)
        else:
            subdir = os.path.join(self.run_dir, subdir_name)
        os.makedirs(subdir, exist_ok=True)

        # Build filename
        with self._lock:
            self._seq += 1
            seq = self._seq

        model_tag = _model_short(record.model_requested or record.model_responded or "unknown")
        if record.round_phase == "sole" or not record.round_phase:
            # Single-LLM simple name
            sess_tag = (session_ctx.target if session_ctx and session_ctx.target
                        else record.session_type)
            # Shorten target for filename safety
            sess_tag = sess_tag.replace("/", "_").replace(" ", "_")[:32]
            filename = f"{seq:03d}_{sess_tag}_{model_tag}.json"
        else:
            # Multi-LLM: encode round + participant + phase
            participant_tag = (record.participant_id or "p")[:12].replace("/", "_")
            filename = (
                f"{seq:03d}_r{record.round_number}_{participant_tag}"
                f"_{record.round_phase[:12]}_{model_tag}.json"
            )

        filepath = os.path.join(subdir, filename)

        # Serialise — use default=str to handle any non-serialisable values
        try:
            payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("LLMTraceLogger: JSON serialisation failed: %s", e)
            payload = json.dumps({
                "trace_id": record.trace_id,
                "error": f"Serialisation failed: {e}",
                "session_type": record.session_type,
            }, indent=2)

        try:
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(payload)
        except Exception as e:
            logger.warning("LLMTraceLogger: failed to write %s: %s", filepath, e)
            return ""

        # Update session stats
        if session_ctx is not None:
            session_ctx._record_trace(record, filepath)

        # Notify Langfuse adapter
        if self._langfuse is not None:
            try:
                self._langfuse.on_trace(record, session_ctx)
            except Exception:
                pass

        return filepath

    # -----------------------------------------------------------------------
    # Run finalisation
    # -----------------------------------------------------------------------

    def get_run_summary(self) -> dict:
        """Return an in-memory summary of the current run's statistics."""
        total_prompt = total_completion = total_reasoning = 0
        total_traces = total_success = total_failed = 0
        total_duration = 0
        models_used: set = set()

        sessions_info = []
        for sess in self._sessions:
            total_prompt += sess.total_prompt_tokens
            total_completion += sess.total_completion_tokens
            total_reasoning += sess.total_reasoning_tokens
            total_traces += sess.trace_count
            total_duration += sess.total_duration_ms

            sessions_info.append({
                "session_id": sess.session_id,
                "session_type": sess.session_type,
                "label": sess.label,
                "target": sess.target,
                "traces": sess.trace_files,
                "trace_count": sess.trace_count,
                "duration_ms": sess.elapsed_ms(),
                "tokens": {
                    "prompt": sess.total_prompt_tokens,
                    "completion": sess.total_completion_tokens,
                    "reasoning": sess.total_reasoning_tokens,
                },
            })

        return {
            "run_id": self.run_id,
            "run_date": self.run_date,
            "run_label": self.run_label,
            "workspace": self.workspace,
            "pipeline_stage": self.pipeline_stage,
            "summary": {
                "total_traces": total_traces,
                "total_sessions": len(self._sessions),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_reasoning_tokens": total_reasoning,
                "total_duration_ms": total_duration,
                "models_used": sorted(models_used),
            },
            "sessions": sessions_info,
        }

    def finalize(self) -> str:
        """Write manifest.json and flush Langfuse.  Returns the run directory path."""
        if not self.enabled:
            return ""

        manifest_data = self.get_run_summary()
        manifest_data["finalized_at"] = datetime.now().isoformat()

        # Build timeline from session trace files
        timeline = []
        for sess_info in manifest_data["sessions"]:
            for trace_path in sess_info.get("traces", []):
                # Read back just the header fields to build timeline
                try:
                    with open(trace_path, "r", encoding="utf-8") as fh:
                        rec = json.load(fh)
                    timeline.append({
                        "seq": os.path.basename(trace_path)[:3],
                        "trace_id": rec.get("trace_id", ""),
                        "session_type": rec.get("session_type", ""),
                        "label": rec.get("label", ""),
                        "model": rec.get("model_requested", ""),
                        "timestamp": rec.get("timestamp", ""),
                        "duration_ms": rec.get("duration_ms", 0),
                        "total_tokens": rec.get("total_tokens", 0),
                        "success": rec.get("success", True),
                        "file": os.path.relpath(trace_path, self.run_dir),
                    })
                except Exception:
                    pass

        manifest_data["timeline"] = timeline

        manifest_path = os.path.join(self.run_dir, "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest_data, fh, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("LLMTraceLogger: failed to write manifest: %s", e)

        # Flush Langfuse async queue
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception:
                pass

        return self.run_dir

    # -----------------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_llm_config(
        cls,
        llm_config: dict,
        workspace_root: str,
        run_date: Optional[str] = None,
        workspace_name: str = "",
        pipeline_stage: str = "single_stage",
        langfuse_adapter: Optional[Any] = None,
        run_label: str = "",
    ) -> "LLMTraceLogger":
        """Create a LLMTraceLogger from a parsed llm_config.json dict.

        The trace config block is::

            {
                "trace": {
                    "enabled": true,
                    "output_dir": "output/deep_analysis/llm_traces"
                }
            }

        When ``trace.enabled`` is False or the key is absent, returns a
        disabled logger (all operations are no-ops).
        """
        trace_cfg = llm_config.get("trace", {})
        enabled = trace_cfg.get("enabled", True)

        if not enabled:
            logger.info("LLMTraceLogger: tracing disabled by config")
            return cls(output_dir="", enabled=False, run_date=run_date,
                       workspace=workspace_name, pipeline_stage=pipeline_stage,
                       run_label=run_label)

        rel_output_dir = trace_cfg.get("output_dir", "output/deep_analysis/llm_traces")
        output_dir = os.path.join(workspace_root, rel_output_dir)

        return cls(
            output_dir=output_dir,
            run_date=run_date,
            workspace=workspace_name,
            pipeline_stage=pipeline_stage,
            langfuse_adapter=langfuse_adapter,
            run_label=run_label,
            enabled=True,
        )
