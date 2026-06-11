"""
Tests for LangfuseAdapter — the optional Langfuse LLM trace push adapter.

All tests mock the ``langfuse`` package via sys.modules injection so no real
Langfuse server is needed.  Covers the soft-dependency disable paths, trace/span
caching, generation push, and the from_config factory.
"""

import logging
import os
import sys
from unittest import mock

import pytest

# Always import the adapter (no langfuse needed at import time).
from quantpits.scripts.deep_analysis.langfuse_adapter import (
    LangfuseAdapter,
    _try_import_langfuse,
    _LANGFUSE_IMPORT_WARNED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides):
    """Build a minimal LLMTraceRecord-like object for push_generation tests."""
    defaults = dict(
        trace_id="trace-1",
        run_id="run-1",
        run_date="2025-01-01",
        workspace="test_ws",
        session_type="test_session",
        label="test_label",
        model_requested="gpt-4",
        model_responded="gpt-4",
        temperature=0.7,
        max_tokens=2000,
        other_params={},
        messages=[{"role": "user", "content": "hello"}],
        response_content="world",
        reasoning_content=None,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        reasoning_tokens=0,
        round_number=1,
        round_phase="analysis",
        participant_id="p1",
        participant_role="analyst",
        finish_reason="stop",
        duration_ms=500,
        base_url="https://api.openai.com",
        pipeline_stage="deep_analysis",
        success=True,
        error=None,
        retry_attempt=0,
        system_prompt_hash="abc123",
    )
    defaults.update(overrides)
    return mock.MagicMock(**defaults)


def _make_session_ctx(**overrides):
    """Build a minimal SessionContext-like object."""
    defaults = dict(
        session_id="sess-1",
        label="Test Session",
        session_type="test",
        target="test_target",
    )
    defaults.update(overrides)
    return mock.MagicMock(**defaults)


# ---------------------------------------------------------------------------
# _try_import_langfuse
# ---------------------------------------------------------------------------

class TestTryImportLangfuse:
    def test_returns_module_when_installed(self):
        with mock.patch.dict(sys.modules, {"langfuse": mock.MagicMock()}):
            # Reset the global warn flag
            import quantpits.scripts.deep_analysis.langfuse_adapter as lfa
            lfa._LANGFUSE_IMPORT_WARNED = False
            result = _try_import_langfuse()
            assert result is sys.modules["langfuse"]

    def test_returns_none_when_not_installed(self):
        with mock.patch.dict(sys.modules):
            if "langfuse" in sys.modules:
                del sys.modules["langfuse"]
            import quantpits.scripts.deep_analysis.langfuse_adapter as lfa
            lfa._LANGFUSE_IMPORT_WARNED = False
            result = _try_import_langfuse()
            assert result is None

    def test_warns_only_once(self, caplog):
        caplog.set_level(logging.WARNING)
        with mock.patch.dict(sys.modules):
            if "langfuse" in sys.modules:
                del sys.modules["langfuse"]
            import quantpits.scripts.deep_analysis.langfuse_adapter as lfa
            lfa._LANGFUSE_IMPORT_WARNED = False

            _try_import_langfuse()
            _try_import_langfuse()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                    and "langfuse package not installed" in r.message]
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# LangfuseAdapter.__init__
# ---------------------------------------------------------------------------

class TestLangfuseAdapterInit:
    def test_disabled_when_enabled_is_false(self):
        adapter = LangfuseAdapter(enabled=False)
        assert adapter._enabled is False
        assert adapter._client is None

    def test_disabled_when_langfuse_not_installed(self, caplog):
        with mock.patch.dict(sys.modules):
            if "langfuse" in sys.modules:
                del sys.modules["langfuse"]
            import quantpits.scripts.deep_analysis.langfuse_adapter as lfa
            lfa._LANGFUSE_IMPORT_WARNED = False
            adapter = LangfuseAdapter(public_key="pk", secret_key="sk", enabled=True)
            assert adapter._enabled is False

    def test_disabled_when_keys_missing(self, caplog):
        caplog.set_level(logging.WARNING)
        with mock.patch.dict(sys.modules, {"langfuse": mock.MagicMock()}):
            adapter = LangfuseAdapter(public_key="", secret_key="sk", enabled=True)
            assert adapter._enabled is False
        assert any("public_key or secret_key missing" in r.message
                   for r in caplog.records)

    def test_disabled_when_client_init_fails(self, caplog):
        caplog.set_level(logging.WARNING)
        mock_lf = mock.MagicMock()
        mock_lf.Langfuse = mock.MagicMock(side_effect=RuntimeError("connection refused"))
        with mock.patch.dict(sys.modules, {"langfuse": mock_lf}):
            adapter = LangfuseAdapter(public_key="pk", secret_key="sk", enabled=True)
            assert adapter._enabled is False
        assert any("failed to initialise client" in r.message
                   for r in caplog.records)

    def test_enabled_when_all_conditions_met(self):
        mock_lf = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_lf.Langfuse.return_value = mock_client
        with mock.patch.dict(sys.modules, {"langfuse": mock_lf}):
            adapter = LangfuseAdapter(public_key="pk", secret_key="sk", enabled=True)
            assert adapter._enabled is True
            assert adapter._client is mock_client
            assert adapter._traces == {}
            assert adapter._spans == {}


# ---------------------------------------------------------------------------
# on_trace / on_session_end / flush  (disabled + exception paths)
# ---------------------------------------------------------------------------

class TestPublicApiDisabled:
    @pytest.fixture
    def adapter(self):
        return LangfuseAdapter(enabled=False)

    def test_on_trace_noop(self, adapter):
        record = _make_record()
        adapter.on_trace(record, None)  # must not raise

    def test_on_session_end_noop(self, adapter):
        session = _make_session_ctx()
        adapter.on_session_end(session)  # must not raise

    def test_flush_noop(self, adapter):
        adapter.flush()  # must not raise

    def test_on_trace_swallows_exception(self, caplog):
        mock_lf = mock.MagicMock()
        mock_lf.Langfuse.return_value = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"langfuse": mock_lf}):
            adapter = LangfuseAdapter(public_key="pk", secret_key="sk", enabled=True)
            adapter._push_generation = mock.MagicMock(side_effect=RuntimeError("boom"))
            caplog.set_level(logging.DEBUG)
            adapter.on_trace(_make_record(), None)
            assert any("on_trace failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _ensure_trace / _ensure_span  caching
# ---------------------------------------------------------------------------

class TestEnsureTraceAndSpan:
    @pytest.fixture
    def adapter(self):
        mock_lf = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_lf.Langfuse.return_value = mock_client
        with mock.patch.dict(sys.modules, {"langfuse": mock_lf}):
            return LangfuseAdapter(public_key="pk", secret_key="sk", enabled=True)

    def test_ensure_trace_creates_on_first_call(self, adapter):
        result = adapter._ensure_trace("run-1", "2025-01-01", "ws")
        assert result is adapter._client.trace.return_value
        adapter._client.trace.assert_called_once()

    def test_ensure_trace_returns_cached_on_second_call(self, adapter):
        first = adapter._ensure_trace("run-1", "2025-01-01", "ws")
        second = adapter._ensure_trace("run-1", "2025-01-01", "ws")
        assert first is second
        assert adapter._client.trace.call_count == 1

    def test_ensure_trace_different_ids_create_separate(self, adapter):
        # Each .trace() call must return a distinct mock
        adapter._client.trace.side_effect = [mock.MagicMock(), mock.MagicMock()]
        first = adapter._ensure_trace("run-1", "2025-01-01", "ws")
        second = adapter._ensure_trace("run-2", "2025-01-02", "ws2")
        assert first is not second
        assert adapter._client.trace.call_count == 2

    def test_ensure_span_caches_by_session_id(self, adapter):
        lf_trace = mock.MagicMock()
        session = _make_session_ctx(session_id="s1")
        first = adapter._ensure_span(session, lf_trace)
        second = adapter._ensure_span(session, lf_trace)
        assert first is second
        assert lf_trace.span.call_count == 1


# ---------------------------------------------------------------------------
# _push_generation
# ---------------------------------------------------------------------------

class TestPushGeneration:
    @pytest.fixture
    def adapter(self):
        mock_lf = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_lf.Langfuse.return_value = mock_client
        with mock.patch.dict(sys.modules, {"langfuse": mock_lf}):
            return LangfuseAdapter(public_key="pk", secret_key="sk", enabled=True)

    def test_creates_generation_on_span_when_session_exists(self, adapter):
        record = _make_record()
        session = _make_session_ctx()
        adapter._push_generation(record, session)
        span = adapter._spans["sess-1"]
        span.generation.assert_called_once()

    def test_creates_generation_on_trace_when_no_session(self, adapter):
        record = _make_record()
        adapter._push_generation(record, None)
        trace = adapter._traces["run-1"]
        trace.generation.assert_called_once()

    def test_passes_correct_model_and_usage(self, adapter):
        record = _make_record(model_responded="gpt-4", prompt_tokens=10,
                              completion_tokens=5, total_tokens=15)
        adapter._push_generation(record, None)
        trace = adapter._traces["run-1"]
        call_kwargs = trace.generation.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["usage"]["prompt_tokens"] == 10
        assert call_kwargs["usage"]["completion_tokens"] == 5
        assert call_kwargs["usage"]["total_tokens"] == 15

    def test_passes_metadata_fields(self, adapter):
        record = _make_record(round_number=2, pipeline_stage="feedback",
                              success=False, error="timeout")
        adapter._push_generation(record, None)
        trace = adapter._traces["run-1"]
        call_kwargs = trace.generation.call_args.kwargs
        assert call_kwargs["metadata"]["round_number"] == 2
        assert call_kwargs["metadata"]["pipeline_stage"] == "feedback"
        assert call_kwargs["metadata"]["success"] is False
        assert call_kwargs["metadata"]["error"] == "timeout"

    def test_error_level_for_failed_record(self, adapter):
        record = _make_record(success=False)
        adapter._push_generation(record, None)
        trace = adapter._traces["run-1"]
        call_kwargs = trace.generation.call_args.kwargs
        assert call_kwargs["level"] == "ERROR"

    def test_default_level_for_successful_record(self, adapter):
        record = _make_record(success=True)
        adapter._push_generation(record, None)
        trace = adapter._traces["run-1"]
        call_kwargs = trace.generation.call_args.kwargs
        assert call_kwargs["level"] == "DEFAULT"

    def test_reasoning_content_wraps_output_as_dict(self, adapter):
        record = _make_record(
            response_content="final answer",
            reasoning_content="chain of thought...",
        )
        adapter._push_generation(record, None)
        trace = adapter._traces["run-1"]
        call_kwargs = trace.generation.call_args.kwargs
        assert isinstance(call_kwargs["output"], dict)
        assert call_kwargs["output"]["content"] == "final answer"
        assert call_kwargs["output"]["reasoning"] == "chain of thought..."

    def test_output_is_plain_string_without_reasoning(self, adapter):
        record = _make_record(response_content="answer", reasoning_content=None)
        adapter._push_generation(record, None)
        trace = adapter._traces["run-1"]
        call_kwargs = trace.generation.call_args.kwargs
        assert call_kwargs["output"] == "answer"


# ---------------------------------------------------------------------------
# from_config factory
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_returns_none_when_enabled_is_false(self):
        cfg = {"trace": {"langfuse": {"enabled": False}}}
        assert LangfuseAdapter.from_config(cfg) is None

    def test_returns_none_when_no_langfuse_section(self):
        cfg = {"trace": {}}
        assert LangfuseAdapter.from_config(cfg) is None

    def test_returns_none_when_env_vars_missing(self, caplog):
        caplog.set_level(logging.WARNING)
        cfg = {
            "trace": {
                "langfuse": {
                    "enabled": True,
                    "host": "http://localhost:3000",
                    "public_key_env": "MISSING_PK",
                    "secret_key_env": "MISSING_SK",
                }
            }
        }
        # Ensure env vars are not set
        with mock.patch.dict(os.environ, {}, clear=True):
            assert LangfuseAdapter.from_config(cfg) is None
        assert any("env vars" in r.message and "not set" in r.message
                   for r in caplog.records)

    def test_returns_adapter_when_configured(self):
        mock_lf = mock.MagicMock()
        mock_lf.Langfuse.return_value = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"langfuse": mock_lf}):
            with mock.patch.dict(os.environ, {"PK": "pk_val", "SK": "sk_val"}):
                cfg = {
                    "trace": {
                        "langfuse": {
                            "enabled": True,
                            "host": "http://example.com:3000",
                            "public_key_env": "PK",
                            "secret_key_env": "SK",
                        }
                    }
                }
                adapter = LangfuseAdapter.from_config(cfg)
                assert isinstance(adapter, LangfuseAdapter)
                assert adapter._enabled is True
