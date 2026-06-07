import os
import json
import uuid
import pytest
import tempfile
import time
from unittest.mock import MagicMock, patch, mock_open
from quantpits.scripts.deep_analysis.llm_trace import (
    LLMTraceRecord,
    SessionContext,
    LLMTraceLogger,
    _model_short,
)

def test_model_short_helper():
    # Test prefixes
    assert _model_short("gpt-4-turbo") == "4-turbo"
    assert _model_short("deepseek-v4-pro") == "v4-pro"
    assert _model_short("claude-3-opus") == "3-opus"
    assert _model_short("gemini-1.5-pro") == "1_5-pro"
    assert _model_short("custom-model-name") == "custom-model-nam"  # Truncated to 16

    # Test replacements
    assert _model_short("gpt-3.5/turbo name") == "3_5_turbo_name"
    assert _model_short("my.model/name space") == "my_model_name_sp"


def test_llm_trace_record_basics():
    rec = LLMTraceRecord(
        model_requested="gpt-4",
        messages=[
            {"role": "system", "content": "You are a quant strategist."},
            {"role": "user", "content": "Analyze risk."},
        ],
        response_content="Diagnosis: healthy",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        reasoning_tokens=20,
        duration_ms=500,
    )

    assert rec.trace_id != ""
    assert isinstance(rec.trace_id, str)
    
    # to_dict
    d = rec.to_dict()
    assert d["model_requested"] == "gpt-4"
    assert d["prompt_tokens"] == 100
    assert d["completion_tokens"] == 50
    assert d["total_tokens"] == 150
    assert d["reasoning_tokens"] == 20

    # compute_system_prompt_hash
    rec.compute_system_prompt_hash()
    assert len(rec.system_prompt_hash) == 16
    
    # Hash of "You are a quant strategist."
    import hashlib
    expected_hash = hashlib.sha256("You are a quant strategist.".encode("utf-8")).hexdigest()[:16]
    assert rec.system_prompt_hash == expected_hash

    # Test when no system prompt is present
    rec2 = LLMTraceRecord(messages=[{"role": "user", "content": "hello"}])
    rec2.compute_system_prompt_hash()
    assert rec2.system_prompt_hash == ""

    # Test non-string system content
    rec3 = LLMTraceRecord(messages=[{"role": "system", "content": None}])
    rec3.compute_system_prompt_hash()
    assert rec3.system_prompt_hash == ""


def test_session_context_basics():
    sess = SessionContext(
        session_id="sess_123",
        session_type="per_model_critic",
        label="Per-Model(gru)",
        target="gru",
        run_id="run_123",
    )

    assert sess.session_id == "sess_123"
    assert sess.session_type == "per_model_critic"
    assert sess.label == "Per-Model(gru)"
    assert sess.target == "gru"
    assert sess.run_id == "run_123"
    assert sess.current_round == 1
    assert sess.current_phase == "sole"
    assert sess.current_participant == "primary"
    assert sess.current_role == "sole"

    # elapsed_ms
    time.sleep(0.005)
    assert sess.elapsed_ms() >= 0

    # new_round
    r = sess.new_round(phase="debate")
    assert r == 1
    assert sess.current_round == 1
    assert sess.current_phase == "debate"

    # set_participant
    sess.set_participant("challenger", role="critic")
    assert sess.current_participant == "challenger"
    assert sess.current_role == "critic"

    # record trace
    rec = LLMTraceRecord(
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=2,
        duration_ms=100,
    )
    sess._record_trace(rec, "dummy/path.json")
    
    assert sess.trace_count == 1
    assert sess.total_prompt_tokens == 10
    assert sess.total_completion_tokens == 5
    assert sess.total_reasoning_tokens == 2
    assert sess.total_duration_ms == 100
    assert sess.trace_files == ["dummy/path.json"]


def test_llm_trace_logger_disabled():
    logger = LLMTraceLogger(output_dir="/dummy", enabled=False, run_id="run_1", run_date="2026-06-05")
    assert logger.enabled is False
    assert logger.run_id == "run_1"
    assert logger.run_date == "2026-06-05"
    assert logger.run_dir == ""

    # session context manager
    with logger.session("triage") as sess:
        assert isinstance(sess, SessionContext)
        assert sess.run_id == "run_1"
        assert sess.session_type == "triage"
        assert sess.current_round == 1

    # log_trace should return empty string
    rec = LLMTraceRecord()
    path = logger.log_trace(rec)
    assert path == ""

    # finalize should return empty string
    assert logger.finalize() == ""

    # from_llm_config disabled
    cfg = {"trace": {"enabled": False}}
    logger_from_cfg = LLMTraceLogger.from_llm_config(cfg, "/workspace", run_date="2026-06-05")
    assert logger_from_cfg.enabled is False


def test_llm_trace_logger_enabled_from_config(tmp_path):
    cfg = {
        "trace": {
            "enabled": True,
            "output_dir": "custom_traces",
        }
    }
    logger = LLMTraceLogger.from_llm_config(cfg, str(tmp_path), workspace_name="ws_test", pipeline_stage="layered")
    assert logger.enabled is True
    assert logger.workspace == "ws_test"
    assert logger.pipeline_stage == "layered"
    assert logger.run_dir.startswith(os.path.join(str(tmp_path), "custom_traces"))


def test_llm_trace_logger_full_flow(tmp_path):
    mock_langfuse = MagicMock()
    
    logger = LLMTraceLogger(
        output_dir=str(tmp_path),
        run_id="run12345",
        run_date="2026-06-05",
        workspace="CSI300_Base",
        pipeline_stage="layered",
        langfuse_adapter=mock_langfuse,
        enabled=True,
    )

    assert logger.enabled is True
    assert logger.run_id == "run12345"
    assert logger.run_date == "2026-06-05"
    assert logger.run_dir == os.path.join(str(tmp_path), "2026-06-05_run_run12345")
    assert os.path.exists(logger.run_dir)

    # Session context management with default round 1
    with logger.session("per_model_critic", label="Critic(gru)", target="gru") as sess:
        assert sess.session_type == "per_model_critic"
        assert sess.label == "Critic(gru)"
        assert sess.target == "gru"
        
        # Create record
        rec = LLMTraceRecord(
            model_requested="deepseek-chat",
            model_responded="deepseek-chat",
            messages=[{"role": "system", "content": "Critic prompt"}],
            response_content="Analysis result",
            duration_ms=450,
            prompt_tokens=80,
            completion_tokens=40,
            reasoning_tokens=0,
        )
        
        # log trace
        path = logger.log_trace(rec, sess)
        
        assert path != ""
        assert os.path.exists(path)
        assert "per_model" in path
        assert "gru" in path
        
        # Verify stored JSON
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["trace_id"] == rec.trace_id
        assert data["run_id"] == "run12345"
        assert data["session_id"] == sess.session_id
        assert data["session_type"] == "per_model_critic"
        assert data["label"] == "Critic(gru)"
        assert data["model_requested"] == "deepseek-chat"
        assert data["model_responded"] == "deepseek-chat"
        assert data["success"] is True
        assert data["round_number"] == 1
        assert data["round_phase"] == "sole"
        assert data["participant_id"] == "primary"
        assert data["participant_role"] == "sole"

    # Langfuse session end callback should be called
    mock_langfuse.on_session_end.assert_called_once_with(sess)
    mock_langfuse.on_trace.assert_called_once_with(rec, sess)

    # Multi-LLM session trace logging (phase != "sole")
    with logger.session("triage", label="TriageLabel") as sess2:
        sess2.new_round("debate")
        sess2.set_participant("challenger", role="critic")
        
        rec2 = LLMTraceRecord(
            model_requested="gpt-4",
            messages=[{"role": "user", "content": "Triage query"}],
            response_content="{}",
            duration_ms=120,
            prompt_tokens=50,
            completion_tokens=25,
        )
        path2 = logger.log_trace(rec2, sess2)
        assert path2 != ""
        
        # Check filename encoding
        filename = os.path.basename(path2)
        assert "r1" in filename
        assert "challenger" in filename
        assert "debate" in filename

    # Run summary and finalize
    summary = logger.get_run_summary()
    assert summary["run_id"] == "run12345"
    assert summary["summary"]["total_traces"] == 2
    assert summary["summary"]["total_sessions"] == 2
    assert summary["summary"]["total_prompt_tokens"] == 130
    assert summary["summary"]["total_completion_tokens"] == 65
    assert len(summary["sessions"]) == 2

    # Finalize
    run_dir = logger.finalize()
    assert run_dir == logger.run_dir
    manifest_path = os.path.join(run_dir, "manifest.json")
    assert os.path.exists(manifest_path)

    # Verify manifest timeline
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    
    assert manifest["summary"]["total_traces"] == 2
    assert len(manifest["timeline"]) == 2
    assert manifest["timeline"][0]["seq"] == "001"
    assert manifest["timeline"][0]["model"] == "deepseek-chat"
    assert manifest["timeline"][1]["seq"] == "002"
    assert manifest["timeline"][1]["model"] == "gpt-4"
    
    mock_langfuse.flush.assert_called_once()


def test_llm_trace_logger_serialization_error(tmp_path):
    # Test JSON serialization error handling
    logger = LLMTraceLogger(output_dir=str(tmp_path), enabled=True)
    
    # Create non-serializable object
    class NonSerializable:
        pass
        
    rec = LLMTraceRecord(
        other_params={"bad": NonSerializable()},
        session_type="synthesizer",
    )
    rec.to_dict = MagicMock(side_effect=TypeError("mocked serialization error"))
    
    path = logger.log_trace(rec)
    assert path != ""
    assert os.path.exists(path)
    
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        
    assert data["trace_id"] == rec.trace_id
    assert "Serialisation failed" in data["error"]
    assert data["session_type"] == "synthesizer"


def test_llm_trace_logger_write_error(tmp_path):
    logger = LLMTraceLogger(output_dir=str(tmp_path), enabled=True)
    rec = LLMTraceRecord()
    
    # Mock open to raise exception
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = IOError("Permission denied")
        path = logger.log_trace(rec)
        assert path == ""


def test_llm_trace_logger_manifest_write_error(tmp_path):
    logger = LLMTraceLogger(output_dir=str(tmp_path), enabled=True)
    
    # Mock open to raise exception during manifest write
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = IOError("Disk full")
        run_dir = logger.finalize()
        assert run_dir == logger.run_dir


def test_llm_trace_logger_timeline_read_error(tmp_path):
    logger = LLMTraceLogger(output_dir=str(tmp_path), enabled=True)
    
    with logger.session("triage") as sess:
        rec = LLMTraceRecord(model_requested="gpt-4")
        path = logger.log_trace(rec, sess)
        
        # Write corrupted JSON to the trace file to trigger timeline read exception
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("corrupted { json")
            
        run_dir = logger.finalize()
        # Corrupted trace should be skipped in timeline
        manifest_path = os.path.join(run_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert len(manifest["timeline"]) == 0


def test_llm_trace_logger_langfuse_exceptions(tmp_path):
    mock_langfuse = MagicMock()
    mock_langfuse.on_session_end.side_effect = RuntimeError("Langfuse session end error")
    mock_langfuse.on_trace.side_effect = RuntimeError("Langfuse trace error")
    mock_langfuse.flush.side_effect = RuntimeError("Langfuse flush error")

    logger = LLMTraceLogger(
        output_dir=str(tmp_path),
        langfuse_adapter=mock_langfuse,
        enabled=True,
    )

    with logger.session("triage") as sess:
        # round_number < 1 triggers line 422 branch
        rec = LLMTraceRecord(model_requested="gpt-4", round_number=0)
        path = logger.log_trace(rec, sess)
        assert path != ""
        assert rec.round_number == 1  # Should be set from session context

    logger.finalize()


def test_llm_trace_logger_with_run_label(tmp_path):
    logger = LLMTraceLogger(
        output_dir=str(tmp_path),
        run_id="abc12345",
        run_date="2026-06-05",
        run_label="after-retrain",
        enabled=True,
    )
    assert logger.run_label == "after-retrain"
    assert logger.run_dir == os.path.join(str(tmp_path), "2026-06-05_after-retrain_run_abc12345")
    assert os.path.exists(logger.run_dir)

    # get_run_summary should include the label
    summary = logger.get_run_summary()
    assert summary["run_label"] == "after-retrain"

    logger.finalize()


def test_llm_trace_logger_from_config_with_run_label(tmp_path):
    cfg = {"trace": {"enabled": True, "output_dir": "traces"}}
    logger = LLMTraceLogger.from_llm_config(cfg, str(tmp_path), run_label="v2")
    assert logger.run_label == "v2"
    assert "v2" in logger.run_dir


def test_llm_trace_logger_without_label_is_unchanged(tmp_path):
    logger = LLMTraceLogger(
        output_dir=str(tmp_path),
        run_id="abc12345",
        run_date="2026-06-05",
        enabled=True,
    )
    assert logger.run_label == ""
    assert logger.run_dir == os.path.join(str(tmp_path), "2026-06-05_run_abc12345")
