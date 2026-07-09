import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from quantpits.scripts.deep_analysis.stage_runner import StageRunner, CheckpointMeta
from quantpits.scripts.deep_analysis.base_agent import Finding, AgentFindings
from quantpits.scripts.deep_analysis.signal_extractor import Signal
from quantpits.scripts.deep_analysis.action_items import ActionItem

def test_checkpoint_meta():
    meta = CheckpointMeta(stage="discover", date="2026-07-07", label="test", timestamp="2026-07-07T00:00:00")
    d = meta.to_dict()
    assert d["stage"] == "discover"
    assert d["date"] == "2026-07-07"
    assert d["label"] == "test"
    assert d["timestamp"] == "2026-07-07T00:00:00"
    assert d["predecessor"] is None

def test_serialization_deserialization_basic(tmp_path):
    runner = StageRunner(str(tmp_path), "2026-07-07", "test")
    
    # 1. Test basic types
    data = {"int": 1, "float": 2.5, "str": "hello", "bool": True, "none": None}
    serialized = runner._serialize_val(data)
    deserialized = runner._deserialize_val(serialized)
    assert deserialized == data

    # 2. Test nested dict/list
    data_nested = {"list": [1, {"a": "b"}], "dict": {"nested": [True, False]}}
    serialized_nested = runner._serialize_val(data_nested)
    deserialized_nested = runner._deserialize_val(serialized_nested)
    assert deserialized_nested == data_nested

def test_serialization_deserialization_pandas(tmp_path):
    runner = StageRunner(str(tmp_path), "2026-07-07", "test")
    
    # 1. Test DataFrame (small)
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    data = {"df": df}
    serialized = runner._serialize_val(data)
    assert serialized["df"]["_type"] == "dataframe"
    deserialized = runner._deserialize_val(serialized)
    assert isinstance(deserialized["df"], pd.DataFrame)
    pd.testing.assert_frame_equal(deserialized["df"], df)

    # 2. Test DataFrame (large sidecar parquet)
    df_large = pd.DataFrame({"col": list(range(15000))})
    data_large = {"df": df_large}
    serialized_large = runner._serialize_val(data_large)
    assert serialized_large["df"]["_type"] == "parquet_ref"
    
    # Check sidecar file actually exists
    sidecar_path = os.path.join(runner.checkpoint_dir, serialized_large["df"]["path"])
    assert os.path.exists(sidecar_path)
    
    deserialized_large = runner._deserialize_val(serialized_large)
    assert isinstance(deserialized_large["df"], pd.DataFrame)
    pd.testing.assert_frame_equal(deserialized_large["df"], df_large)

    # 3. Test Series
    series = pd.Series([1.1, 2.2, 3.3], index=["a", "b", "c"], name="my_series")
    data_series = {"series": series}
    serialized_series = runner._serialize_val(data_series)
    assert serialized_series["series"]["_type"] == "series"
    deserialized_series = runner._deserialize_val(serialized_series)
    assert isinstance(deserialized_series["series"], pd.Series)
    pd.testing.assert_series_equal(deserialized_series["series"], series)

def test_serialization_deserialization_objects(tmp_path):
    runner = StageRunner(str(tmp_path), "2026-07-07", "test")

    # 1. Finding
    f = Finding(severity="critical", category="TestAgent", title="Alert Title", detail="Detail of warning", data={"k": "v"})
    serialized_f = runner._serialize_val(f)
    deserialized_f = runner._deserialize_val(serialized_f)
    assert isinstance(deserialized_f, Finding)
    assert deserialized_f.severity == f.severity
    assert deserialized_f.category == f.category
    assert deserialized_f.title == f.title
    assert deserialized_f.detail == f.detail
    assert deserialized_f.data == f.data

    # 2. AgentFindings
    af = AgentFindings(agent_name="Model Health", window_label="1y", findings=[f], recommendations=["rec1"], raw_metrics={"foo": "bar"})
    serialized_af = runner._serialize_val(af)
    deserialized_af = runner._deserialize_val(serialized_af)
    assert isinstance(deserialized_af, AgentFindings)
    assert deserialized_af.agent_name == af.agent_name
    assert deserialized_af.window_label == af.window_label
    assert len(deserialized_af.findings) == 1
    assert deserialized_af.findings[0].title == f.title
    assert deserialized_af.recommendations == af.recommendations
    assert deserialized_af.raw_metrics == af.raw_metrics

    # 3. Signal
    s = Signal(signal_type="ic_decay", severity="warning", scope="hyperparams", source_agent="Model Health", target="lstm", metrics={"val": 0.01}, context="desc")
    serialized_s = runner._serialize_val(s)
    deserialized_s = runner._deserialize_val(serialized_s)
    assert isinstance(deserialized_s, Signal)
    assert deserialized_s.signal_type == s.signal_type
    assert deserialized_s.severity == s.severity
    assert deserialized_s.scope == s.scope
    assert deserialized_s.source_agent == s.source_agent
    assert deserialized_s.target == s.target
    assert deserialized_s.metrics == s.metrics
    assert deserialized_s.context == s.context

    # 4. ActionItem
    ai = ActionItem(action_id="1234-abcd", action_type="adjust_hyperparam", scope="hyperparams", target="lstm", params={"lr": 0.01}, reason="tuning", source_signals=["sig1"], expected_outcome="outcome", confidence=0.8, risk_level="low", scope_status="in_scope", rejected_reason="", validated_at="2026-07-07", execution_context={"requires_retrain": True})
    serialized_ai = runner._serialize_val(ai)
    deserialized_ai = runner._deserialize_val(serialized_ai)
    assert isinstance(deserialized_ai, ActionItem)
    assert deserialized_ai.action_id == ai.action_id
    assert deserialized_ai.action_type == ai.action_type
    assert deserialized_ai.scope == ai.scope
    assert deserialized_ai.target == ai.target
    assert deserialized_ai.params == ai.params
    assert deserialized_ai.reason == ai.reason
    assert deserialized_ai.source_signals == ai.source_signals
    assert deserialized_ai.expected_outcome == ai.expected_outcome
    assert deserialized_ai.confidence == ai.confidence
    assert deserialized_ai.risk_level == ai.risk_level
    assert deserialized_ai.scope_status == ai.scope_status
    assert deserialized_ai.rejected_reason == ai.rejected_reason
    assert deserialized_ai.validated_at == ai.validated_at
    assert deserialized_ai.execution_context == ai.execution_context

def test_checkpoint_io(tmp_path):
    runner = StageRunner(str(tmp_path), "2026-07-07", "test_label")
    
    # Check save
    data = {"all_findings": [{"agent_name": "Test", "window_label": "1y", "findings": [], "recommendations": [], "raw_metrics": {}}]}
    path = runner.save_checkpoint("agents", data)
    assert os.path.exists(path)
    assert "agents_2026-07-07_test_label.json" in path

    # Check find latest
    latest_path = runner.find_latest_checkpoint("agents")
    assert latest_path == path

    # Check load
    loaded = runner.load_checkpoint(path)
    assert loaded["meta"]["stage"] == "agents"
    assert loaded["meta"]["date"] == "2026-07-07"
    assert loaded["meta"]["label"] == "test_label"
    assert isinstance(loaded["data"]["all_findings"][0], AgentFindings)


def test_stage_run_loads_only_upstream_and_reruns_target(tmp_path):
    from quantpits.scripts.deep_analysis.stage_runner import STAGE_REGISTRY
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState

    snapshot = StageRunner._snapshot_registry()
    STAGE_REGISTRY.clear()
    calls = []

    def run_discover(state, **kwargs):
        calls.append("discover")
        state.discovered_files = {"from": "fresh"}
        state.windows = [{"label": "fresh"}]
        return state

    def run_agents(state, **kwargs):
        calls.append("agents")
        state.all_findings = []
        return state

    try:
        STAGE_REGISTRY["discover"] = {
            "name": "discover",
            "depends_on": [],
            "provides": ["discovered_files", "windows"],
            "fn": run_discover,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }
        STAGE_REGISTRY["agents"] = {
            "name": "agents",
            "depends_on": ["discover"],
            "provides": ["all_findings"],
            "fn": run_agents,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }

        runner = StageRunner(str(tmp_path), "2026-07-07", "rerun")
        runner.save_checkpoint(
            "discover",
            {"discovered_files": {"from": "checkpoint"}, "windows": [{"label": "ckpt"}]},
        )
        runner.save_checkpoint("agents", {"all_findings": ["stale"]})

        state = DeepAnalysisState(workspace_root=str(tmp_path))
        result = runner.run(target="agents", state=state)

        assert calls == ["agents"]
        assert result.discovered_files == {"from": "checkpoint"}
        assert result.windows == [{"label": "ckpt"}]
        assert result.all_findings == []
    finally:
        StageRunner._restore_registry(snapshot)


def test_explain_plan_loads_upstream_without_running_target(tmp_path):
    from quantpits.scripts.deep_analysis.stage_runner import STAGE_REGISTRY
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState

    snapshot = StageRunner._snapshot_registry()
    STAGE_REGISTRY.clear()
    calls = []

    def run_discover(state, **kwargs):
        calls.append("discover")
        state.discovered_files = {"from": "fresh"}
        state.windows = [{"label": "fresh"}]
        return state

    def run_agents(state, **kwargs):
        calls.append("agents")
        state.all_findings = ["fresh"]
        return state

    try:
        STAGE_REGISTRY["discover"] = {
            "name": "discover",
            "depends_on": [],
            "provides": ["discovered_files", "windows"],
            "fn": run_discover,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }
        STAGE_REGISTRY["agents"] = {
            "name": "agents",
            "depends_on": ["discover"],
            "provides": ["all_findings"],
            "fn": run_agents,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }

        runner = StageRunner(str(tmp_path), "2026-07-07", "dry")
        runner.save_checkpoint(
            "discover",
            {
                "discovered_files": {"from": "checkpoint"},
                "windows": [{"label": "1m"}],
            },
        )

        state = DeepAnalysisState(workspace_root=str(tmp_path))
        plan = runner.explain_plan(
            target="agents",
            state=state,
            windows=["1m"],
            stage_selector="agents:model_health",
            specific_agents=["model_health"],
        )

        assert calls == []
        assert state.completed_stages == []
        assert plan["dependency_closure"] == ["discover"]
        assert plan["execution_plan"] == ["agents"]
        assert plan["checkpoint_events"][0]["stage"] == "discover"
        assert plan["checkpoint_events"][0]["status"] == "loaded"
    finally:
        StageRunner._restore_registry(snapshot)


def test_explain_plan_reports_incompatible_checkpoint(tmp_path):
    from quantpits.scripts.deep_analysis.stage_runner import STAGE_REGISTRY
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState

    snapshot = StageRunner._snapshot_registry()
    STAGE_REGISTRY.clear()

    try:
        STAGE_REGISTRY["discover"] = {
            "name": "discover",
            "depends_on": [],
            "provides": ["discovered_files", "windows"],
            "fn": lambda state, **kwargs: state,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }
        STAGE_REGISTRY["agents"] = {
            "name": "agents",
            "depends_on": ["discover"],
            "provides": ["all_findings"],
            "fn": lambda state, **kwargs: state,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }

        runner = StageRunner(str(tmp_path), "2026-07-07", "dry_mismatch")
        runner.save_checkpoint(
            "discover",
            {
                "discovered_files": {"from": "checkpoint"},
                "windows": [{"label": "3m"}],
            },
        )

        plan = runner.explain_plan(
            target="agents",
            state=DeepAnalysisState(workspace_root=str(tmp_path)),
            windows=["1m"],
        )

        assert plan["checkpoint_events"][0]["status"] == "skipped"
        assert "window labels differ" in plan["checkpoint_events"][0]["reason"]
        assert plan["execution_plan"] == ["discover", "agents"]
    finally:
        StageRunner._restore_registry(snapshot)


def test_empty_stage_outputs_are_valid_when_marked_completed():
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState

    state = DeepAnalysisState(signals=[])
    assert state.has("signals") is False
    assert state.is_stage_done("signals", ["signals"]) is False

    state.mark_completed("signals")
    assert state.is_stage_done("signals", ["signals"]) is True


def test_custom_stage_manifest_runs_and_registry_is_restored(tmp_path):
    import json
    from quantpits.scripts.deep_analysis.stage_runner import STAGE_REGISTRY
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState

    snapshot = StageRunner._snapshot_registry()
    STAGE_REGISTRY.clear()
    plugin_dir = tmp_path / "plugin_stages"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("")
    (plugin_dir / "custom.py").write_text(
        "def run_custom(state, **kwargs):\n"
        "    state.custom_output = 'ok'\n"
        "    return state\n"
    )
    manifest = tmp_path / "config" / "pipeline_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "stages": [{
            "name": "custom_check",
            "class_path": "plugin_stages.custom.run_custom",
            "depends_on": [],
            "provides": ["custom_output"],
            "enabled": True,
        }]
    }))

    try:
        runner = StageRunner(str(tmp_path), "2026-07-07", "custom")
        state = runner.run(
            target="custom_check",
            state=DeepAnalysisState(workspace_root=str(tmp_path)),
            workspace_root=str(tmp_path),
            manifest_path="config/pipeline_manifest.json",
        )

        assert state.custom_output == "ok"
        assert "custom_check" not in STAGE_REGISTRY
        assert os.path.exists(
            tmp_path / "output" / "deep_analysis" / "checkpoints" /
            "custom_check_2026-07-07_custom.json"
        )
    finally:
        StageRunner._restore_registry(snapshot)


def test_explicit_stage_manifest_parse_error_is_reported(tmp_path):
    manifest = tmp_path / "config" / "pipeline_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{not-json")

    runner = StageRunner(str(tmp_path), "2026-07-07", "bad_manifest")

    with pytest.raises(ValueError, match="Failed to parse stage manifest"):
        runner.load_workspace_stages(
            str(tmp_path),
            manifest_path="config/pipeline_manifest.json",
        )


def test_agent_checkpoint_requires_matching_manifest(tmp_path):
    from quantpits.scripts.deep_analysis.stage_runner import STAGE_REGISTRY
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState

    snapshot = StageRunner._snapshot_registry()
    STAGE_REGISTRY.clear()
    calls = []

    def run_discover(state, **kwargs):
        state.discovered_files = {"from": "fresh"}
        state.windows = [{"label": "1m"}]
        return state

    def run_agents(state, **kwargs):
        calls.append("agents")
        state.all_findings = ["fresh"]
        return state

    def run_synthesis(state, **kwargs):
        calls.append("synthesis")
        state.synthesis_result = {"ok": True}
        return state

    try:
        STAGE_REGISTRY["discover"] = {
            "name": "discover",
            "depends_on": [],
            "provides": ["discovered_files", "windows"],
            "fn": run_discover,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }
        STAGE_REGISTRY["agents"] = {
            "name": "agents",
            "depends_on": ["discover"],
            "provides": ["all_findings"],
            "fn": run_agents,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }
        STAGE_REGISTRY["synthesis"] = {
            "name": "synthesis",
            "depends_on": ["agents"],
            "provides": ["synthesis_result"],
            "fn": run_synthesis,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }

        runner = StageRunner(str(tmp_path), "2026-07-07", "manifest_match")
        runner._checkpoint_agent_manifest_path = runner._normalize_path_key(
            "config/old_agent_manifest.json"
        )
        runner.save_checkpoint("agents", {"all_findings": ["stale"]})

        state = DeepAnalysisState(
            workspace_root=str(tmp_path),
            discovered_files={"from": "preloaded"},
            windows=[{"label": "1m"}],
        )
        state.mark_completed("discover")

        result = runner.run(
            target="synthesis",
            state=state,
            agent_manifest_path="config/new_agent_manifest.json",
        )

        assert calls == ["agents", "synthesis"]
        assert result.all_findings == ["fresh"]
        assert result.synthesis_result == {"ok": True}
    finally:
        StageRunner._restore_registry(snapshot)


def test_agent_checkpoint_requires_matching_manifest_fingerprint(tmp_path):
    from quantpits.scripts.deep_analysis.stage_runner import STAGE_REGISTRY
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState

    snapshot = StageRunner._snapshot_registry()
    STAGE_REGISTRY.clear()
    calls = []

    def run_agents(state, **kwargs):
        calls.append("agents")
        state.all_findings = ["fresh"]
        return state

    def run_synthesis(state, **kwargs):
        calls.append("synthesis")
        state.synthesis_result = {"ok": True}
        return state

    try:
        STAGE_REGISTRY["agents"] = {
            "name": "agents",
            "depends_on": ["discover"],
            "provides": ["all_findings"],
            "fn": run_agents,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }
        STAGE_REGISTRY["synthesis"] = {
            "name": "synthesis",
            "depends_on": ["agents"],
            "provides": ["synthesis_result"],
            "fn": run_synthesis,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }
        STAGE_REGISTRY["discover"] = {
            "name": "discover",
            "depends_on": [],
            "provides": ["discovered_files", "windows"],
            "fn": lambda state, **kwargs: state,
            "description": "",
            "source": "builtin",
            "workspace_root": None,
        }

        manifest = tmp_path / "config" / "agent_manifest.json"
        manifest.parent.mkdir()
        manifest.write_text('{"agents": []}')

        runner = StageRunner(str(tmp_path), "2026-07-07", "manifest_fp")
        runner._checkpoint_agent_manifest_path = runner._resolve_agent_manifest_path(None)
        runner._checkpoint_agent_manifest_fingerprint = runner._file_fingerprint(
            runner._checkpoint_agent_manifest_path
        )
        runner.save_checkpoint("agents", {"all_findings": ["stale"]})

        manifest.write_text('{"agents": [{"name": "changed"}]}')

        state = DeepAnalysisState(
            workspace_root=str(tmp_path),
            discovered_files={"from": "preloaded"},
            windows=[{"label": "1m"}],
        )
        state.mark_completed("discover")

        result = runner.run(target="synthesis", state=state)

        assert calls == ["agents", "synthesis"]
        assert result.all_findings == ["fresh"]
    finally:
        StageRunner._restore_registry(snapshot)
