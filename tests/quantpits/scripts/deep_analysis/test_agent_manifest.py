import os
import json
import sys
import pytest
from unittest.mock import MagicMock

from quantpits.scripts.deep_analysis.agents import load_manifest_agents, ALL_AGENTS
from quantpits.scripts.deep_analysis.base_agent import BaseAgent, AgentFindings, AnalysisContext

# Dummy Agent for testing
class DummyCustomAgent(BaseAgent):
    name = "Dummy Custom"
    description = "Test dynamic loading."
    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        return AgentFindings(self.name, ctx.window_label, [], [], {})

def test_agent_manifest_loading(tmp_path):
    # Create temp workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config").mkdir()

    # Define a custom python module dynamically
    custom_module_dir = tmp_path / "my_custom_package"
    custom_module_dir.mkdir()
    (custom_module_dir / "__init__.py").touch()
    
    agent_code = """
from quantpits.scripts.deep_analysis.base_agent import BaseAgent, AgentFindings, AnalysisContext
class CustomPluginAgent(BaseAgent):
    name = "Custom Plugin"
    description = "Custom agent loaded from manifest."
    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        return AgentFindings(self.name, ctx.window_label, [], [], {})
"""
    with open(custom_module_dir / "my_agent.py", "w") as f:
        f.write(agent_code)

    # Add custom module directory to sys.path so it's importable
    sys.path.insert(0, str(tmp_path))

    try:
        # Create agent_manifest.json
        manifest_data = {
            "agents": [
                {
                    "name": "custom_plugin",
                    "class_path": "my_custom_package.my_agent.CustomPluginAgent",
                    "enabled": True
                },
                {
                    "name": "disabled_plugin",
                    "class_path": "my_custom_package.my_agent.CustomPluginAgent",
                    "enabled": False
                }
            ]
        }
        
        manifest_path = workspace / "config" / "agent_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        # Load manifest agents
        loaded = load_manifest_agents(str(workspace))
        
        # Verify custom_plugin is loaded and disabled_plugin is ignored
        assert "custom_plugin" in loaded
        assert "disabled_plugin" not in loaded
        
        agent_cls = loaded["custom_plugin"]
        assert agent_cls.name == "Custom Plugin"
        
        # Verify instantiation
        agent_inst = agent_cls()
        ctx = AnalysisContext(start_date="2026-01-01", end_date="2026-02-01", window_label="1m", workspace_root=str(workspace))
        res = agent_inst.analyze(ctx)
        assert res.agent_name == "Custom Plugin"

    finally:
        # Clean up path
        sys.path.remove(str(tmp_path))
