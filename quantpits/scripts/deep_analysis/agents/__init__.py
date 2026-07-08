"""
Agent registry for the MAS Deep Analysis System.
"""

import os
import json
import importlib

from .market_regime import MarketRegimeAgent
from .model_health import ModelHealthAgent
from .ensemble_eval import EnsembleEvolutionAgent
from .execution_quality import ExecutionQualityAgent
from .portfolio_risk import PortfolioRiskAgent
from .prediction_audit import PredictionAuditAgent
from .trade_pattern import TradePatternAgent
from .training_health import TrainingHealthAgent

ALL_AGENTS = {
    'market_regime': MarketRegimeAgent,
    'model_health': ModelHealthAgent,
    'ensemble_eval': EnsembleEvolutionAgent,
    'execution_quality': ExecutionQualityAgent,
    'portfolio_risk': PortfolioRiskAgent,
    'prediction_audit': PredictionAuditAgent,
    'trade_pattern': TradePatternAgent,
    'training_health': TrainingHealthAgent,
}


def load_manifest_agents(workspace_root: str, manifest_path: str = None) -> dict:
    """Load dynamically registered agents from agent_manifest.json/yaml."""
    import sys
    ws_added = False
    loaded_agents = {}
    try:
        if workspace_root and workspace_root not in sys.path:
            sys.path.insert(0, workspace_root)
            ws_added = True

        manifests_to_try = []
        if manifest_path:
            manifests_to_try.append(manifest_path)
        else:
            manifests_to_try.append(os.path.join(workspace_root, "config", "agent_manifest.json"))
            manifests_to_try.append(os.path.join(workspace_root, "config", "agent_manifest.yaml"))

        for path in manifests_to_try:
            if path and os.path.exists(path):
                try:
                    if path.endswith('.yaml') or path.endswith('.yml'):
                        import yaml
                        with open(path, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f) or {}
                    else:
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f) or {}

                    agent_list = data.get('agents', [])
                    for agent_spec in agent_list:
                        if not agent_spec.get('enabled', True):
                            continue
                        name = agent_spec.get('name')
                        class_path = agent_spec.get('class_path')
                        if name and class_path:
                            try:
                                module_path, class_name = class_path.rsplit('.', 1)
                                module = importlib.import_module(module_path)
                                agent_cls = getattr(module, class_name)
                                loaded_agents[name] = agent_cls
                                print(f"  🔌 Loaded dynamic agent: {name} from {class_path}")
                            except Exception as import_err:
                                print(f"  ❌ Failed to import agent '{name}' from '{class_path}': {import_err}")
                except Exception as e:
                    print(f"  ❌ Failed to parse agent manifest at {path}: {e}")
                break
    finally:
        if ws_added and workspace_root in sys.path:
            sys.path.remove(workspace_root)
    return loaded_agents
