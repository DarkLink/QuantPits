"""Generated, sanitized workspace fixtures for semantic contracts."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from quantpits.utils.workspace import WorkspaceContext


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ScenarioWorkspace:
    root: Path
    ctx: WorkspaceContext

    @classmethod
    def create(cls, tmp_path, *, family="static"):
        root = tmp_path / "Demo_Workspace"
        for relative in ("config", "data/history", "output", "mlruns"):
            (root / relative).mkdir(parents=True, exist_ok=True)
        suffix = "rolling" if family == "rolling" else "static"
        models = {"m1@%s" % suffix: "source-m1", "m2@%s" % suffix: "source-m2"}
        _write_json(root / "latest_train_records.json", {
            "anchor_date": "2026-07-16", "experiment_name": "SanitizedTraining", "models": models,
        })
        _write_json(root / "config/model_config.json", {"freq": "week", "market": "csi300"})
        _write_json(root / "config/prod_config.json", {
            "current_date": "2026-07-16", "last_processed_date": "2026-07-16",
            "current_cash": 1000.0, "current_holding": [],
        })
        _write_json(root / "config/cashflow.json", {"cashflows": {}})
        _write_json(root / "config/ensemble_config.json", {
            "combos": {"sentinel": {"models": ["m1", "m2"], "method": "equal", "default": True}}
        })
        (root / "config/strategy_config.yaml").write_text(yaml.safe_dump({
            "strategy": {"name": "topk_dropout", "params": {"topk": 1, "n_drop": 0, "buy_suggestion_factor": 1}}
        }), encoding="utf-8")
        (root / "config/rolling_config.yaml").write_text(yaml.safe_dump({
            "rolling_start": "2020-01-01", "train_years": 3, "valid_years": 1,
            "test_step": "3M", "training_method": "slide",
        }), encoding="utf-8")
        (root / "config/model_registry.yaml").write_text(yaml.safe_dump({
            "models": {"demo": {"enabled": True, "algorithm": "linear", "dataset": "Alpha158", "yaml_file": "config/demo.yaml"}}
        }), encoding="utf-8")
        (root / "config/demo.yaml").write_text("model: demo\n", encoding="utf-8")
        return cls(root.resolve(), WorkspaceContext.from_root(root))

    def write_json(self, relative, value):
        _write_json(self.root / relative, value)

    def read_json(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))
