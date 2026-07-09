from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from quantpits.runtime import CommandPlan, CommandStep, InputRef, OutputRef, StateRef
from quantpits.runtime.render import command_plan_to_public_dict


def test_command_plan_public_dict_serializes_refs_without_raw_config():
    plan = CommandPlan(
        command="ensemble_fusion",
        workspace="/tmp/ws",
        run_id="run-1",
        mode="from-config-all",
        args=("--all",),
        inputs=(InputRef("config/ensemble_config.json", kind="config", fingerprint="abc"),),
        outputs=(OutputRef("output/ensemble/pred.pkl", kind="prediction", overwrite=True),),
        states=(StateRef("config/ensemble_records.json", action="read_write"),),
        steps=(CommandStep("backtest", "run qlib backtest", expensive=True),),
        config_fingerprints={"ensemble_config": "abc"},
        warnings=("production workspace selected",),
        metadata={
            "summary": {"combos": 2, "path": Path("config/ensemble_config.json")},
            "tags": {"b", "a"},
            "raw_config": {"secret": "do-not-render"},
        },
    )

    payload = command_plan_to_public_dict(plan)

    assert payload["command"] == "ensemble_fusion"
    assert payload["args"] == ["--all"]
    assert payload["inputs"][0]["fingerprint"] == "abc"
    assert payload["outputs"][0]["overwrite"] is True
    assert payload["states"][0]["action"] == "read_write"
    assert payload["steps"][0]["expensive"] is True
    assert "raw_config" not in payload["metadata"]
    assert payload["metadata"]["summary"]["path"] == "config/ensemble_config.json"
    assert payload["metadata"]["tags"] == ["a", "b"]


def test_command_plan_is_frozen():
    plan = CommandPlan(command="order_gen", workspace="/tmp/ws", run_id="run-1")

    with pytest.raises(FrozenInstanceError):
        plan.command = "other"
