import importlib
import json
import os
from unittest.mock import patch


def test_import_does_not_change_cwd():
    before = os.getcwd()
    import quantpits.scripts.order_gen as order_gen

    importlib.reload(order_gen)
    assert os.getcwd() == before


def test_main_json_plan_does_not_execute(order_gen_workspace, capsys):
    """The CLI adapter prints one JSON payload and skips the heavy callback."""
    order_gen, workspace = order_gen_workspace
    from quantpits.order.command import OrderRunConfig

    config = OrderRunConfig(
        merged_config={"market": "csi300", "current_holding": []},
        cashflow_config={},
        strategy_config={"strategy": {"name": "topk_dropout", "params": {}}},
        ensemble_config={},
        ensemble_records={},
        train_records={},
    )
    with (
        patch.object(order_gen, "_default_order_run_config", return_value=config),
        patch.object(order_gen, "run_order_generation") as execute,
        patch("quantpits.order.command.validate_workspace", return_value=None),
    ):
        order_gen.main(["--json-plan", "--run-id", "demo-run"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "order_gen"
    execute.assert_not_called()


import pytest


@pytest.fixture
def order_gen_workspace(tmp_path):
    workspace = tmp_path / "Demo_Workspace"
    (workspace / "config").mkdir(parents=True)
    (workspace / "output").mkdir()
    from quantpits.scripts import order_gen
    return order_gen, workspace
