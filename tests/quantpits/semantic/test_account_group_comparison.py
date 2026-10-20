"""Saved paper/actual records flow into a read-only multi-account CLI report."""
import json
from pathlib import Path

from quantpits.scripts.compare_accounts import main
from tests.quantpits.research.test_account_comparison import group  # noqa: F401

from .artifact_graph import assert_write_conservation, observe_artifact_graph


def test_three_account_cli_preserves_sources_and_declares_derived_writes(group, monkeypatch):
    definition, _a, _b, provider = group
    root = definition.path.parent
    baseline = observe_artifact_graph(root)
    before_accounts = {m.id: observe_artifact_graph(m.ctx.root) for m in definition.members}
    before_provider = observe_artifact_graph(provider)
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(root / "not-a-member"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////unselected-backend.db")
    assert main(["--group", str(definition.path), "--since", "2024-04-05", "--date", "2024-04-12",
                 "--output", str(root / "reports")]) == 0
    report = json.loads((root / "reports/comparison_2024-04-05_2024-04-12.json").read_text())
    assert report["status"] == "complete"
    assert [m["id"] for m in report["members"]] == ["live", "a", "b"]
    assert report["members"][1]["period"]["pnl_excluding_external"] == "4.99"
    assert report["members"][1]["sources"]["settled_period"][0]["anchor"] == "2024-04-05"
    assert report["members"][1]["sources"]["pending_current"][0]["anchor"] == "2024-04-12"
    for member in definition.members:
        after = observe_artifact_graph(member.ctx.root)
        assert after.artifacts == before_accounts[member.id].artifacts
        assert after.physical_escapes == before_accounts[member.id].physical_escapes == ()
    assert observe_artifact_graph(provider).artifacts == before_provider.artifacts
    assert_write_conservation(observe_artifact_graph(root), baseline, ("reports/",))
    assert not any(Path(path).name in ("mlflow.db", "account.lock") for path in observe_artifact_graph(root).changed_paths(baseline))
