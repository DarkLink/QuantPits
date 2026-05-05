"""Tests for ConfigPromoter and update_promote_status."""

import json
import os
import pytest
import yaml

from quantpits.scripts.deep_analysis.promote_config import (
    ConfigPromoter,
    update_promote_status,
)


@pytest.fixture
def promote_workspaces(tmp_path):
    """Create production + playground workspaces for promotion testing."""
    prod = tmp_path / "CSI300_Base"
    prod.mkdir()
    pg = tmp_path / "CSI300_Base_Playground"
    pg.mkdir()

    # Production config
    prod_config = prod / "config"
    prod_config.mkdir()

    workflow = {
        "task": {
            "model": {
                "class": "GRU",
                "kwargs": {"n_epochs": 200, "early_stop": 10},
            }
        },
        "data_handler_config": {"label": ["Ref($close, -2)"]},
    }
    with open(prod_config / "workflow_config_gru.yaml", "w") as f:
        yaml.dump(workflow, f)

    with open(prod_config / "ensemble_config.json", "w") as f:
        json.dump({"combo_groups": {}}, f)

    # Playground config (modified)
    pg_config = pg / "config"
    pg_config.mkdir()

    workflow_modified = {
        "task": {
            "model": {
                "class": "GRU",
                "kwargs": {"n_epochs": 200, "early_stop": 20},  # Changed!
            }
        },
        "data_handler_config": {"label": ["Ref($close, -2)"]},
    }
    with open(pg_config / "workflow_config_gru.yaml", "w") as f:
        yaml.dump(workflow_modified, f)

    with open(pg_config / "ensemble_config.json", "w") as f:
        json.dump({"combo_groups": {}}, f)

    # Data dirs
    (prod / "data").mkdir()
    (pg / "data").mkdir()

    # Playground metadata
    meta = {
        "created_at": "2026-05-01T10:00:00",
        "synced_at": "2026-05-01T10:00:00",
        "source_workspace": str(prod),
        "action_items_applied": [],
        "status": "active",
    }
    with open(pg / "_playground_meta.json", "w") as f:
        json.dump(meta, f)

    return prod, pg


class TestConfigPromoter:
    def test_preview_detects_changes(self, promote_workspaces):
        """Test that preview identifies changed config files."""
        prod, pg = promote_workspaces
        promoter = ConfigPromoter(str(pg), str(prod))

        preview = promoter.preview()
        assert len(preview.files_to_copy) > 0
        # Should detect the workflow YAML change
        changed_files = [os.path.basename(f) for f in preview.files_to_copy]
        assert "workflow_config_gru.yaml" in changed_files

    def test_preview_no_changes(self, promote_workspaces):
        """Test preview with identical configs shows no changes."""
        prod, pg = promote_workspaces

        # Make playground identical to production
        import shutil
        shutil.rmtree(str(pg / "config"))
        shutil.copytree(str(prod / "config"), str(pg / "config"))

        promoter = ConfigPromoter(str(pg), str(prod))
        preview = promoter.preview()
        assert len(preview.files_to_copy) == 0

    def test_promote_success(self, promote_workspaces):
        """Test successful promotion copies files and creates records."""
        prod, pg = promote_workspaces
        promoter = ConfigPromoter(str(pg), str(prod))

        result = promoter.promote(
            action_item_ids=["test-action-001"],
            validation_results=[{
                "model": "gru_Alpha158",
                "baseline_ic": 0.05,
                "playground_ic": 0.06,
                "ic_delta": 0.01,
                "passed": True,
            }],
            reason="Test promotion",
        )

        assert result.success
        assert len(result.promoted_files) > 0

        # Verify production config was updated
        with open(prod / "config" / "workflow_config_gru.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        assert cfg["task"]["model"]["kwargs"]["early_stop"] == 20

        # Verify promote_history.jsonl was created
        history_path = prod / "data" / "promote_history.jsonl"
        assert history_path.exists()
        with open(history_path, "r") as f:
            record = json.loads(f.readline())
        assert record["status"] == "promoted_pending_retrain"
        assert "test-action-001" in record["action_item_ids"]

        # Verify human-readable report
        report_dir = prod / "data" / "promote_history"
        assert report_dir.exists()
        reports = list(report_dir.glob("promote_*.md"))
        assert len(reports) == 1

        # Verify CHANGELOG.md
        changelog = prod / "data" / "CHANGELOG.md"
        assert changelog.exists()
        content = changelog.read_text()
        assert "test-action-001" in content

    def test_promote_updates_playground_meta(self, promote_workspaces):
        """Test that promote updates _playground_meta.json."""
        prod, pg = promote_workspaces
        promoter = ConfigPromoter(str(pg), str(prod))

        promoter.promote(action_item_ids=["test-001"], reason="test")

        with open(pg / "_playground_meta.json", "r") as f:
            meta = json.load(f)
        assert "test-001" in meta["action_items_applied"]
        assert "last_promote_id" in meta


class TestUpdatePromoteStatus:
    def test_update_pending_to_active(self, tmp_path):
        """Test status transition: promoted_pending_retrain → active."""
        ws = tmp_path / "workspace"
        (ws / "data").mkdir(parents=True)

        record = {
            "promote_id": "abc123",
            "status": "promoted_pending_retrain",
            "retrained_at": None,
            "changes": [{"model": "gru_Alpha158", "param": "early_stop"}],
        }
        with open(ws / "data" / "promote_history.jsonl", "w") as f:
            f.write(json.dumps(record) + "\n")

        update_promote_status(str(ws), model_names=["gru_Alpha158"])

        with open(ws / "data" / "promote_history.jsonl", "r") as f:
            updated = json.loads(f.readline())
        assert updated["status"] == "active"
        assert updated["retrained_at"] is not None

    def test_update_filters_by_model(self, tmp_path):
        """Test that model filter only updates matching records."""
        ws = tmp_path / "workspace"
        (ws / "data").mkdir(parents=True)

        records = [
            {"promote_id": "1", "status": "promoted_pending_retrain",
             "retrained_at": None, "changes": [{"model": "gru"}]},
            {"promote_id": "2", "status": "promoted_pending_retrain",
             "retrained_at": None, "changes": [{"model": "lstm"}]},
        ]
        with open(ws / "data" / "promote_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        update_promote_status(str(ws), model_names=["gru"])

        with open(ws / "data" / "promote_history.jsonl", "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines[0]["status"] == "active"      # gru → updated
        assert lines[1]["status"] == "promoted_pending_retrain"  # lstm → unchanged

    def test_no_history_file(self, tmp_path):
        """Test graceful handling when no promote_history.jsonl exists."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        # Should not raise
        update_promote_status(str(ws))


# ── Additional promote_config tests ────────────────────────────

def test_promote_preview_detects_changes_standalone(tmp_path):
    """preview() should detect YAML differences between playground and production."""
    import shutil

    # Create a minimal production + playground setup
    prod = tmp_path / "prod"
    prod.mkdir()
    (prod / "config").mkdir()
    pg = tmp_path / "pg"
    pg.mkdir()
    (pg / "config").mkdir()

    # Same base config in both
    workflow = {
        "task": {"model": {"kwargs": {"early_stop": 10, "n_epochs": 200}}},
        "data_handler_config": {},
    }
    with open(prod / "config" / "workflow_config_test.yaml", "w") as f:
        yaml.dump(workflow, f)

    # Slightly modified in playground
    workflow_mod = {
        "task": {"model": {"kwargs": {"early_stop": 20, "n_epochs": 200}}},
        "data_handler_config": {},
    }
    with open(pg / "config" / "workflow_config_test.yaml", "w") as f:
        yaml.dump(workflow_mod, f)

    from quantpits.scripts.deep_analysis.promote_config import ConfigPromoter
    promoter = ConfigPromoter(str(pg), str(prod))
    preview = promoter.preview()
    assert len(preview.files_to_copy) > 0


def test_update_promote_status_transitions(tmp_path):
    """update_promote_status should transition pending_retrain → active."""
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "data"), exist_ok=True)
    history_path = os.path.join(ws, "data", "promote_history.jsonl")

    record = json.dumps({
        "promote_id": "test-promote-001",
        "status": "promoted_pending_retrain",
        "changes": [{"model": "gru_Alpha158", "param": "early_stop", "old": 10, "new": 20}],
        "action_item_ids": ["ai-001"],
    })
    with open(history_path, "w") as f:
        f.write(record + "\n")

    update_promote_status(ws, model_names=["gru_Alpha158"])

    with open(history_path, "r") as f:
        updated = json.loads(f.readline())
    assert updated["status"] == "active"
    assert updated.get("retrained_at") is not None


def test_update_promote_status_partial_model_match(tmp_path):
    """update_promote_status with model filter should only update matching records."""
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "data"), exist_ok=True)
    history_path = os.path.join(ws, "data", "promote_history.jsonl")

    records = [
        json.dumps({
            "promote_id": "prom-001",
            "status": "promoted_pending_retrain",
            "changes": [{"model": "model_a", "param": "lr", "old": 0.01, "new": 0.02}],
        }),
        json.dumps({
            "promote_id": "prom-002",
            "status": "promoted_pending_retrain",
            "changes": [{"model": "model_b", "param": "n_epochs", "old": 100, "new": 150}],
        }),
    ]
    with open(history_path, "w") as f:
        f.write("\n".join(records) + "\n")

    update_promote_status(ws, model_names=["model_a"])

    with open(history_path, "r") as f:
        updated = [json.loads(line) for line in f if line.strip()]
    assert updated[0]["status"] == "active"
    assert updated[1]["status"] == "promoted_pending_retrain"


def test_update_promote_status_corrupt_jsonl(tmp_path):
    """update_promote_status should preserve corrupt JSON lines unchanged."""
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "data"), exist_ok=True)
    history_path = os.path.join(ws, "data", "promote_history.jsonl")

    with open(history_path, "w") as f:
        f.write("not valid json at all\n")
        f.write('{"promote_id": "ok", "status": "promoted_pending_retrain", "changes": []}\n')

    update_promote_status(ws)

    with open(history_path, "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == "not valid json at all"
    updated = json.loads(lines[1])
    assert updated["status"] == "active"


def test_promote_result_dataclass():
    """PromoteResult should support construction and attribute access."""
    from quantpits.scripts.deep_analysis.promote_config import PromoteResult, PromotePreview

    preview = PromotePreview(
        diff_summary=[{"param": "lr", "old": 0.01, "new": 0.02}],
        files_to_copy=["config/a.yaml"],
        production_snapshot_id="snap-001",
        playground_snapshot_id="snap-002",
    )
    assert len(preview.files_to_copy) == 1

    result = PromoteResult(
        success=True,
        promoted_files=["config/a.yaml"],
        promote_record={"promote_id": "test"},
    )
    assert result.success
