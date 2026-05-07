"""Tests for PlaygroundManager."""

import json
import os
import pytest

from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager


@pytest.fixture
def production_workspace(tmp_path):
    """Create a realistic production workspace for testing."""
    ws = tmp_path / "CSI300_Base"
    ws.mkdir()

    # config/
    config_dir = ws / "config"
    config_dir.mkdir()

    import yaml
    workflow = {
        "task": {
            "model": {
                "class": "GRU",
                "kwargs": {"n_epochs": 200, "early_stop": 10, "lr": 0.001},
            }
        },
        "data_handler_config": {"label": ["Ref($close, -2) / Ref($close, -1) - 1"]},
    }
    with open(config_dir / "workflow_config_gru_Alpha158.yaml", "w") as f:
        yaml.dump(workflow, f)

    with open(config_dir / "feedback_scope.json", "w") as f:
        json.dump({"active_scopes": ["hyperparams"]}, f)

    with open(config_dir / "hyperparam_bounds.json", "w") as f:
        json.dump({"bounds": {"n_epochs": {"min": 10, "max": 500}}}, f)

    skills_dir = config_dir / "skills"
    skills_dir.mkdir()
    (skills_dir / "critic.md").write_text("# Critic skill")

    # data/
    data_dir = ws / "data"
    data_dir.mkdir()

    # Entity-copy candidates
    (data_dir / "training_history.jsonl").write_text(
        json.dumps({"model_name": "gru", "trained_at": "2026-04-24", "duration_seconds": 120}) + "\n"
    )
    (data_dir / "fusion_run_ledger.jsonl").write_text(
        json.dumps({"run_date": "2026-04-29"}) + "\n"
    )
    (data_dir / "operator_log.jsonl").write_text("")
    (data_dir / "action_item_history.jsonl").write_text("")
    (data_dir / "run_state.json").write_text("{}")
    (data_dir / "rolling_state.json").write_text("{}")

    # Symlink-candidate files
    (data_dir / "trade_log_full.csv").write_text("col1,col2\n1,2\n")
    (data_dir / "daily_amount.xlsx").write_bytes(b"fake xlsx content")

    # Symlink-candidate directories
    (data_dir / "order_history").mkdir()
    (data_dir / "order_history" / "orders_2026.json").write_text("{}")
    (data_dir / "history").mkdir()
    (data_dir / "history" / "snapshot.json").write_text("{}")
    (data_dir / "config_history").mkdir()
    (data_dir / "config_history" / "config_snapshot_2026-04-24.json").write_text(
        json.dumps({"snapshot_date": "2026-04-24"})
    )

    # Root-level entity copy
    (ws / "latest_train_records.json").write_text(json.dumps({"models": {}}))

    # output/ (should NOT be synced)
    (ws / "output").mkdir()
    (ws / "output" / "results.json").write_text("{}")

    # mlruns/ (should NOT be synced)
    (ws / "mlruns").mkdir()

    return ws


class TestPlaygroundManager:
    def test_create_playground(self, production_workspace):
        """Test Playground creation with correct directory structure."""
        mgr = PlaygroundManager(str(production_workspace))
        pg_root = mgr.create_or_sync()

        assert os.path.isdir(pg_root)
        assert pg_root.endswith("_Playground")

        # config/ should be fully copied
        assert os.path.isfile(os.path.join(pg_root, "config", "workflow_config_gru_Alpha158.yaml"))
        assert os.path.isfile(os.path.join(pg_root, "config", "feedback_scope.json"))
        assert os.path.isfile(os.path.join(pg_root, "config", "skills", "critic.md"))

        # data/ entity copies should exist and be real files (not symlinks)
        th = os.path.join(pg_root, "data", "training_history.jsonl")
        assert os.path.isfile(th)
        assert not os.path.islink(th)

        # data/ symlinked dirs should be symlinks
        oh = os.path.join(pg_root, "data", "order_history")
        assert os.path.islink(oh)

        hist = os.path.join(pg_root, "data", "history")
        assert os.path.islink(hist)

        # data/ symlinked files
        csv = os.path.join(pg_root, "data", "trade_log_full.csv")
        assert os.path.islink(csv)

        xlsx = os.path.join(pg_root, "data", "daily_amount.xlsx")
        assert os.path.islink(xlsx)

        # Root-level entity copy
        assert os.path.isfile(os.path.join(pg_root, "latest_train_records.json"))

        # output/ should exist but be empty (independent)
        assert os.path.isdir(os.path.join(pg_root, "output"))
        # Production output should NOT be copied
        assert not os.path.isfile(os.path.join(pg_root, "output", "results.json"))

        # mlruns/ should NOT be synced
        assert not os.path.isdir(os.path.join(pg_root, "mlruns"))

    def test_metadata_created(self, production_workspace):
        """Test _playground_meta.json is created with correct content."""
        mgr = PlaygroundManager(str(production_workspace))
        mgr.create_or_sync()

        meta = mgr.get_meta()
        assert meta["source_workspace"] == str(production_workspace)
        assert meta["status"] == "active"
        assert "created_at" in meta
        assert "synced_at" in meta
        assert meta["action_items_applied"] == []
        assert meta["baseline_snapshot"]["training_history_latest_date"] == "2026-04-24"
        assert meta["baseline_snapshot"]["fusion_ledger_latest_date"] == "2026-04-29"

    def test_resync_updates_meta(self, production_workspace):
        """Test that re-syncing updates synced_at but preserves created_at."""
        mgr = PlaygroundManager(str(production_workspace))
        mgr.create_or_sync()
        meta1 = mgr.get_meta()

        # Resync
        mgr.create_or_sync()
        meta2 = mgr.get_meta()

        assert meta2["created_at"] == meta1["created_at"]
        # synced_at should be updated (or same if very fast)
        assert "synced_at" in meta2

    def test_data_isolation(self, production_workspace):
        """Test that Playground data files are isolated from production."""
        mgr = PlaygroundManager(str(production_workspace))
        pg_root = mgr.create_or_sync()

        # Write to playground training_history — should NOT affect production
        pg_th = os.path.join(pg_root, "data", "training_history.jsonl")
        with open(pg_th, "a") as f:
            f.write(json.dumps({"model_name": "playground_model"}) + "\n")

        # Production should still have original content
        prod_th = os.path.join(str(production_workspace), "data", "training_history.jsonl")
        with open(prod_th, "r") as f:
            content = f.read()
        assert "playground_model" not in content

    def test_clean(self, production_workspace):
        """Test clean removes the Playground entirely."""
        mgr = PlaygroundManager(str(production_workspace))
        mgr.create_or_sync()
        assert mgr.get_playground_root() is not None

        mgr.clean()
        assert mgr.get_playground_root() is None

    def test_get_playground_root_none(self, production_workspace):
        """Test get_playground_root returns None when not created."""
        mgr = PlaygroundManager(str(production_workspace))
        assert mgr.get_playground_root() is None

    def test_symlinked_dirs_are_readable(self, production_workspace):
        """Test that symlinked directories are readable in the Playground."""
        mgr = PlaygroundManager(str(production_workspace))
        pg_root = mgr.create_or_sync()

        # Should be able to list files through symlink
        oh_path = os.path.join(pg_root, "data", "order_history")
        files = os.listdir(oh_path)
        assert "orders_2026.json" in files


# ── Additional playground_manager tests ────────────────────────

def test_create_or_sync_pretrain_dir(production_workspace):
    """create_or_sync should entity-copy data/pretrained/ when it exists in production."""
    # Create pretrained data in production
    pretrained_dir = os.path.join(production_workspace, "data", "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)
    with open(os.path.join(pretrained_dir, "lstm_Alpha360_latest.pkl"), "w") as f:
        f.write("mock model weights")

    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    mgr = PlaygroundManager(str(production_workspace))
    pg_root = mgr.create_or_sync()

    # pretrained/ should be entity-copied (not symlinked)
    pg_pretrained = os.path.join(pg_root, "data", "pretrained")
    assert os.path.isdir(pg_pretrained)
    assert not os.path.islink(pg_pretrained)
    assert os.path.isfile(os.path.join(pg_pretrained, "lstm_Alpha360_latest.pkl"))

    mgr.clean()


def test_create_or_sync_no_pretrain_dir(production_workspace):
    """create_or_sync should not crash when data/pretrained/ is absent."""
    # Ensure no pretrained dir
    pretrained_dir = os.path.join(production_workspace, "data", "pretrained")
    if os.path.exists(pretrained_dir):
        shutil.rmtree(pretrained_dir)

    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    mgr = PlaygroundManager(str(production_workspace))
    pg_root = mgr.create_or_sync()

    # Should complete without error
    assert os.path.isdir(pg_root)
    mgr.clean()


def test_create_or_sync_log_file_isolation(production_workspace):
    """Playground training_history.jsonl should be independent from production."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    mgr = PlaygroundManager(str(production_workspace))
    pg_root = mgr.create_or_sync()

    # Write to playground's training_history (simulating training)
    pg_history = os.path.join(pg_root, "data", "training_history.jsonl")
    with open(pg_history, "a") as f:
        f.write('{"model_name": "playground_model", "trained_at": "2026-05-05"}\n')

    # Production training_history should NOT have the playground entry
    prod_history = os.path.join(production_workspace, "data", "training_history.jsonl")
    with open(prod_history, "r") as f:
        prod_lines = f.readlines()
    assert not any("playground_model" in line for line in prod_lines)

    mgr.clean()


def test_create_or_sync_xlsx_csv_symlinked(production_workspace):
    """*.xlsx and *.csv files in data/ should be symlinked, not copied."""
    # Create a CSV and XLSX in production data
    csv_path = os.path.join(production_workspace, "data", "test_data.csv")
    with open(csv_path, "w") as f:
        f.write("col1,col2\n1,2\n")
    xlsx_path = os.path.join(production_workspace, "data", "test_data.xlsx")
    with open(xlsx_path, "w") as f:
        f.write("dummy")

    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    mgr = PlaygroundManager(str(production_workspace))
    pg_root = mgr.create_or_sync()

    pg_csv = os.path.join(pg_root, "data", "test_data.csv")
    assert os.path.islink(pg_csv), f"Expected symlink for CSV, got real file"
    pg_xlsx = os.path.join(pg_root, "data", "test_data.xlsx")
    assert os.path.islink(pg_xlsx), f"Expected symlink for XLSX, got real file"

    mgr.clean()


def test_create_or_sync_idempotent(production_workspace):
    """create_or_sync() should work correctly when playground already exists."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    mgr = PlaygroundManager(str(production_workspace))
    pg_root_1 = mgr.create_or_sync()
    pg_root_2 = mgr.create_or_sync()
    # Second call should not crash and should return the same path
    assert pg_root_1 == pg_root_2
    mgr.clean()


# ── sync_single_config (lines 118-155) ──────────────────────────

def test_sync_single_config_exact_match(production_workspace):
    """Line 124-129: sync_single_config finds workflow YAML by exact name."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    result = mgr.sync_single_config("gru_Alpha158")
    assert result is True

    mgr.clean()


def test_sync_single_config_via_registry(production_workspace):
    """Lines 131-150: sync_single_config falls back to model_registry lookup."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    import yaml

    # Remove the exact-match file from production so it falls through
    exact_path = production_workspace / "config" / "workflow_config_gru_Alpha158.yaml"
    exact_path.unlink()

    # Create a YAML file referenced by registry
    nested_yaml = production_workspace / "config" / "nested" / "workflow_config_gru_Alpha158.yaml"
    nested_yaml.parent.mkdir(exist_ok=True)
    nested_yaml.write_text("model: {}")

    registry_path = production_workspace / "config" / "model_registry.yaml"
    with open(registry_path, "w") as f:
        yaml.dump({"models": {
            "gru_Alpha158": {"yaml_file": "config/nested/workflow_config_gru_Alpha158.yaml"}
        }}, f)

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    result = mgr.sync_single_config("gru_Alpha158")
    assert result is True

    mgr.clean()


def test_sync_single_config_not_found(production_workspace):
    """Lines 154-155: sync_single_config returns False when no file found."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager

    # Remove the exact-match file
    exact_path = production_workspace / "config" / "workflow_config_gru_Alpha158.yaml"
    exact_path.unlink()
    # And don't have a registry entry for it
    registry_path = production_workspace / "config" / "model_registry.yaml"
    if registry_path.exists():
        registry_path.unlink()

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    result = mgr.sync_single_config("nonexistent_model")
    assert result is False

    mgr.clean()


def test_sync_single_config_registry_error(production_workspace):
    """Lines 151-152: exception during registry lookup."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager

    # Remove exact match file
    exact_path = production_workspace / "config" / "workflow_config_gru_Alpha158.yaml"
    exact_path.unlink()

    # Corrupt registry
    registry_path = production_workspace / "config" / "model_registry.yaml"
    registry_path.write_text(":: not valid yaml :::")

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    result = mgr.sync_single_config("gru_Alpha158")
    assert result is False

    mgr.clean()


# ── get_meta edge case (line 163) ────────────────────────────────

def test_get_meta_file_missing(production_workspace):
    """Line 163: get_meta returns {} when meta file doesn't exist."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    mgr = PlaygroundManager(str(production_workspace))
    # Don't create playground — meta file won't exist
    meta = mgr.get_meta()
    assert meta == {}


def test_sync_data_no_source_data_dir(production_workspace):
    """Line 186: _sync_data returns early when source data/ doesn't exist."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager
    import shutil

    # Remove the data/ directory entirely
    shutil.rmtree(str(production_workspace / "data"))

    mgr = PlaygroundManager(str(production_workspace))
    pg_root = mgr.create_or_sync()

    # Should still succeed — data/ dir is created empty in playground
    assert os.path.isdir(pg_root)
    assert os.path.isdir(os.path.join(pg_root, "data"))

    mgr.clean()


# ── _sync_data edge cases ─────────────────────────────────────────

def test_sync_data_symlink_dir_is_real_dir(production_workspace):
    """Line 198: existing symlink dir is a real directory, uses rmtree."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    # Turn order_history from symlink into real dir
    oh_path = os.path.join(mgr.playground_root, "data", "order_history")
    os.unlink(oh_path)
    os.makedirs(oh_path)
    with open(os.path.join(oh_path, "stale.txt"), "w") as f:
        f.write("old")

    # Resync — should rmtree the real dir and replace with symlink
    mgr.create_or_sync()
    assert os.path.islink(oh_path)

    mgr.clean()


def test_sync_data_skip_entity_copied_files(production_workspace, monkeypatch):
    """Line 217: skip symlink-by-extension for files already entity-copied."""
    from quantpits.scripts.deep_analysis import playground_manager as pm

    # Temporarily add a .csv file to the entity-copy list
    original = list(pm._DATA_ENTITY_COPY_FILES)
    monkeypatch.setattr(pm, "_DATA_ENTITY_COPY_FILES", original + ["test_skip.csv"])

    mgr = pm.PlaygroundManager(str(production_workspace))
    # Create a CSV in production that's also in entity-copy list
    csv_path = production_workspace / "data" / "test_skip.csv"
    csv_path.write_text("a,b\n1,2\n")

    pg_root = mgr.create_or_sync()
    # Should be entity-copied (real file), not symlinked
    pg_csv = os.path.join(pg_root, "data", "test_skip.csv")
    assert os.path.isfile(pg_csv)
    assert not os.path.islink(pg_csv)

    mgr.clean()


def test_sync_data_remove_real_file_for_symlink(production_workspace):
    """Line 223: existing file is real file (not symlink), uses os.remove."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    # Replace symlinked CSV with a real file
    csv_path = os.path.join(mgr.playground_root, "data", "trade_log_full.csv")
    os.unlink(csv_path)
    with open(csv_path, "w") as f:
        f.write("col1,col2\n3,4\n")

    assert not os.path.islink(csv_path)

    # Resync — should remove real file and create symlink
    mgr.create_or_sync()
    assert os.path.islink(csv_path)

    mgr.clean()


def test_sync_data_pretrained_exists_in_playground(production_workspace):
    """Line 232: pretrained dir already exists in playground, rmtree first."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager

    # Create pretrained in production
    pretrained_dir = production_workspace / "data" / "pretrained"
    pretrained_dir.mkdir(exist_ok=True)
    (pretrained_dir / "model.pkl").write_text("weights")

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    # Now add a stale file to playground's pretrained
    pg_pretrained = os.path.join(mgr.playground_root, "data", "pretrained")
    with open(os.path.join(pg_pretrained, "stale_file.txt"), "w") as f:
        f.write("should be removed")

    # Resync — stale file should be gone
    mgr.create_or_sync()
    assert not os.path.exists(os.path.join(pg_pretrained, "stale_file.txt"))
    assert os.path.exists(os.path.join(pg_pretrained, "model.pkl"))

    mgr.clean()


# ── _capture_baseline_snapshot error paths (lines 294-295, 306-307)

def test_capture_baseline_corrupt_training_history(production_workspace):
    """Lines 294-295: exception when parsing training_history.jsonl."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager

    # Corrupt the training_history.jsonl
    (production_workspace / "data" / "training_history.jsonl").write_text(
        "not valid json\n"
    )

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    meta = mgr.get_meta()
    assert meta["baseline_snapshot"]["training_history_latest_date"] is None

    mgr.clean()


def test_capture_baseline_corrupt_fusion_ledger(production_workspace):
    """Lines 306-307: exception when parsing fusion_run_ledger.jsonl."""
    from quantpits.scripts.deep_analysis.playground_manager import PlaygroundManager

    # Corrupt the fusion_run_ledger.jsonl
    (production_workspace / "data" / "fusion_run_ledger.jsonl").write_text(
        "not valid json\n"
    )

    mgr = PlaygroundManager(str(production_workspace))
    mgr.create_or_sync()

    meta = mgr.get_meta()
    assert meta["baseline_snapshot"]["fusion_ledger_latest_date"] is None

    mgr.clean()
