"""
Playground Manager for RLFF Phase 4.

Manages the lifecycle of a Playground workspace — a lightweight sibling copy
of the production workspace used to safely execute LLM-suggested configuration
changes before promoting them to production.

Key design decisions:
- Playground is a sibling directory: {workspace_name}_Playground
- data/ is a real directory with selective symlinks for read-only data
- Log files (training_history.jsonl, etc.) are entity copies for isolation
- config/ is fully copied so adapters can modify freely
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Sync rules -----------------------------------------------------------
# Files in data/ that must be entity-copied (training writes to these)
_DATA_ENTITY_COPY_FILES = [
    "training_history.jsonl",
    "fusion_run_ledger.jsonl",
    "operator_log.jsonl",
    "action_item_history.jsonl",
    "run_state.json",
    "rolling_state.json",
]

# Root-level files to entity-copy
_ROOT_ENTITY_COPY_FILES = [
    "latest_train_records.json",
]

# Directories under data/ to symlink (read-only shared data)
_DATA_SYMLINK_DIRS = [
    "order_history",
    "history",
    "config_history",
]

# File extensions in data/ to symlink (large read-only files)
_DATA_SYMLINK_EXTENSIONS = {".xlsx", ".csv"}

# Directories to NOT sync
_SKIP_DIRS = {"output", "archive", "mlruns", "catboost_info", "__pycache__"}


class PlaygroundManager:
    """Manage Playground workspace lifecycle (create, sync, clean)."""

    def __init__(self, production_root: str):
        """
        Args:
            production_root: Absolute path to the production workspace.
        """
        self.production_root = os.path.abspath(production_root)
        workspace_name = os.path.basename(self.production_root)
        parent_dir = os.path.dirname(self.production_root)
        self.playground_root = os.path.join(parent_dir, f"{workspace_name}_Playground")

    def create_or_sync(self) -> str:
        """Create the Playground (or re-sync if it already exists).

        Returns:
            Absolute path to the Playground root.
        """
        exists = os.path.isdir(self.playground_root)
        action = "Syncing" if exists else "Creating"
        logger.info("%s Playground at %s", action, self.playground_root)

        os.makedirs(self.playground_root, exist_ok=True)

        # 1. Sync config/ — full copy
        self._sync_config()

        # 2. Sync data/ — mixed strategy
        self._sync_data()

        # 3. Sync root-level entity files
        self._sync_root_files()

        # 4. Ensure output dir exists (independent)
        os.makedirs(os.path.join(self.playground_root, "output", "deep_analysis"), exist_ok=True)

        # 5. Write/update metadata
        self._write_meta(existed=exists)

        logger.info("Playground ready at %s", self.playground_root)
        return self.playground_root

    def get_playground_root(self) -> Optional[str]:
        """Return the Playground path if it exists, else None."""
        if os.path.isdir(self.playground_root):
            return self.playground_root
        return None

    def clean(self):
        """Remove the Playground entirely."""
        if os.path.isdir(self.playground_root):
            shutil.rmtree(self.playground_root)
            logger.info("Cleaned Playground at %s", self.playground_root)

    def sync_single_config(self, model_name: str) -> bool:
        """Re-copy a single model's workflow YAML from production to Playground.

        Used by the independent experiment mode to reset a model's config
        before each round so that only one parameter change is active.

        Returns:
            True if a file was copied, False if not found.
        """
        import glob

        src_config = os.path.join(self.production_root, "config")
        dst_config = os.path.join(self.playground_root, "config")

        # Try exact match first
        src_file = os.path.join(src_config, f"workflow_config_{model_name}.yaml")
        if os.path.exists(src_file):
            dst_file = os.path.join(dst_config, f"workflow_config_{model_name}.yaml")
            shutil.copy2(src_file, dst_file)
            logger.debug("Reset config for %s from production", model_name)
            return True

        # Fallback: search model_registry.yaml for the yaml_file path
        registry_path = os.path.join(src_config, "model_registry.yaml")
        if os.path.exists(registry_path):
            try:
                import yaml
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = yaml.safe_load(f) or {}
                model_info = registry.get("models", {}).get(model_name, {})
                yaml_rel = model_info.get("yaml_file", "")
                if yaml_rel:
                    src_file = os.path.join(self.production_root, yaml_rel)
                    dst_file = os.path.join(self.playground_root, yaml_rel)
                    if os.path.exists(src_file):
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        logger.debug(
                            "Reset config for %s via registry: %s",
                            model_name, yaml_rel,
                        )
                        return True
            except Exception as e:
                logger.warning("Failed to look up %s in registry: %s", model_name, e)

        logger.warning("No config file found for model %s", model_name)
        return False

    def get_meta(self) -> dict:
        """Read _playground_meta.json."""
        meta_path = os.path.join(self.playground_root, "_playground_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ------------------------------------------------------------------
    # Internal sync helpers
    # ------------------------------------------------------------------

    def _sync_config(self):
        """Full copy of config/ directory."""
        src = os.path.join(self.production_root, "config")
        dst = os.path.join(self.playground_root, "config")
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.debug("Synced config/ (full copy)")

    def _sync_data(self):
        """Mixed sync of data/ directory per plan §1.2."""
        src_data = os.path.join(self.production_root, "data")
        dst_data = os.path.join(self.playground_root, "data")
        os.makedirs(dst_data, exist_ok=True)

        if not os.path.isdir(src_data):
            return

        # 1. Symlink specified subdirectories
        for dirname in _DATA_SYMLINK_DIRS:
            src_dir = os.path.join(src_data, dirname)
            dst_dir = os.path.join(dst_data, dirname)
            if os.path.isdir(src_dir):
                # Remove existing link/dir so we can re-create
                if os.path.islink(dst_dir) or os.path.isdir(dst_dir):
                    if os.path.islink(dst_dir):
                        os.unlink(dst_dir)
                    else:
                        shutil.rmtree(dst_dir)
                os.symlink(os.path.abspath(src_dir), dst_dir)
                logger.debug("Symlinked data/%s", dirname)

        # 2. Entity-copy log/state files
        for filename in _DATA_ENTITY_COPY_FILES:
            src_file = os.path.join(src_data, filename)
            dst_file = os.path.join(dst_data, filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                logger.debug("Entity-copied data/%s", filename)

        # 3. Symlink individual data files by extension (xlsx, csv)
        for entry in os.scandir(src_data):
            if entry.is_file():
                _, ext = os.path.splitext(entry.name)
                if ext.lower() in _DATA_SYMLINK_EXTENSIONS:
                    # Skip files already entity-copied
                    if entry.name in _DATA_ENTITY_COPY_FILES:
                        continue
                    dst_file = os.path.join(dst_data, entry.name)
                    if os.path.islink(dst_file) or os.path.exists(dst_file):
                        if os.path.islink(dst_file):
                            os.unlink(dst_file)
                        else:
                            os.remove(dst_file)
                    os.symlink(os.path.abspath(entry.path), dst_file)
                    logger.debug("Symlinked data/%s", entry.name)

        # 4. Entity-copy pretrained/ dir (future-proofing, currently no-op)
        src_pretrained = os.path.join(src_data, "pretrained")
        dst_pretrained = os.path.join(dst_data, "pretrained")
        if os.path.isdir(src_pretrained):
            if os.path.isdir(dst_pretrained):
                shutil.rmtree(dst_pretrained)
            shutil.copytree(src_pretrained, dst_pretrained)
            logger.debug("Entity-copied data/pretrained/")

    def _sync_root_files(self):
        """Entity-copy root-level files (e.g. latest_train_records.json)."""
        for filename in _ROOT_ENTITY_COPY_FILES:
            src = os.path.join(self.production_root, filename)
            dst = os.path.join(self.playground_root, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                logger.debug("Entity-copied root/%s", filename)

    def _write_meta(self, existed: bool):
        """Write _playground_meta.json with sync metadata."""
        now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        meta_path = os.path.join(self.playground_root, "_playground_meta.json")

        if existed and os.path.exists(meta_path):
            # Update existing
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["synced_at"] = now_str
        else:
            meta = {
                "created_at": now_str,
                "synced_at": now_str,
                "source_workspace": self.production_root,
                "action_items_applied": [],
                "pretrained_models": {},
                "baseline_snapshot": self._capture_baseline_snapshot(),
                "status": "active",
            }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _capture_baseline_snapshot(self) -> dict:
        """Capture baseline info from production for the meta file."""
        snapshot = {
            "config_snapshot_id": None,
            "training_history_latest_date": None,
            "fusion_ledger_latest_date": None,
        }

        # Try to find latest config snapshot
        config_hist_dir = os.path.join(self.production_root, "data", "config_history")
        if os.path.isdir(config_hist_dir):
            import glob
            snapshots = sorted(glob.glob(os.path.join(config_hist_dir, "config_snapshot_*.json")))
            if snapshots:
                snapshot["config_snapshot_id"] = os.path.basename(snapshots[-1])

        # Get latest date from training_history.jsonl
        th_path = os.path.join(self.production_root, "data", "training_history.jsonl")
        if os.path.exists(th_path):
            try:
                with open(th_path, "r") as f:
                    lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    snapshot["training_history_latest_date"] = last.get("trained_at", last.get("date"))
            except Exception:
                pass

        # Get latest date from fusion_run_ledger.jsonl
        frl_path = os.path.join(self.production_root, "data", "fusion_run_ledger.jsonl")
        if os.path.exists(frl_path):
            try:
                with open(frl_path, "r") as f:
                    lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    snapshot["fusion_ledger_latest_date"] = last.get("run_date", last.get("date"))
            except Exception:
                pass

        return snapshot
