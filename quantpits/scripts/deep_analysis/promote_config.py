"""
Config Promoter for RLFF Phase 4.

Promotes validated configuration changes from the Playground workspace to
Production.  Generates both machine-readable (JSONL) and human-readable
(Markdown) audit records.

Key constraint: Promote only pushes config file changes, NOT model weights.
Promoted configs take effect on the next production training cycle.
"""

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from quantpits.scripts.deep_analysis.config_ledger import (
    annotate_with_llm_context,
    diff_snapshots,
    save_snapshot,
    snapshot_configs,
)

logger = logging.getLogger(__name__)


@dataclass
class PromotePreview:
    """Preview of what a promote operation would do."""

    diff_summary: List[dict] = field(default_factory=list)
    files_to_copy: List[str] = field(default_factory=list)
    production_snapshot_id: str = ""
    playground_snapshot_id: str = ""


@dataclass
class PromoteResult:
    """Result of a promote operation."""

    success: bool
    promoted_files: List[str] = field(default_factory=list)
    promote_record: dict = field(default_factory=dict)
    error: str = ""


class ConfigPromoter:
    """Promote Playground config changes to Production."""

    def __init__(self, playground_root: str, production_root: str):
        self.playground_root = os.path.abspath(playground_root)
        self.production_root = os.path.abspath(production_root)

    def preview(self) -> PromotePreview:
        """Preview the changes that would be promoted.

        Compares config snapshots between playground and production
        to determine what has changed.
        """
        prod_snapshot = snapshot_configs(self.production_root)
        pg_snapshot = snapshot_configs(self.playground_root)

        diffs = diff_snapshots(prod_snapshot, pg_snapshot)

        # Determine which config files differ
        files_to_copy = self._find_changed_config_files()

        return PromotePreview(
            diff_summary=diffs,
            files_to_copy=files_to_copy,
            production_snapshot_id=prod_snapshot.get("snapshot_date", ""),
            playground_snapshot_id=pg_snapshot.get("snapshot_date", ""),
        )

    def promote(
        self,
        action_item_ids: List[str],
        validation_results: List[dict] = None,
        reason: str = "",
    ) -> PromoteResult:
        """Execute the promotion: copy changed configs to production.

        Args:
            action_item_ids: ActionItem IDs associated with this promote.
            validation_results: Validation data to include in the report.
            reason: Human-readable reason for the promote.

        Returns:
            PromoteResult with success/failure and audit record.
        """
        try:
            # 1. Compute diff
            preview = self.preview()
            if not preview.files_to_copy:
                return PromoteResult(
                    success=True,
                    promote_record={"note": "No config changes to promote"},
                )

            # 2. Annotate changes with LLM context
            if preview.diff_summary:
                annotate_with_llm_context(
                    preview.diff_summary,
                    reason=reason or "Promoted via feedback loop",
                    action_item_id=",".join(action_item_ids) if action_item_ids else None,
                )

            # 3. Save pre-promote snapshot of production
            prod_snapshot = snapshot_configs(self.production_root)
            save_snapshot(self.production_root, prod_snapshot)

            # 4. Copy changed files to production
            promoted_files = []
            for rel_path in preview.files_to_copy:
                src = os.path.join(self.playground_root, rel_path)
                dst = os.path.join(self.production_root, rel_path)
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    promoted_files.append(rel_path)
                    logger.info("Promoted: %s", rel_path)

            # 5. Save post-promote snapshot
            post_snapshot = snapshot_configs(self.production_root)
            save_snapshot(self.production_root, post_snapshot)

            # 6. Build promote record
            promote_id = str(uuid.uuid4())
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            date_str = datetime.now().strftime("%Y-%m-%d")

            # Extract concrete changes for the record
            changes_summary = []
            for diff in preview.diff_summary:
                if diff.get("type") == "hyperparam" and "." in diff.get("key", ""):
                    model, param = diff["key"].rsplit(".", 1)
                    changes_summary.append({
                        "param": param,
                        "model": model,
                        "old": diff.get("old"),
                        "new": diff.get("new"),
                    })

            report_rel_path = f"data/promote_history/promote_{date_str}.md"

            promote_record = {
                "promote_id": promote_id,
                "promoted_at": now_str,
                "action_item_ids": action_item_ids,
                "changes": changes_summary,
                "source": "llm_critic",
                "status": "promoted_pending_retrain",
                "retrained_at": None,
                "rolled_back_at": None,
                "rollback_reason": None,
                "validation_result": validation_results or {},
                "reason": reason,
                "human_readable_report": report_rel_path,
            }

            # 7. Write machine-readable JSONL
            self._write_promote_history(promote_record)

            # 8. Generate human-readable Markdown report
            self._write_human_report(
                promote_record,
                preview.diff_summary,
                validation_results,
                date_str,
            )

            # 9. Update CHANGELOG.md
            self._update_changelog(promote_record, changes_summary, date_str)

            # 10. Update playground metadata
            self._update_playground_meta(action_item_ids, promote_id)

            return PromoteResult(
                success=True,
                promoted_files=promoted_files,
                promote_record=promote_record,
            )

        except Exception as e:
            logger.exception("ConfigPromoter.promote failed")
            return PromoteResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_changed_config_files(self) -> List[str]:
        """Compare config/ files between playground and production.

        Returns list of relative paths (e.g. 'config/workflow_config_x.yaml')
        that differ.
        """
        changed = []
        pg_config = os.path.join(self.playground_root, "config")
        prod_config = os.path.join(self.production_root, "config")

        if not os.path.isdir(pg_config):
            return changed

        for root, dirs, files in os.walk(pg_config):
            # Skip backup directory
            dirs[:] = [d for d in dirs if d != "_backup"]
            for fname in files:
                pg_file = os.path.join(root, fname)
                rel_path = os.path.relpath(pg_file, self.playground_root)
                prod_file = os.path.join(self.production_root, rel_path)

                if not os.path.exists(prod_file):
                    changed.append(rel_path)
                    continue

                # Compare file contents
                try:
                    with open(pg_file, "rb") as f1, open(prod_file, "rb") as f2:
                        if f1.read() != f2.read():
                            changed.append(rel_path)
                except Exception:
                    changed.append(rel_path)

        return changed

    def _write_promote_history(self, record: dict):
        """Append promote record to promote_history.jsonl in production."""
        history_path = os.path.join(self.production_root, "data", "promote_history.jsonl")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_human_report(
        self,
        record: dict,
        diffs: List[dict],
        validation_results: Optional[List[dict]],
        date_str: str,
    ):
        """Generate human-readable Markdown promote report."""
        report_dir = os.path.join(self.production_root, "data", "promote_history")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"promote_{date_str}.md")

        lines = [
            f"# Promote Report — {date_str}",
            "",
            f"**Promote ID**: `{record['promote_id']}`",
            f"**Time**: {record['promoted_at']}",
            f"**Status**: `{record['status']}`",
            "",
            "## 变更摘要",
            "",
        ]

        # Changes table
        if record["changes"]:
            lines.append("| Model | Parameter | Before | After |")
            lines.append("|-------|-----------|--------|-------|")
            for ch in record["changes"]:
                lines.append(f"| {ch['model']} | {ch['param']} | {ch['old']} | {ch['new']} |")
            lines.append("")

        # Reason
        if record.get("reason"):
            lines.extend(["## 变更理由", "", record["reason"], ""])

        # Validation results
        if validation_results:
            lines.extend(["## 验证结果", ""])
            lines.append("| Model | Baseline IC | Playground IC | Delta | Passed |")
            lines.append("|-------|------------|--------------|-------|--------|")
            for vr in validation_results:
                if isinstance(vr, dict):
                    lines.append(
                        f"| {vr.get('model', '?')} "
                        f"| {vr.get('baseline_ic', '?'):.6f} "
                        f"| {vr.get('playground_ic', '?'):.6f} "
                        f"| {vr.get('ic_delta', '?'):+.6f} "
                        f"| {'✅' if vr.get('passed') else '❌'} |"
                    )
            lines.append("")

        # ActionItem traceability
        lines.extend([
            "## ActionItem 溯源",
            "",
            f"- **Action Item IDs**: {', '.join(record.get('action_item_ids', []))}",
            f"- **Source**: {record.get('source', 'unknown')}",
            "",
            "## 回退指南",
            "",
            "```bash",
            "# 回退此次 promote 的配置变更：",
            "git checkout HEAD~1 -- config/",
            "```",
            "",
            "> 回退后需要重新训练才能让模型权重与回退的配置匹配。",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("Human-readable promote report: %s", report_path)

    def _update_changelog(self, record: dict, changes: List[dict], date_str: str):
        """Prepend a summary entry to data/CHANGELOG.md."""
        changelog_path = os.path.join(self.production_root, "data", "CHANGELOG.md")

        # Build summary
        models = list({ch["model"] for ch in changes})
        params = list({ch["param"] for ch in changes})
        models_str = ", ".join(models) if models else "N/A"
        params_str = ", ".join(params) if params else "N/A"

        entry_lines = [
            f"## {date_str}: {params_str} 调整 — {len(models)} 模型",
            f"- **来源**: LLM Critic ({', '.join(record.get('action_item_ids', [])[:3])})",
        ]
        for ch in changes[:5]:
            entry_lines.append(
                f"- **变更**: {ch['model']} 的 {ch['param']}: {ch['old']} → {ch['new']}"
            )
        entry_lines.append(
            f"- **状态**: {record.get('status', 'unknown')}"
        )
        report_rel = record.get("human_readable_report", "")
        if report_rel:
            entry_lines.append(f"- [详细报告]({os.path.basename(report_rel)})")
        entry_lines.append("")

        new_entry = "\n".join(entry_lines)

        if os.path.exists(changelog_path):
            with open(changelog_path, "r", encoding="utf-8") as f:
                existing = f.read()
        else:
            existing = "# 配置变更历史\n\n"

        # Insert after the title line
        title_end = existing.find("\n\n")
        if title_end == -1:
            updated = existing + "\n" + new_entry
        else:
            updated = existing[: title_end + 2] + new_entry + existing[title_end + 2:]

        with open(changelog_path, "w", encoding="utf-8") as f:
            f.write(updated)

    def _update_playground_meta(self, action_item_ids: List[str], promote_id: str):
        """Update _playground_meta.json to record the promotion."""
        meta_path = os.path.join(self.playground_root, "_playground_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            meta.setdefault("action_items_applied", []).extend(action_item_ids)
            meta["last_promote_id"] = promote_id
            meta["last_promoted_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Utility: update promote status after training
# ------------------------------------------------------------------

def update_promote_status(workspace_root: str, model_names: List[str] = None):
    """Update promote_history.jsonl: promoted_pending_retrain → active.

    Called by static_train.py / rolling_train.py after training completes.

    Args:
        workspace_root: Production workspace root.
        model_names: If provided, only update records involving these models.
            If None, update all pending records.
    """
    history_path = os.path.join(workspace_root, "data", "promote_history.jsonl")
    if not os.path.exists(history_path):
        return

    updated = False
    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                records.append(line)
                continue

            if rec.get("status") == "promoted_pending_retrain":
                # Check model filter
                if model_names:
                    rec_models = {ch.get("model") for ch in rec.get("changes", [])}
                    if not rec_models.intersection(set(model_names)):
                        records.append(json.dumps(rec, ensure_ascii=False))
                        continue

                rec["status"] = "active"
                rec["retrained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated = True
                logger.info(
                    "Promote %s status updated: pending_retrain → active",
                    rec.get("promote_id", "?"),
                )

            records.append(json.dumps(rec, ensure_ascii=False))

    if updated:
        with open(history_path, "w", encoding="utf-8") as f:
            f.write("\n".join(records) + "\n")
