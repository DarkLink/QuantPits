"""
Stage Runner for the MAS Deep Analysis System (Phase 1).

Orchestrates sequential stages of deep analysis:
1. discover: file scanning and initial window generation
2. agents: specialist agent execution
3. synthesis: cross-agent synthesis
4. window_analysis: training window rules
5. signals: signal extraction
6. critic: action item generation and validation
7. report: summary and markdown rendering

Supports checkpointing (save/load) of each stage's output.
Stages self-register via ``@register_stage``; the DAG is built dynamically.
Workspace-local custom stages can be loaded via ``pipeline_manifest.json``.
"""

from __future__ import annotations

import os
import json
import time
import logging
import uuid
import importlib
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global stage registry (engine-side built-in stages register here)
# ------------------------------------------------------------------

STAGE_REGISTRY: Dict[str, dict] = {}
"""Global registry of all known pipeline stages.

Each entry:
    name: str           — stage identifier (e.g. 'discover', 'agents')
    depends_on: List[str] — upstream stages that must complete first
    provides: List[str]   — DeepAnalysisState fields this stage populates
    fn: Callable          — stage function (state, **kwargs) -> state
    description: str      — human-readable summary
    source: str           — 'builtin' or 'workspace'
    workspace_root: str|None — non-None when source=='workspace'
"""


def register_stage(name: str, depends_on: List[str] = None,
                   provides: List[str] = None,
                   description: str = ""):
    """Decorator: register a function as a pipeline stage.

    Each stage self-declares its dependencies and outputs.  The
    execution DAG is built dynamically from all registered stages.

    Example::

        @register_stage(
            name='signals',
            depends_on=['agents', 'synthesis', 'window_analysis'],
            provides=['signals'],
            description='Extract structured Signals from agent findings',
        )
        def run_signals(state, **kwargs):
            ...
    """
    def decorator(fn):
        STAGE_REGISTRY[name] = {
            'name': name,
            'depends_on': list(depends_on or []),
            'provides': list(provides or []),
            'fn': fn,
            'description': description,
            'source': 'builtin',
            'workspace_root': None,
        }
        logger.debug("Registered stage '%s' (deps=%s, provides=%s)",
                      name, STAGE_REGISTRY[name]['depends_on'],
                      STAGE_REGISTRY[name]['provides'])
        return fn
    return decorator


def unregister_workspace_stages(workspace_root: str) -> int:
    """Remove all stages registered from *workspace_root*.

    Called automatically after pipeline execution to avoid
    cross-workspace pollution of the global registry.

    Returns:
        Number of stages removed.
    """
    to_remove = [
        name for name, meta in STAGE_REGISTRY.items()
        if meta.get('workspace_root') == workspace_root
    ]
    for name in to_remove:
        del STAGE_REGISTRY[name]
    if to_remove:
        logger.debug("Unregistered %d workspace stages from %s",
                      len(to_remove), workspace_root)
    return len(to_remove)


# ------------------------------------------------------------------
@dataclass
class CheckpointMeta:
    stage: str
    date: str
    label: str
    timestamp: str
    predecessor: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class StageRunner:
    """Orchestrates staged execution of deep analysis with checkpointing."""

    STAGES = [
        'discover',
        'agents',
        'synthesis',
        'window_analysis',
        'signals',
        'critic',
        'report'
    ]

    def __init__(self, workspace_root: str, run_date: str, run_label: str = ""):
        self.workspace_root = workspace_root
        self.run_date = run_date
        self.run_label = run_label.strip()
        self.checkpoint_dir = os.path.join(workspace_root, 'output', 'deep_analysis', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Serialization / Deserialization helpers
    # ------------------------------------------------------------------

    def _serialize_val(self, val: Any) -> Any:
        if val is None or isinstance(val, (str, int, float, bool)):
            return val
        if hasattr(val, '_mock_return_value') or 'mock' in str(type(val)).lower():
            return f"MockObject({type(val).__name__})"
        if isinstance(val, (list, tuple, set)):
            return [self._serialize_val(x) for x in val]
        if isinstance(val, dict):
            return {str(k): self._serialize_val(v) for k, v in val.items()}
        if isinstance(val, pd.DataFrame):
            # If large, use parquet sidecar
            if len(val) > 10000:
                sidecar_name = f"sidecar_{uuid.uuid4().hex}.parquet"
                sidecar_path = os.path.join(self.checkpoint_dir, sidecar_name)
                val.to_parquet(sidecar_path)
                return {
                    "_type": "parquet_ref",
                    "path": sidecar_name
                }
            else:
                return {
                    "_type": "dataframe",
                    "columns": list(val.columns),
                    "records": val.to_dict(orient='records')
                }
        if isinstance(val, pd.Series):
            return {
                "_type": "series",
                "name": val.name,
                "index": list(val.index),
                "data": list(val.values)
            }
        if hasattr(val, 'to_dict'):
            try:
                return val.to_dict()
            except Exception:
                pass
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.isoformat()
        if hasattr(val, '__dataclass_fields__'):
            return asdict(val)
        return str(val)

    def _deserialize_val(self, val: Any) -> Any:
        if isinstance(val, dict):
            if val.get("_type") == "dataframe":
                # Convert records dict back to pandas DataFrame
                return pd.DataFrame(val["records"], columns=val["columns"])
            if val.get("_type") == "parquet_ref":
                sidecar_path = os.path.join(self.checkpoint_dir, val["path"])
                if os.path.exists(sidecar_path):
                    return pd.read_parquet(sidecar_path)
                logger.error(f"Sidecar Parquet file not found: {sidecar_path}")
                return pd.DataFrame()
            if val.get("_type") == "series":
                return pd.Series(val["data"], index=val["index"], name=val["name"])

            # Findings reconstruction
            if "severity" in val and "category" in val and "title" in val and "detail" in val:
                # Do NOT import at module-level to avoid circular imports
                from quantpits.scripts.deep_analysis.base_agent import Finding
                return Finding(
                    severity=val["severity"],
                    category=val["category"],
                    title=val["title"],
                    detail=val["detail"],
                    data=self._deserialize_val(val.get("data", {}))
                )
            if "agent_name" in val and "window_label" in val and "findings" in val:
                from quantpits.scripts.deep_analysis.base_agent import AgentFindings
                return AgentFindings(
                    agent_name=val["agent_name"],
                    window_label=val["window_label"],
                    findings=self._deserialize_val(val["findings"]),
                    recommendations=val.get("recommendations", []),
                    raw_metrics=self._deserialize_val(val.get("raw_metrics", {}))
                )

            # Signal reconstruction
            if "signal_type" in val and "severity" in val and "scope" in val and "source_agent" in val:
                from quantpits.scripts.deep_analysis.signal_extractor import Signal
                return Signal(
                    signal_type=val["signal_type"],
                    severity=val["severity"],
                    scope=val["scope"],
                    source_agent=val["source_agent"],
                    target=val.get("target", ""),
                    metrics=self._deserialize_val(val.get("metrics", {})),
                    context=val.get("context", "")
                )

            # ActionItem reconstruction
            if "action_id" in val and "action_type" in val and "scope" in val:
                from quantpits.scripts.deep_analysis.action_items import ActionItem
                return ActionItem(
                    action_id=val["action_id"],
                    action_type=val["action_type"],
                    scope=val["scope"],
                    target=val.get("target", ""),
                    params=self._deserialize_val(val.get("params", {})),
                    reason=val.get("reason", ""),
                    source_signals=val.get("source_signals", []),
                    expected_outcome=val.get("expected_outcome", ""),
                    confidence=val.get("confidence", 0.0),
                    risk_level=val.get("risk_level", "medium"),
                    scope_status=val.get("scope_status", "pending"),
                    rejected_reason=val.get("rejected_reason", ""),
                    validated_at=val.get("validated_at", ""),
                    execution_context=val.get("execution_context", {})
                )

            return {k: self._deserialize_val(v) for k, v in val.items()}

        if isinstance(val, list):
            return [self._deserialize_val(x) for x in val]
        return val

    # ------------------------------------------------------------------
    # Checkpoint File operations
    # ------------------------------------------------------------------

    def save_checkpoint(self, stage: str, data: dict, predecessor: Optional[str] = None) -> str:
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage name: {stage}. Must be one of {self.STAGES}")

        meta = CheckpointMeta(
            stage=stage,
            date=self.run_date,
            label=self.run_label,
            timestamp=datetime.now().isoformat(),
            predecessor=predecessor
        )

        serialized_data = self._serialize_val(data)
        payload = {
            "_meta": meta.to_dict(),
            "data": serialized_data
        }

        label_suffix = f"_{self.run_label}" if self.run_label else ""
        filename = f"{stage}_{self.run_date}{label_suffix}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Checkpoint saved for stage '{stage}' at {filepath}")
        return filepath

    def load_checkpoint(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        data = self._deserialize_val(payload.get("data", {}))
        meta_dict = payload.get("_meta", {})
        
        # Verify dates / stages match if applicable
        logger.info(f"Loaded checkpoint for stage '{meta_dict.get('stage')}' dated {meta_dict.get('date')}")
        return {
            "meta": meta_dict,
            "data": data
        }

    def find_latest_checkpoint(self, stage: str) -> Optional[str]:
        # Accept any registered stage (not just the legacy STAGES list)
        valid = set(self.STAGES) | set(STAGE_REGISTRY.keys())
        if stage not in valid:
            raise ValueError(f"Unknown stage: '{stage}'. Registered: {sorted(valid)}")

        label_suffix = f"_{self.run_label}" if self.run_label else ""
        prefix = f"{stage}_{self.run_date}{label_suffix}"

        candidates = []
        if os.path.isdir(self.checkpoint_dir):
            for entry in os.listdir(self.checkpoint_dir):
                if entry.startswith(prefix) and entry.endswith('.json'):
                    candidates.append(os.path.join(self.checkpoint_dir, entry))

        if not candidates:
            return None

        if not candidates:
            return None

        # Return the newest based on file mtime
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # DAG building and plan resolution
    # ------------------------------------------------------------------

    def build_dag(self) -> Dict[str, List[str]]:
        """Build an adjacency list from all registered stages.

        Returns:
            ``{stage_name: [predecessor_names], ...}`` — suitable for
            topological sort.
        """
        dag: Dict[str, List[str]] = {}
        for name, meta in STAGE_REGISTRY.items():
            dag[name] = list(meta.get('depends_on', []))
        return dag

    @staticmethod
    def _topological_sort(dag: Dict[str, List[str]]) -> List[str]:
        """Kahn's algorithm — return stages in dependency order.

        Raises:
            ValueError: If the DAG contains a cycle.
        """
        in_degree: Dict[str, int] = {n: 0 for n in dag}
        for name, deps in dag.items():
            for dep in deps:
                if dep not in in_degree:
                    in_degree[dep] = 0
                in_degree[name] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for other, deps in dag.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(result) != len(dag):
            missing = set(dag) - set(result)
            raise ValueError(
                f"Cycle detected in pipeline DAG. "
                f"Unresolved stages: {sorted(missing)}"
            )

        return result

    def resolve_plan(
        self, target: str, completed_stages: List[str],
        provides_map: Dict[str, List[str]] = None,
        state: Any = None,
    ) -> List[str]:
        """Return ordered list of stages to execute up to *target*.

        Stages whose outputs are already present (via checkpoint or
        prior execution) are skipped.

        Args:
            target: Stage name to stop at (inclusive).
            completed_stages: Names of stages already done.
            provides_map: ``{stage: [field_names]}`` — if omitted,
                          built from ``STAGE_REGISTRY``.
            state: ``DeepAnalysisState`` used to check field
                   population when *provides_map* is given.

        Returns:
            Ordered stage names to execute.
        """
        dag = self.build_dag()
        try:
            sorted_stages = self._topological_sort(dag)
        except ValueError as exc:
            raise StageDependencyError(str(exc)) from exc

        if provides_map is None:
            provides_map = {
                name: meta.get('provides', [])
                for name, meta in STAGE_REGISTRY.items()
            }

        plan: List[str] = []
        for stage in sorted_stages:
            if stage == target:
                if state is None or not state.is_stage_done(stage, provides_map.get(stage, [])):
                    plan.append(stage)
                break

            if stage in completed_stages:
                continue
            if state is not None and state.is_stage_done(stage, provides_map.get(stage, [])):
                continue

            # Verify dependencies can be satisfied: each dep must either
            # be already completed OR scheduled earlier in this plan.
            deps = dag.get(stage, [])
            satisfiable = (
                set(deps)
                <= set(completed_stages)
                | {s for s in plan if s not in completed_stages}
                | {
                    s for s in sorted_stages
                    if state and state.is_stage_done(s, provides_map.get(s, []))
                }
            )
            if not satisfiable:
                missing = set(deps) - set(completed_stages) - set(plan)
                raise StageDependencyError(
                    f"Stage '{stage}' depends on {sorted(missing)}, "
                    f"which are not completed and not in the execution plan."
                )
            plan.append(stage)

        return plan

    # ------------------------------------------------------------------
    # Workspace-local stage loading
    # ------------------------------------------------------------------

    def load_workspace_stages(
        self, workspace_root: str, manifest_path: str = None,
    ) -> Dict[str, dict]:
        """Load custom stages declared in a workspace-local manifest.

        Pattern mirrors ``load_manifest_agents()`` in
        ``quantpits/scripts/deep_analysis/agents/__init__.py``.

        Manifest format::

            {
                "stages": [
                    {
                        "name": "custom_check",
                        "depends_on": ["signals"],
                        "provides": ["check_report"],
                        "class_path": "custom_stages.check.run",
                        "insert_after": "signals",
                        "enabled": true
                    }
                ]
            }

        Returns:
            Dict of ``{stage_name: stage_meta}`` for newly loaded stages.
        """
        import sys

        ws_added = False
        loaded: Dict[str, dict] = {}

        manifests_to_try: List[str] = []
        if manifest_path:
            manifests_to_try.append(
                manifest_path if os.path.isabs(manifest_path)
                else os.path.join(workspace_root, manifest_path)
            )
        else:
            manifests_to_try.append(
                os.path.join(workspace_root, "config", "pipeline_manifest.json"))
            manifests_to_try.append(
                os.path.join(workspace_root, "config", "pipeline_manifest.yaml"))

        try:
            if workspace_root and workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
                ws_added = True

            manifest_specs: List[dict] = []
            for path in manifests_to_try:
                if not os.path.exists(path):
                    continue
                try:
                    manifest_specs = self._parse_manifest(path)
                except Exception as exc:
                    logger.debug("Failed to parse manifest %s: %s", path, exc)
                    continue
                break  # first found manifest wins

            if not manifest_specs:
                return loaded

            for spec in manifest_specs:
                if not spec.get('enabled', True):
                    continue
                name = spec.get('name')
                class_path = spec.get('class_path')
                if not (name and class_path):
                    continue

                try:
                    module_path, fn_name = class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    fn = getattr(module, fn_name)
                    # If fn used @register_stage, it's already in STAGE_REGISTRY.
                    if name in STAGE_REGISTRY:
                        STAGE_REGISTRY[name]['source'] = 'workspace'
                        STAGE_REGISTRY[name]['workspace_root'] = workspace_root
                    loaded[name] = STAGE_REGISTRY.get(name, {})
                    print(f"  🔌 Loaded workspace stage: {name} from {class_path}")
                except Exception as exc:
                    print(f"  ❌ Failed to load stage '{name}' from "
                          f"'{class_path}': {exc}")

            # Apply insertion hints (insert_after / insert_before)
            self._apply_insertion_hints(manifest_specs)

        finally:
            if ws_added and workspace_root in sys.path:
                sys.path.remove(workspace_root)

        return loaded

    @staticmethod
    def _parse_manifest(path: str) -> List[dict]:
        """Parse a pipeline manifest file (JSON or YAML).

        Returns:
            List of stage specification dicts (the ``"stages"`` key).
        """
        if path.endswith('.yaml') or path.endswith('.yml'):
            import yaml
            with open(path, 'r', encoding='utf-8') as fh:
                data = yaml.safe_load(fh) or {}
        else:
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh) or {}
        return list(data.get('stages', []))

    def _apply_insertion_hints(self, manifest_specs: List[dict]) -> None:
        """Process ``insert_after`` / ``insert_before`` hints.

        - ``insert_after="X"``: inherit X's depends_on + add X itself;
          downstream stages that previously depended on X are rewired
          to depend on the new stage instead.
        - ``insert_before="Y"``: the new stage's depends_on is set to
          Y's current depends_on, and Y's depends_on is updated to
          include the new stage.
        """
        for spec in manifest_specs:
            name = spec.get('name', '')
            meta = STAGE_REGISTRY.get(name)
            if not meta:
                continue

            insert_after = spec.get('insert_after')
            insert_before = spec.get('insert_before')

            if insert_after:
                ref = STAGE_REGISTRY.get(insert_after)
                if ref:
                    meta['depends_on'] = (
                        list(ref.get('depends_on', [])) + [insert_after]
                    )
                    for other_name, other_meta in STAGE_REGISTRY.items():
                        if other_name == name:
                            continue
                        deps = other_meta.get('depends_on', [])
                        if insert_after in deps:
                            deps.remove(insert_after)
                            deps.append(name)

            elif insert_before:
                ref = STAGE_REGISTRY.get(insert_before)
                if ref:
                    meta['depends_on'] = list(ref.get('depends_on', []))
                    ref_deps = ref.get('depends_on', [])
                    if name not in ref_deps:
                        ref_deps.append(name)

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def run(
        self, target: str = "report",
        state: Any = None,
        workspace_root: str = "",
        manifest_path: str = None,
        **stage_kwargs,
    ):
        """Execute pipeline from current state to *target* stage.

        1. Load workspace-local custom stages (if any)
        2. Build DAG from all registered stages
        3. Resolve execution plan via topological sort
        4. Execute each stage in order, checkpoint after each
        5. Clean up workspace stages
        """
        from quantpits.scripts.deep_analysis.state import DeepAnalysisState

        # 1. Load workspace-local stages
        if workspace_root:
            self.load_workspace_stages(workspace_root, manifest_path)

        try:
            if state is None:
                state = DeepAnalysisState(workspace_root=workspace_root)

            provides_map = {
                name: meta.get('provides', [])
                for name, meta in STAGE_REGISTRY.items()
            }

            # 2a. Auto-load available checkpoints into state so that
            #     resolve_plan() can skip already-completed upstream stages.
            #     This is what makes --stage X truly "run-only".
            dag = self.build_dag()
            try:
                sorted_all = self._topological_sort(dag)
            except ValueError:
                sorted_all = list(dag.keys())

            for stage_name in sorted_all:
                if state.is_stage_done(stage_name, provides_map.get(stage_name, [])):
                    continue
                ckpt_path = self.find_latest_checkpoint(stage_name)
                if ckpt_path:
                    try:
                        loaded = self.load_checkpoint(ckpt_path)
                        ckpt_data = loaded.get("data", {})
                        # Populate state fields from checkpoint
                        for key in provides_map.get(stage_name, []):
                            if key in ckpt_data and not state.has(key):
                                setattr(state, key, ckpt_data[key])
                        state.mark_completed(stage_name)
                        print(f"   📥 Auto-loaded checkpoint: {stage_name}")
                    except Exception as exc:
                        logger.debug("Failed to auto-load checkpoint %s: %s",
                                     stage_name, exc)

            # 2b-3. Resolve plan
            plan = self.resolve_plan(
                target, state.completed_stages, provides_map, state,
            )

            if not plan:
                print(f"  ℹ️  Target stage '{target}' already completed — nothing to run.")
                return state

            # 4. Execute
            for stage_name in plan:
                meta = STAGE_REGISTRY.get(stage_name)
                if not meta:
                    raise StageDependencyError(
                        f"Stage '{stage_name}' not found in registry."
                    )

                fn = meta['fn']
                print(f"\n--- [Stage: {stage_name}] ---")
                start = time.time()

                # Execute the stage function
                state = fn(state, **stage_kwargs)

                elapsed = time.time() - start
                state.mark_completed(stage_name)

                # Checkpoint
                try:
                    self.save_checkpoint(stage_name, state.to_dict())
                except Exception as exc:
                    logger.warning(
                        "Failed to save checkpoint for stage '%s': %s",
                        stage_name, exc,
                    )

                print(f"  ✅ {stage_name} completed in {elapsed:.1f}s")

            return state

        finally:
            # 5. Clean up workspace-local stages
            if workspace_root:
                unregister_workspace_stages(workspace_root)


class StageDependencyError(Exception):
    """Raised when a stage's dependencies cannot be satisfied."""
    pass
