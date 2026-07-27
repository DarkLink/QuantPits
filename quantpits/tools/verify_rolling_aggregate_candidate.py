"""Reusable preflight for the bounded Rolling aggregate candidate gate.

The command is deliberately no-write unless ``--execute`` is present.  The
execute route creates deterministic execution-bound source fixtures without
training, materializes exactly one candidate, and proves zero-write reuse in a
second process.  Substantive scenario policy stays here rather than in a
candidate-specific wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


GATE_PROTOCOL = "rolling_aggregate_candidate_gate_v1"
EXECUTE_AUTHORIZATION = "authorize-rolling-aggregate-candidate-gate-v1"
CLEANUP_AUTHORIZATION = "authorize-rolling-aggregate-gate-cleanup-v1"
WALL_SECONDS = 300
MIN_FREE_BYTES = 2 * 1024 ** 3
MAX_WRITE_BYTES = 512 * 1024 ** 2


class AggregateGateError(RuntimeError):
    pass


def _strict_int(value, field):
    if type(value) is not int:
        raise AggregateGateError("%s must be an exact integer" % field)
    return value


def _strict_bool(value, field):
    if type(value) is not bool:
        raise AggregateGateError("%s must be an exact boolean" % field)
    return value


@dataclass(frozen=True)
class AggregateGateScenario:
    protocol: str
    family: str
    target_count: int
    window_count: int
    source_unit_count: int
    training: bool
    expected_new_recorders: int
    expected_artifacts: tuple
    second_process_writer_calls: int
    second_process_new_recorders: int
    cleanup_default: str

    def __post_init__(self):
        if self.protocol != GATE_PROTOCOL:
            raise AggregateGateError("wrong gate protocol")
        if self.family != "rolling":
            raise AggregateGateError("gate family must be rolling")
        if (
            _strict_int(self.target_count, "target_count") != 1
            or _strict_int(self.window_count, "window_count") != 2
            or _strict_int(self.source_unit_count, "source_unit_count") != 2
        ):
            raise AggregateGateError(
                "selector must be exactly one target and two windows"
            )
        if _strict_bool(self.training, "training"):
            raise AggregateGateError("training is forbidden")
        if _strict_int(
            self.expected_new_recorders, "expected_new_recorders"
        ) != 1:
            raise AggregateGateError("gate requires exactly one new candidate")
        if self.expected_artifacts != (
            "pred.pkl", "aggregate_manifest.json",
        ):
            raise AggregateGateError("candidate artifact set is not exact")
        if (
            _strict_int(
                self.second_process_writer_calls,
                "second_process_writer_calls",
            ) != 0
            or _strict_int(
                self.second_process_new_recorders,
                "second_process_new_recorders",
            ) != 0
        ):
            raise AggregateGateError("reuse must perform zero writes")
        if self.cleanup_default != "preserve":
            raise AggregateGateError("gate cleanup must default to preserve")

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                self.to_public_dict(), sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_public_dict(self):
        return {
            "protocol": self.protocol,
            "family": self.family,
            "target_count": self.target_count,
            "window_count": self.window_count,
            "source_unit_count": self.source_unit_count,
            "training": self.training,
            "expected_new_recorders": self.expected_new_recorders,
            "expected_artifacts": list(self.expected_artifacts),
            "second_process_writer_calls": self.second_process_writer_calls,
            "second_process_new_recorders":
                self.second_process_new_recorders,
            "cleanup_default": self.cleanup_default,
        }


def frozen_scenario():
    return AggregateGateScenario(
        GATE_PROTOCOL, "rolling", 1, 2, 2, False, 1,
        ("pred.pkl", "aggregate_manifest.json"), 0, 0, "preserve",
    )


def scenario_from_mapping(payload):
    if not isinstance(payload, Mapping):
        raise AggregateGateError("scenario root must be a mapping")
    expected = frozenset(frozen_scenario().to_public_dict())
    if frozenset(payload) != expected:
        raise AggregateGateError("scenario fields are not exact")
    artifacts = payload["expected_artifacts"]
    if not isinstance(artifacts, list):
        raise AggregateGateError("expected_artifacts must be a JSON array")
    return AggregateGateScenario(
        payload["protocol"], payload["family"], payload["target_count"],
        payload["window_count"], payload["source_unit_count"],
        payload["training"], payload["expected_new_recorders"],
        tuple(artifacts), payload["second_process_writer_calls"],
        payload["second_process_new_recorders"],
        payload["cleanup_default"],
    )


def _real_directory(path, field):
    path = Path(path).expanduser().absolute()
    if not path.exists() or path.is_symlink() or not path.is_dir():
        raise AggregateGateError("%s must be a real existing directory" % field)
    if path.resolve(strict=True) != path:
        raise AggregateGateError("%s contains a symlink" % field)
    return path


def snapshot_tree(root):
    root = _real_directory(root, "snapshot root")
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        node = path.lstat()
        if stat.S_ISLNK(node.st_mode):
            raise AggregateGateError("snapshot encountered a symlink")
        if path.is_file():
            data = path.read_bytes()
            rows.append((relative, "file", len(data), hashlib.sha256(data).hexdigest()))
        elif path.is_dir():
            rows.append((relative, "directory", None, None))
        else:
            raise AggregateGateError("snapshot encountered a special node")
    return tuple(rows)


def validate_binding(
    scenario, workspace, protected_workspace, commit, tree,
    execute=False, authorization=None,
):
    if not isinstance(scenario, AggregateGateScenario):
        raise AggregateGateError("binding requires a validated scenario")
    workspace = _real_directory(workspace, "disposable workspace")
    protected = _real_directory(
        protected_workspace, "protected workspace",
    )
    if workspace == protected:
        raise AggregateGateError("disposable and protected workspaces differ")
    for value, field in ((commit, "commit"), (tree, "tree")):
        if (
            not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise AggregateGateError("%s must be a full lowercase git id" % field)
    repository = Path(__file__).resolve().parents[2]
    try:
        actual_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repository),
            text=True, timeout=10,
        ).strip()
        actual_tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=str(repository),
            text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AggregateGateError("candidate git identity is unavailable") from exc
    if commit != actual_commit or tree != actual_tree:
        raise AggregateGateError("binding does not match the checked-out candidate")
    if shutil.disk_usage(str(workspace)).free < MIN_FREE_BYTES:
        raise AggregateGateError("disposable filesystem has insufficient capacity")
    if type(execute) is not bool:
        raise AggregateGateError("execute must be an exact boolean")
    if execute and authorization != EXECUTE_AUTHORIZATION:
        raise AggregateGateError("execute authorization is missing or invalid")
    if execute:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repository), text=True, timeout=10,
        )
        if dirty.strip():
            raise AggregateGateError(
                "execute requires a clean tracked candidate"
            )
    return {
        "scenario_fingerprint": scenario.fingerprint,
        "workspace": workspace,
        "protected_workspace": protected,
        "commit": commit,
        "tree": tree,
        "execute": execute,
    }


def preflight_evidence(binding):
    protected = snapshot_tree(binding["protected_workspace"])
    return {
        "protocol": GATE_PROTOCOL,
        "status": "preflight_passed",
        "reason_code": "rolling_aggregate_gate_preflight_passed",
        "scenario_fingerprint": binding["scenario_fingerprint"],
        "commit": binding["commit"],
        "tree": binding["tree"],
        "budgets": {
            "wall_seconds": WALL_SECONDS,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "maximum_write_bytes": MAX_WRITE_BYTES,
            "network": 0,
            "gpu": 0,
            "training_calls": 0,
        },
        "protected_snapshot_fingerprint": hashlib.sha256(
            json.dumps(protected, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "execute_authorized": binding["execute"],
        "cleanup": "preserve",
    }


def _initialize_fixture_tracking(context):
    import qlib
    from qlib.constant import REG_CN

    qlib.init(
        provider_uri=context.qlib_data_dir,
        region=REG_CN,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": context.mlflow_uri,
                "default_exp_name": "Experiment",
            },
        },
    )


def _build_fixture_scope(context):
    from quantpits.model_capabilities.catalog import AUTHORITATIVE_CATALOG
    from quantpits.model_capabilities.inspector import ModelCapabilityInspector
    from quantpits.rolling import (
        PreparedRollingRun,
        ResolvedRollingRun,
        RollingAnchorPolicy,
        RollingRunIdentity,
        RollingRunOptions,
        RollingTarget,
        RollingTargetIdentity,
        RollingWindowDescriptor,
        RollingWindowIdentity,
        build_rolling_execution_scope,
        map_workflow_capability,
        observe_rolling_business_sessions,
        workspace_fingerprint,
    )
    from quantpits.runtime.command import CommandPlan, fingerprint_command_plan
    from quantpits.utils.workspace import fingerprint_value

    declaration = next(
        item for item in AUTHORITATIVE_CATALOG
        if item.model_module == "qlib.contrib.model.linear"
        and item.model_class == "LinearModel"
        and item.dataset_module == "qlib.data.dataset"
        and item.dataset_class == "DatasetH"
        and item.action == "train"
        and item.execution_family == "rolling"
    )
    matrix = ModelCapabilityInspector().inspect((declaration,))
    if not matrix.results[0].preflight_allowed:
        raise AggregateGateError(
            "deterministic fixture capability is unavailable"
        )
    relative = "config/demo_linear_gate.yaml"
    workflow = context.root / relative
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(
        "task:\n"
        "  model: {class: LinearModel, module_path: qlib.contrib.model.linear}\n"
        "  dataset:\n"
        "    class: DatasetH\n"
        "    module_path: qlib.data.dataset\n"
        "    kwargs:\n"
        "      handler:\n"
        "        class: Alpha158\n"
        "        module_path: qlib.contrib.data.handler\n"
        "        kwargs: {}\n"
        "      segments: {}\n",
        encoding="utf-8",
    )
    target_identity = RollingTargetIdentity("demo_linear", "rolling")
    mapped = map_workflow_capability(
        context, target_identity.target_key, relative, matrix,
    )
    prepared_target = RollingTarget(
        target_identity, relative, mapped.workflow_fingerprint,
        "gate_fixture", {},
    )
    identities = (
        RollingWindowIdentity(
            "rolling", "2023-01-01", "2023-12-31",
            "2025-01-05", "2025-01-06", "a" * 64,
            valid_start="2024-01-01", valid_end="2024-12-31",
        ),
        RollingWindowIdentity(
            "rolling", "2023-02-01", "2024-01-31",
            "2025-02-05", "2025-02-06", "b" * 64,
            valid_start="2024-02-01", valid_end="2025-01-31",
        ),
    )
    windows = tuple(
        RollingWindowDescriptor(index, identity, {
            "window_idx": index,
            "train_start": identity.train_start,
            "train_end": identity.train_end,
            "valid_start": identity.valid_start,
            "valid_end": identity.valid_end,
            "test_start": identity.test_start,
            "test_end": identity.test_end,
        })
        for index, identity in enumerate(identities)
    )
    plan = CommandPlan(
        "rolling_train", context.root.name, "aggregate-gate",
        mode="rolling:merge",
    )
    plan_fingerprint = fingerprint_command_plan(plan, length=64)
    prepared = PreparedRollingRun(
        context, RollingRunOptions(action="merge"), (), {},
        (prepared_target,), None, RollingAnchorPolicy("gate_fixture"),
        plan, plan_fingerprint, {},
    )
    runtime_params = {"market": "demo", "benchmark": "demo"}
    identity = RollingRunIdentity(
        workspace_fingerprint(context.root), "rolling", "merge",
        plan_fingerprint, fingerprint_value({}), identities[-1].test_end,
        (mapped.target_key,), tuple(item.window_key for item in windows),
        fingerprint_value(runtime_params),
    )
    resolved = ResolvedRollingRun(
        prepared, identity.anchor_date, runtime_params, windows, identity,
    )
    observed = observe_rolling_business_sessions(
        windows, lambda start, end: (start, end),
    )
    return build_rolling_execution_scope(
        prepared, resolved, tuple(item.window_key for item in observed),
        (mapped,), observed,
    ), runtime_params


class _FixtureExecutionBackend:
    """Delegate real evidence/MLflow work while freezing a tiny calendar."""

    def __init__(self, delegate):
        self._delegate = delegate

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    @staticmethod
    def calendar_sessions(start, end):
        return (start, end)


class _FixtureRunner:
    def __init__(self, context, runtime_params):
        self.context = context
        self.runtime_params = runtime_params
        self.calls = 0

    @property
    def runtime_params_fingerprint(self):
        from quantpits.utils.workspace import fingerprint_value
        return fingerprint_value(self.runtime_params)

    def execute(self, scope, unit, attempt_id):
        import pandas as pd
        from qlib.workflow import R
        from quantpits.rolling import RollingUnitRunnerObservation

        self.calls += 1
        experiment_name = "Rolling_Aggregate_Gate_Sources"
        with R.start(experiment_name=experiment_name):
            recorder = R.get_recorder()
            R.set_tags(
                execution_protocol=scope.execution_protocol_version,
                run_fingerprint=scope.run_identity.fingerprint,
                attempt_id=attempt_id,
                target_key=unit.unit_key[0],
                window_key=unit.unit_key[1],
                source_operation=scope.run_identity.action,
                fixture_kind="deterministic_no_training",
            )
            rows = [
                (pd.Timestamp(session), "DEMO")
                for session in unit.window.expected_sessions
            ]
            prediction = pd.DataFrame(
                {"score": [float(index + 1) for index in range(len(rows))]},
                index=pd.MultiIndex.from_tuples(
                    rows, names=("datetime", "instrument"),
                ),
            )
            recorder.save_objects(
                **{"model.pkl": {"fixture": True}, "pred.pkl": prediction}
            )
            recorder_id = str(recorder.info["id"])
            experiment_id = str(
                R.get_exp(
                    experiment_name=experiment_name, create=False,
                ).id
            )
        return RollingUnitRunnerObservation(
            unit.unit_key, attempt_id, "candidate_success",
            experiment_name, experiment_id, recorder_id,
        )


def _workspace_bytes(snapshot):
    return sum(
        size for _path, kind, size, _digest_value in snapshot
        if kind == "file"
    )


def _assert_gate_budgets(elapsed_seconds, write_bytes):
    if elapsed_seconds > WALL_SECONDS:
        raise AggregateGateError("gate wall-clock budget exceeded")
    if write_bytes > MAX_WRITE_BYTES:
        raise AggregateGateError("gate write-byte budget exceeded")


def _assert_workspace_write_allowlist(before, after):
    before_map = {item[0]: item[1:] for item in before}
    after_map = {item[0]: item[1:] for item in after}
    changed = {
        path for path in set(before_map) | set(after_map)
        if before_map.get(path) != after_map.get(path)
    }
    allowed_exact = {
        "config",
        "config/demo_linear_gate.yaml",
        "data",
        "data/rolling_state.json",
        "data/rolling_state.json.lock",
        "data/locks",
        "data/locks/rolling_aggregate_candidate.lock",
        "data/locks/training_execution.lock",
        "data/aggregate_gate_scenario.json",
        "mlruns",
        "mlruns/.aggregate-gate",
        "output",
        "qlib_data",
    }
    allowed_prefixes = ("mlruns/", "qlib_data/")
    unexpected = tuple(sorted(
        path for path in changed
        if path not in allowed_exact
        and not any(path.startswith(prefix) for prefix in allowed_prefixes)
    ))
    if unexpected:
        raise AggregateGateError("gate observed an undeclared write")
    before_bytes = _workspace_bytes(before)
    after_bytes = _workspace_bytes(after)
    write_bytes = max(0, after_bytes - before_bytes)
    _assert_gate_budgets(0, write_bytes)
    return len(changed), write_bytes


def _snapshot_tracked_repository(root):
    try:
        names = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=str(root), timeout=20,
        ).split(b"\0")
    except (OSError, subprocess.SubprocessError) as exc:
        raise AggregateGateError("tracked repository inventory failed") from exc
    rows = []
    for raw in names:
        if not raw:
            continue
        relative = raw.decode("utf-8")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise AggregateGateError("tracked repository node is not regular")
        data = path.read_bytes()
        rows.append((relative, len(data), hashlib.sha256(data).hexdigest()))
    return tuple(rows)


def _run_real_gate(binding, reuse_only=False):
    started = time.monotonic()
    workspace = binding["workspace"]
    protected_before = snapshot_tree(binding["protected_workspace"])
    repository_root = Path(__file__).resolve().parents[2]
    repository_before = _snapshot_tracked_repository(repository_root)
    workspace_before = snapshot_tree(workspace)
    for name in ("config", "data", "mlruns", "output", "qlib_data"):
        path = workspace / name
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or path.resolve(strict=True).parent != workspace:
            raise AggregateGateError("fixture directory is not contained")
    marker = workspace / "mlruns" / ".aggregate-gate"
    marker.touch(exist_ok=True)
    scenario_marker = workspace / "data" / "aggregate_gate_scenario.json"
    if not scenario_marker.exists():
        scenario_marker.write_text(json.dumps({
            "protocol": GATE_PROTOCOL,
            "scenario_fingerprint": binding["scenario_fingerprint"],
        }, sort_keys=True), encoding="utf-8")
    try:
        marker_payload = json.loads(
            scenario_marker.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        raise AggregateGateError("disposable scenario marker is invalid")
    if marker_payload != {
        "protocol": GATE_PROTOCOL,
        "scenario_fingerprint": binding["scenario_fingerprint"],
    }:
        raise AggregateGateError("disposable scenario marker is stale")
    from quantpits.utils.workspace import WorkspaceContext
    context = WorkspaceContext.from_root(
        workspace, qlib_data_dir=workspace / "qlib_data",
        qlib_region="cn",
    )
    if not str(context.mlflow_uri).startswith("file://"):
        raise AggregateGateError("fixture tracking backend is not file-contained")
    _initialize_fixture_tracking(context)
    scope, runtime_params = _build_fixture_scope(context)
    from quantpits.rolling import (
        QlibMlflowAggregateBackend,
        QlibMlflowExecutionBackend,
        RollingExecutionKernel,
        RollingStateRepository,
        build_rolling_aggregate_scope,
        materialize_rolling_aggregate_candidates,
    )
    repository = RollingStateRepository.for_workspace(context, "rolling")
    source = _FixtureExecutionBackend(QlibMlflowExecutionBackend(context))
    state_view = repository.inspect_readonly()
    runner_calls = 0
    if state_view.inspection.classification == "missing":
        if reuse_only:
            raise AggregateGateError("reuse process found no fixture state")
        runner = _FixtureRunner(context, runtime_params)
        execution = RollingExecutionKernel(
            repository, source, runner,
        ).execute(scope, "gate-execution-attempt")
        runner_calls = runner.calls
        if execution.status != "success" or runner_calls != 2:
            raise AggregateGateError("deterministic source fixture failed")
        state_view = repository.inspect_readonly()
    elif not reuse_only:
        raise AggregateGateError("disposable workspace is stale")
    aggregate = build_rolling_aggregate_scope(
        scope, state_view, "gate-aggregate-attempt",
    )
    candidate_backend = QlibMlflowAggregateBackend(context)
    before_count = candidate_backend.inventory(aggregate)["raw_count"]
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, candidate_backend,
    )
    after_count = candidate_backend.inventory(aggregate)["raw_count"]
    if reuse_only:
        if (
            result.status != "success"
            or result.target_results[0].status != "reused_success"
            or after_count != before_count
            or runner_calls != 0
        ):
            raise AggregateGateError("second-process reuse was not zero-write")
    elif (
        result.status != "success"
        or result.target_results[0].status != "materialized_success"
        or after_count != before_count + 1
    ):
        raise AggregateGateError("candidate materialization was not exact-one")
    candidate = result.target_results[0].candidate
    if candidate.row_count != 4:
        raise AggregateGateError("candidate row cardinality is unexpected")
    if snapshot_tree(binding["protected_workspace"]) != protected_before:
        raise AggregateGateError("protected workspace drifted")
    if _snapshot_tracked_repository(repository_root) != repository_before:
        raise AggregateGateError("tracked repository drifted")
    elapsed = time.monotonic() - started
    workspace_after = snapshot_tree(workspace)
    changed_file_count, write_bytes = _assert_workspace_write_allowlist(
        workspace_before, workspace_after,
    )
    _assert_gate_budgets(elapsed, write_bytes)
    total_bytes = _workspace_bytes(workspace_after)
    return {
        "status": "reused_success" if reuse_only else "materialized_success",
        "candidate_fingerprint": candidate.content_fingerprint,
        "candidate_row_count": candidate.row_count,
        "new_candidate_recorders": after_count - before_count,
        "training_calls": 0,
        "runner_calls": runner_calls,
        "protected_unchanged": True,
        "repository_unchanged": True,
        "elapsed_seconds": round(elapsed, 6),
        "workspace_bytes": total_bytes,
        "write_bytes": write_bytes,
        "changed_path_count": changed_file_count,
    }


def execute_gate(binding):
    primary = _run_real_gate(binding, reuse_only=False)
    command = [
        sys.executable, "-m",
        "quantpits.tools.verify_rolling_aggregate_candidate",
        "--workspace", str(binding["workspace"]),
        "--protected-workspace", str(binding["protected_workspace"]),
        "--commit", binding["commit"], "--tree", binding["tree"],
        "--execute", "--authorization", EXECUTE_AUTHORIZATION,
        "--internal-reuse",
    ]
    child = subprocess.run(
        command, capture_output=True, text=True, timeout=WALL_SECONDS,
    )
    if child.returncode != 0:
        raise AggregateGateError("separate-process reuse failed")
    try:
        reuse = json.loads(child.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise AggregateGateError("separate-process evidence is invalid")
    if (
        reuse.get("status") != "gate_passed"
        or reuse.get("result", {}).get("status") != "reused_success"
        or reuse.get("result", {}).get("new_candidate_recorders") != 0
    ):
        raise AggregateGateError("separate-process reuse evidence failed")
    return {
        "status": "gate_passed",
        "reason_code": "rolling_aggregate_gate_passed",
        "scenario_fingerprint": binding["scenario_fingerprint"],
        "commit": binding["commit"],
        "tree": binding["tree"],
        "result": primary,
        "reuse": reuse["result"],
        "cleanup": "preserved",
    }


def cleanup_gate_workspace(
    workspace, protected_workspace, scenario_fingerprint, authorization,
):
    if authorization != CLEANUP_AUTHORIZATION:
        raise AggregateGateError("cleanup authorization is missing or invalid")
    workspace = _real_directory(workspace, "cleanup workspace")
    protected = _real_directory(
        protected_workspace, "protected workspace",
    )
    repository = Path(__file__).resolve().parents[2]
    if workspace in (protected, repository) or workspace == workspace.parent:
        raise AggregateGateError("cleanup target is protected or broad")
    marker = workspace / "data" / "aggregate_gate_scenario.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AggregateGateError("cleanup scenario marker is unavailable") from exc
    if payload != {
        "protocol": GATE_PROTOCOL,
        "scenario_fingerprint": scenario_fingerprint,
    }:
        raise AggregateGateError("cleanup scenario identity disagrees")
    shutil.rmtree(str(workspace))
    if workspace.exists():
        raise AggregateGateError("cleanup postcondition is uncertain")
    return {
        "status": "cleanup_completed",
        "reason_code": "rolling_aggregate_gate_cleanup_completed",
        "scenario_fingerprint": scenario_fingerprint,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Preflight the bounded immutable Rolling aggregate gate",
    )
    parser.add_argument("--scenario")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--protected-workspace", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--internal-reuse", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--cleanup-authorization")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.cleanup and (args.execute or args.internal_reuse):
            raise AggregateGateError(
                "cleanup and execute/reuse modes are mutually exclusive"
            )
        if args.internal_reuse and not args.execute:
            raise AggregateGateError(
                "internal reuse requires explicit execute mode"
            )
        if args.scenario:
            payload = json.loads(
                Path(args.scenario).read_text(encoding="utf-8"),
            )
            scenario = scenario_from_mapping(payload)
        else:
            scenario = frozen_scenario()
        binding = validate_binding(
            scenario, args.workspace, args.protected_workspace,
            args.commit, args.tree,
            args.execute or args.internal_reuse, args.authorization,
        )
        evidence = (
            cleanup_gate_workspace(
                binding["workspace"], binding["protected_workspace"],
                binding["scenario_fingerprint"],
                args.cleanup_authorization,
            )
            if args.cleanup
            else
            {
                "status": "gate_passed",
                "reason_code": "rolling_aggregate_gate_reuse_passed",
                "result": _run_real_gate(binding, reuse_only=True),
            }
            if args.internal_reuse
            else execute_gate(binding) if args.execute
            else preflight_evidence(binding)
        )
        print(json.dumps(evidence, sort_keys=True))
        return 0
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        print(json.dumps({
            "protocol": GATE_PROTOCOL,
            "status": "blocked",
            "reason_code": "rolling_aggregate_gate_blocked",
            "error": exc.__class__.__name__,
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
