"""Import-pure contracts for evidence-bound exact Rolling unit execution."""

from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Protocol

from quantpits.model_capabilities.contracts import (
    ModelCapabilityIdentity,
    ModelCapabilityMatrix,
    ModelCapabilityResult,
)
from quantpits.rolling.errors import (
    RollingExecutionContractError,
)
from quantpits.rolling.identity import (
    RollingRunIdentity,
    RollingTargetIdentity,
    parse_rolling_window_key,
    workspace_fingerprint,
)
from quantpits.rolling.command import PreparedRollingRun
from quantpits.rolling.windows import (
    ResolvedRollingRun,
    RollingWindowExecutionDescriptor,
    rolling_sessions_fingerprint,
)
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value
from quantpits.runtime.command import CommandPlan, fingerprint_command_plan


EXECUTION_PROTOCOL_VERSION = "rolling_execution_v1"
UNIT_STATUSES = (
    "executed_success", "reused_success", "failed", "blocked", "interrupted",
)
BATCH_STATUSES = ("success", "failed", "blocked", "interrupted")
_DIGEST_CHARS = frozenset("0123456789abcdef")
_TARGET_TOKEN = object()
_SCOPE_TOKEN = object()
_RESULT_TOKEN = object()
_BATCH_TOKEN = object()
_RUNTIME_TOKEN = object()


def _contract(message):
    raise RollingExecutionContractError(message)


def _digest(value, field_name):
    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in _DIGEST_CHARS for char in value)
    ):
        _contract("%s must be a lowercase SHA-256" % field_name)
    return value


def _public_text(value, field_name):
    if not isinstance(value, str) or not value or value != value.strip():
        _contract("%s must be a non-empty trimmed string" % field_name)
    lowered = value.lower()
    windows_path = len(value) > 2 and value[0].isalpha() and value[1] == ":" and value[2] in ("/", "\\")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _contract("%s contains a control character" % field_name)
    if value.startswith(("/", "\\")) or windows_path or "://" in value or lowered.startswith("file:"):
        _contract("%s must not expose an absolute path or URI" % field_name)
    return value


def _canonical_relative_path(value):
    value = _public_text(value, "workflow_relative_path")
    path = Path(value)
    if path.is_absolute() or value in (".", "..") or ".." in path.parts or path.as_posix() != value:
        _contract("workflow_relative_path must be canonical and relative")
    return value


def _result_fingerprint(result):
    if not isinstance(result, ModelCapabilityResult):
        _contract("capability result must be an inspector terminal row")
    return fingerprint_value(result.to_public_dict())


def _context_runtime_fingerprints(context):
    if not isinstance(context, WorkspaceContext):
        _contract("execution runtime binding requires WorkspaceContext")
    return (
        fingerprint_value({
            "qlib_data_dir": str(Path(context.qlib_data_dir).resolve()),
            "qlib_region": str(context.qlib_region),
        }),
        fingerprint_value({"tracking_uri": str(context.mlflow_uri)}),
    )


@dataclass(frozen=True)
class RollingExecutionRuntimeBinding:
    runtime_params_fingerprint: str
    sessions_fingerprint: str
    qlib_provider_fingerprint: str
    mlflow_backend_fingerprint: str
    fingerprint: str = field(init=False)
    _authority: InitVar[object] = None
    _builder_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority):
        if _authority is not _RUNTIME_TOKEN:
            _contract("execution runtime bindings are canonical-builder-owned")
        for name in (
            "runtime_params_fingerprint", "sessions_fingerprint",
            "qlib_provider_fingerprint", "mlflow_backend_fingerprint",
        ):
            _digest(getattr(self, name), name)
        object.__setattr__(self, "fingerprint", fingerprint_value(
            self.to_fingerprint_dict(),
        ))
        object.__setattr__(self, "_builder_authority", True)

    def to_fingerprint_dict(self):
        return {
            "runtime_params_fingerprint": self.runtime_params_fingerprint,
            "sessions_fingerprint": self.sessions_fingerprint,
            "qlib_provider_fingerprint": self.qlib_provider_fingerprint,
            "mlflow_backend_fingerprint": self.mlflow_backend_fingerprint,
        }


@dataclass(frozen=True)
class RollingExecutionTargetDescriptor:
    target_key: str
    workflow_relative_path: str
    workflow_fingerprint: str
    capability_identity: ModelCapabilityIdentity
    capability_result: ModelCapabilityResult
    _authority: InitVar[object] = None
    _mapper_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority):
        if _authority is not _TARGET_TOKEN:
            _contract("execution target descriptors are workflow-mapper-owned")
        target = RollingTargetIdentity.parse(self.target_key)
        object.__setattr__(self, "target_key", target.target_key)
        object.__setattr__(self, "workflow_relative_path", _canonical_relative_path(self.workflow_relative_path))
        _digest(self.workflow_fingerprint, "workflow_fingerprint")
        if not isinstance(self.capability_identity, ModelCapabilityIdentity):
            _contract("capability_identity must be canonical")
        if not isinstance(self.capability_result, ModelCapabilityResult):
            _contract("capability_result must be an inspector terminal row")
        if self.capability_result.identity != self.capability_identity:
            _contract("capability result identity is foreign")
        _result_fingerprint(self.capability_result)
        object.__setattr__(self, "_mapper_authority", True)

    @property
    def capability_result_fingerprint(self):
        return _result_fingerprint(self.capability_result)

    def to_public_dict(self):
        return {
            "target_key": self.target_key,
            "workflow_relative_path": self.workflow_relative_path,
            "workflow_fingerprint": self.workflow_fingerprint,
            "capability_identity": self.capability_identity.to_public_dict(),
            "capability_result_fingerprint": self.capability_result_fingerprint,
        }


@dataclass(frozen=True)
class RollingExecutionUnit:
    position: int
    target: RollingExecutionTargetDescriptor
    window: RollingWindowExecutionDescriptor

    def __post_init__(self):
        if type(self.position) is not int or self.position < 0:
            _contract("unit position must be a non-negative integer")
        if not isinstance(self.target, RollingExecutionTargetDescriptor):
            _contract("unit target must be typed")
        if not isinstance(self.window, RollingWindowExecutionDescriptor):
            _contract("unit window must be typed")
        target = RollingTargetIdentity.parse(self.target.target_key)
        if target.family != self.window.identity.family:
            _contract("unit target/window families disagree")
        if self.target.capability_identity.execution_family != target.family:
            _contract("unit capability family disagrees with its target")

    @property
    def unit_key(self):
        return (self.target.target_key, self.window.window_key)

    def to_public_dict(self):
        return {
            "position": self.position,
            "unit_key": list(self.unit_key),
            "target": self.target.to_public_dict(),
            "window": self.window.to_public_dict(),
        }


@dataclass(frozen=True)
class RollingExecutionScope:
    run_identity: RollingRunIdentity
    runtime_binding: RollingExecutionRuntimeBinding
    targets: tuple
    windows: tuple
    units: tuple
    prepared: PreparedRollingRun = field(repr=False, compare=False)
    resolved: ResolvedRollingRun = field(repr=False, compare=False)
    execution_protocol_version: str = EXECUTION_PROTOCOL_VERSION
    scope_fingerprint: str = field(init=False)
    _authority: InitVar[object] = None
    _builder_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority):
        if _authority is not _SCOPE_TOKEN:
            _contract("execution scopes are canonical-builder-owned")
        if not isinstance(self.run_identity, RollingRunIdentity):
            _contract("scope requires a canonical run identity")
        if (
            not isinstance(self.prepared, PreparedRollingRun)
            or not isinstance(self.resolved, ResolvedRollingRun)
            or self.resolved.prepared is not self.prepared
        ):
            _contract("scope lost Prepared/Resolved authority")
        if not isinstance(self.runtime_binding, RollingExecutionRuntimeBinding):
            _contract("scope requires a canonical runtime binding")
        if not isinstance(self.targets, tuple) or any(
            not isinstance(item, RollingExecutionTargetDescriptor) for item in self.targets
        ):
            _contract("scope targets must be an ordered typed tuple")
        if not isinstance(self.windows, tuple) or any(
            not isinstance(item, RollingWindowExecutionDescriptor) for item in self.windows
        ):
            _contract("scope windows must be an ordered typed tuple")
        if not isinstance(self.units, tuple) or any(not isinstance(item, RollingExecutionUnit) for item in self.units):
            _contract("scope units must be an ordered typed tuple")
        target_keys = tuple(item.target_key for item in self.targets)
        window_keys = tuple(item.window_key for item in self.windows)
        if target_keys != self.run_identity.target_keys or window_keys != self.run_identity.window_keys:
            _contract("scope descriptors disagree with run identity")
        expected = tuple((target, window) for target in target_keys for window in window_keys)
        if tuple(item.unit_key for item in self.units) != expected:
            _contract("scope units do not preserve target-major requested order")
        if tuple(item.position for item in self.units) != tuple(range(len(self.units))):
            _contract("scope unit positions are not canonical")
        if rolling_sessions_fingerprint(self.windows) != self.runtime_binding.sessions_fingerprint:
            _contract("scope sessions disagree with runtime binding")
        if self.run_identity.runtime_params_fingerprint != self.runtime_binding.fingerprint:
            _contract("run identity is not bound to the complete runtime identity")
        if self.execution_protocol_version != EXECUTION_PROTOCOL_VERSION:
            _contract("execution protocol version is unsupported")
        object.__setattr__(self, "scope_fingerprint", fingerprint_value(self.to_fingerprint_dict()))
        object.__setattr__(self, "_builder_authority", True)

    @property
    def requested_unit_keys(self):
        return tuple(item.unit_key for item in self.units)

    def to_fingerprint_dict(self):
        return {
            "run": self.run_identity.to_public_dict(),
            "runtime_binding": self.runtime_binding.to_fingerprint_dict(),
            "targets": [item.to_public_dict() for item in self.targets],
            "windows": [item.to_public_dict() for item in self.windows],
            "unit_keys": [list(item.unit_key) for item in self.units],
            "execution_protocol_version": self.execution_protocol_version,
        }

    def to_public_dict(self):
        payload = self.to_fingerprint_dict()
        payload["scope_fingerprint"] = self.scope_fingerprint
        payload["execution_authority"] = "audit_only"
        return payload


@dataclass(frozen=True)
class RollingExecutionPreflight:
    requested_unit_keys: tuple
    decisions: tuple
    status: str
    reason_code: str

    def __post_init__(self):
        if not isinstance(self.requested_unit_keys, tuple) or not isinstance(self.decisions, tuple):
            _contract("preflight members must be ordered tuples")
        if len(self.requested_unit_keys) != len(self.decisions):
            _contract("preflight must decide every requested unit")
        if tuple(item[0] for item in self.decisions) != self.requested_unit_keys:
            _contract("preflight decisions do not preserve requested identity")
        if any(type(item[1]) is not bool for item in self.decisions):
            _contract("preflight decisions require strict booleans")
        expected = "allowed" if self.decisions and all(item[1] for item in self.decisions) else "blocked"
        if self.status != expected:
            _contract("preflight status disagrees with decisions")
        reason = "rolling_execution_preflight_allowed" if expected == "allowed" else "rolling_execution_preflight_blocked"
        if self.reason_code != reason:
            _contract("preflight reason disagrees with status")

    @property
    def preflight_allowed(self):
        return self.status == "allowed"


@dataclass(frozen=True)
class RollingUnitRunnerObservation:
    unit_key: tuple
    attempt_id: str
    candidate_status: str
    experiment_name: Optional[str] = None
    experiment_id: Optional[str] = None
    recorder_id: Optional[str] = None
    failure_code: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.unit_key, tuple) or len(self.unit_key) != 2:
            _contract("runner observation unit_key must be a pair")
        target = RollingTargetIdentity.parse(self.unit_key[0])
        family, _start, _end, _digest_value = parse_rolling_window_key(self.unit_key[1])
        if target.family != family:
            _contract("runner observation target/window families disagree")
        _public_text(self.attempt_id, "attempt_id")
        if self.candidate_status not in ("candidate_success", "failed"):
            _contract("runner candidate_status is unsupported")
        identities = (self.experiment_name, self.experiment_id, self.recorder_id)
        if self.candidate_status == "candidate_success":
            if any(item is None for item in identities) or self.failure_code is not None:
                _contract("candidate success requires exact recorder facts and no failure")
            for name, value in zip(("experiment_name", "experiment_id", "recorder_id"), identities):
                _public_text(value, name)
        else:
            if any(item is not None for item in identities) or self.failure_code is None:
                _contract("runner failure cannot grant recorder capability")
            _public_text(self.failure_code, "failure_code")


@dataclass(frozen=True)
class RollingExecutionUnitResult:
    unit_key: tuple
    position: int
    status: str
    did_execute: bool
    attempt_id: Optional[str] = None
    record_id: Optional[str] = None
    evidence_id: Optional[str] = None
    reason_code: str = "rolling_execution_failed"
    _authority: InitVar[object] = None
    _kernel_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority):
        if not isinstance(self.unit_key, tuple) or len(self.unit_key) != 2:
            _contract("unit result key must be a pair")
        target = RollingTargetIdentity.parse(self.unit_key[0])
        family, _start, _end, _digest_value = parse_rolling_window_key(self.unit_key[1])
        if target.family != family:
            _contract("unit result target/window families disagree")
        if type(self.position) is not int or self.position < 0:
            _contract("unit result position is invalid")
        if self.status not in UNIT_STATUSES or type(self.did_execute) is not bool:
            _contract("unit result status/did_execute is invalid")
        success = self.status in ("executed_success", "reused_success")
        if success:
            if _authority is not _RESULT_TOKEN:
                _contract("successful unit results are kernel-owned")
            if any(item is None for item in (self.attempt_id, self.record_id, self.evidence_id)):
                _contract("successful unit result lacks exact source identity")
            _digest(self.evidence_id, "evidence_id")
        elif self.record_id is not None or self.evidence_id is not None:
            _contract("non-success unit result cannot grant evidence capability")
        if self.status == "executed_success" and not self.did_execute:
            _contract("executed_success requires did_execute")
        if self.status == "reused_success" and self.did_execute:
            _contract("reused_success forbids did_execute")
        if self.status == "blocked" and self.did_execute:
            _contract("blocked unit cannot claim runner mutation")
        _public_text(self.reason_code, "reason_code")
        object.__setattr__(self, "_kernel_authority", _authority is _RESULT_TOKEN)


@dataclass(frozen=True)
class RollingExecutionBatchResult:
    requested_unit_keys: tuple
    unit_results: tuple
    status: str
    reason_code: str
    _authority: InitVar[object] = None
    _kernel_authority: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _authority):
        if not isinstance(self.requested_unit_keys, tuple) or not isinstance(self.unit_results, tuple):
            _contract("batch identities/results must be ordered tuples")
        if any(not isinstance(item, RollingExecutionUnitResult) for item in self.unit_results):
            _contract("batch results contain a foreign member")
        if tuple(item.unit_key for item in self.unit_results) != self.requested_unit_keys:
            _contract("batch results do not preserve requested identity/order/cardinality")
        if len(self.requested_unit_keys) != len(set(self.requested_unit_keys)):
            _contract("batch requested identity contains duplicates")
        if tuple(item.position for item in self.unit_results) != tuple(range(len(self.unit_results))):
            _contract("batch result positions are not canonical")
        expected = (
            "interrupted" if any(item.status == "interrupted" for item in self.unit_results)
            else "failed" if any(item.status == "failed" for item in self.unit_results)
            else "blocked" if not self.unit_results or any(item.status == "blocked" for item in self.unit_results)
            else "success"
        )
        if self.status != expected or self.reason_code != "rolling_execution_batch_%s" % expected:
            _contract("batch status/reason disagrees with terminal members")
        if self.status == "success" and _authority is not _BATCH_TOKEN:
            _contract("successful batch results are kernel-owned")
        object.__setattr__(self, "_kernel_authority", _authority is _BATCH_TOKEN)

    @property
    def n_requested(self):
        return len(self.unit_results)

    @property
    def n_executed_success(self):
        return sum(item.status == "executed_success" for item in self.unit_results)

    @property
    def n_reused_success(self):
        return sum(item.status == "reused_success" for item in self.unit_results)

    @property
    def n_failed(self):
        return sum(item.status == "failed" for item in self.unit_results)

    @property
    def n_blocked(self):
        return sum(item.status == "blocked" for item in self.unit_results)

    @property
    def n_interrupted(self):
        return sum(item.status == "interrupted" for item in self.unit_results)

    @property
    def n_runner_calls(self):
        return sum(item.did_execute for item in self.unit_results)


def _ordered_window_selection(resolved, selected_window_keys):
    if not isinstance(selected_window_keys, tuple):
        _contract("selected_window_keys must be an ordered tuple")
    resolved_keys = tuple(item.window_key for item in resolved.windows)
    if len(selected_window_keys) != len(set(selected_window_keys)):
        _contract("selected windows contain duplicates")
    positions = []
    for key in selected_window_keys:
        try:
            positions.append(resolved_keys.index(key))
        except ValueError:
            _contract("selected window is outside the ResolvedRollingRun")
    if positions != sorted(positions):
        _contract("selected windows do not preserve resolved order")
    return selected_window_keys


def build_rolling_execution_scope(
    prepared, resolved, selected_window_keys, targets, windows,
):
    """Build authority from Prepared + Resolved + one ordered window selection."""

    if not isinstance(prepared, PreparedRollingRun):
        _contract("scope builder requires PreparedRollingRun")
    if not isinstance(resolved, ResolvedRollingRun) or resolved.prepared is not prepared:
        _contract("scope builder requires the matching ResolvedRollingRun")
    if prepared.ctx != resolved.prepared.ctx:
        _contract("prepared and resolved workspace contexts disagree")
    base_identity = resolved.identity
    resolved_window_keys = tuple(item.window_key for item in resolved.windows)
    prepared_target_keys = tuple(item.target_key for item in prepared.targets)
    if (
        not isinstance(prepared.plan, CommandPlan)
        or prepared.plan_fingerprint
        != fingerprint_command_plan(prepared.plan, length=64)
        or
        base_identity.workspace_fingerprint != workspace_fingerprint(prepared.ctx.root)
        or base_identity.plan_fingerprint != prepared.plan_fingerprint
        or base_identity.config_fingerprint
        != fingerprint_value(prepared.effective_config)
        or base_identity.action != prepared.options.action
        or base_identity.target_keys != prepared_target_keys
        or base_identity.window_keys != resolved_window_keys
        or base_identity.anchor_date != resolved.actual_anchor
    ):
        _contract("Prepared/Resolved logical identity failed canonical revalidation")
    if not isinstance(targets, tuple) or not isinstance(windows, tuple):
        _contract("targets and windows must be ordered tuples")
    if tuple(item.target_key for item in targets) != prepared_target_keys:
        _contract("mapped targets do not preserve the Prepared target tuple")
    for source, mapped in zip(prepared.targets, targets):
        if (
            source.workflow_path != mapped.workflow_relative_path
            or source.workflow_fingerprint != mapped.workflow_fingerprint
        ):
            _contract("mapped target source disagrees with PreparedRollingRun")
    selected_window_keys = _ordered_window_selection(resolved, selected_window_keys)
    if tuple(item.window_key for item in windows) != selected_window_keys:
        _contract("observed windows do not exactly match the ordered selection")
    resolved_by_key = {item.window_key: item for item in resolved.windows}
    if tuple(item.window for item in windows) != tuple(
        resolved_by_key[key] for key in selected_window_keys
    ):
        _contract("observed windows are foreign to the ResolvedRollingRun")
    run_identity, runtime_binding = bind_rolling_execution_run_identity(
        resolved, selected_window_keys, windows,
    )
    units = tuple(
        RollingExecutionUnit(position, target, window)
        for position, (target, window) in enumerate(
            (target, window) for target in targets for window in windows
        )
    )
    return RollingExecutionScope(
        run_identity, runtime_binding, targets, windows, units, prepared, resolved,
        _authority=_SCOPE_TOKEN,
    )


def bind_rolling_execution_run_identity(resolved, selected_window_keys, windows):
    """Bind resolved params, selected sessions and provider/backend identities."""

    if not isinstance(resolved, ResolvedRollingRun):
        _contract("execution identity binding requires ResolvedRollingRun")
    base_identity = resolved.identity
    selected_window_keys = _ordered_window_selection(resolved, selected_window_keys)
    if not isinstance(windows, tuple) or any(
        not isinstance(item, RollingWindowExecutionDescriptor) for item in windows
    ):
        _contract("execution identity binding requires observed windows")
    window_keys = tuple(item.window_key for item in windows)
    if window_keys != selected_window_keys:
        _contract("execution windows disagree with selected resolved windows")
    runtime_params_fingerprint = fingerprint_value(dict(resolved.params))
    if runtime_params_fingerprint != base_identity.runtime_params_fingerprint:
        _contract("ResolvedRollingRun runtime params changed")
    qlib_provider_fingerprint, mlflow_backend_fingerprint = (
        _context_runtime_fingerprints(resolved.prepared.ctx)
    )
    runtime_binding = RollingExecutionRuntimeBinding(
        runtime_params_fingerprint,
        rolling_sessions_fingerprint(windows),
        qlib_provider_fingerprint,
        mlflow_backend_fingerprint,
        _authority=_RUNTIME_TOKEN,
    )
    identity = RollingRunIdentity(
        workspace_fingerprint=base_identity.workspace_fingerprint,
        family=base_identity.family,
        action=base_identity.action,
        plan_fingerprint=base_identity.plan_fingerprint,
        config_fingerprint=base_identity.config_fingerprint,
        anchor_date=base_identity.anchor_date,
        target_keys=base_identity.target_keys,
        window_keys=window_keys,
        runtime_params_fingerprint=runtime_binding.fingerprint,
        attempt_id=base_identity.attempt_id,
    )
    return identity, runtime_binding


def preflight_rolling_execution(scope):
    if not isinstance(scope, RollingExecutionScope):
        _contract("preflight requires a typed scope")
    decisions = tuple(
        (unit.unit_key, unit.target.capability_result.preflight_allowed)
        for unit in scope.units
    )
    status = "allowed" if decisions and all(item[1] for item in decisions) else "blocked"
    return RollingExecutionPreflight(
        scope.requested_unit_keys, decisions, status,
        "rolling_execution_preflight_%s" % status,
    )


def _load_yaml(path):
    try:
        import yaml
        with path.open("rb") as handle:
            data = handle.read()
        payload = yaml.safe_load(data)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        _contract("workflow cannot be read: %s" % exc.__class__.__name__)
    if not isinstance(payload, Mapping):
        _contract("workflow root must be a mapping")
    return data, payload


def map_workflow_capability(
    context, target_key, workflow_relative_path, matrix,
    action="train", execution_family="rolling",
    dataset_protocol="point_in_time",
    processor_profile="standard_infer_no_label_drop",
    artifact_protocol="qlib_recorder_model_v1",
    dependency_profile="python_qlib",
):
    """Map exact YAML module/class facts to one authoritative row."""

    if not isinstance(context, WorkspaceContext):
        _contract("workflow mapping requires WorkspaceContext")
    if not isinstance(matrix, ModelCapabilityMatrix):
        _contract("workflow mapping requires an authoritative matrix")
    relative = _canonical_relative_path(workflow_relative_path)
    root = Path(context.root).resolve(strict=True)
    path = (root / relative).absolute()
    try:
        path.parent.resolve(strict=True).relative_to(root)
        physical = path.resolve(strict=True)
        physical.relative_to(root)
    except (OSError, ValueError):
        _contract("workflow path is missing or physically outside workspace")
    data, workflow = _load_yaml(physical)
    try:
        model = workflow["task"]["model"]
        dataset = workflow["task"]["dataset"]
        model_module, model_class = model["module_path"], model["class"]
        dataset_module, dataset_class = dataset["module_path"], dataset["class"]
    except (KeyError, TypeError):
        _contract("workflow lacks exact model/dataset module and class")
    identity = ModelCapabilityIdentity(
        model_module=model_module, model_class=model_class,
        wrapper_kind="external_passthrough",
        dataset_module=dataset_module, dataset_class=dataset_class,
        dataset_protocol=dataset_protocol, action=action,
        execution_family=execution_family,
        processor_profile=processor_profile,
        artifact_protocol=artifact_protocol,
        dependency_profile=dependency_profile,
    )
    try:
        result = matrix.query(identity)
    except Exception as exc:
        _contract("workflow has no unique exact capability row: %s" % exc.__class__.__name__)
    target = RollingTargetIdentity.parse(target_key)
    if target.family != execution_family:
        _contract("workflow target and capability families disagree")
    return RollingExecutionTargetDescriptor(
        target.target_key, relative, hashlib.sha256(data).hexdigest(), identity, result,
        _authority=_TARGET_TOKEN,
    )


class RollingUnitRunner(Protocol):
    def execute(self, scope, unit, attempt_id): ...


def execute_rolling_units(scope, repository, backend, runner, attempt_id):
    from quantpits.rolling.execution_backend import RollingExecutionKernel
    return RollingExecutionKernel(repository, backend, runner).execute(scope, attempt_id)


def resume_rolling_units(scope, repository, backend, runner, attempt_id):
    from quantpits.rolling.execution_backend import RollingExecutionKernel
    return RollingExecutionKernel(repository, backend, runner).resume(scope, attempt_id)
