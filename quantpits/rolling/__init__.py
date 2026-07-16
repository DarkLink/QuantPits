"""Rolling command-domain boundaries.

The execution implementation remains in :mod:`quantpits.scripts.rolling`
during the Phase 28 compatibility migration.
"""

from quantpits.rolling.command import (
    LegacyRollingStateInspection,
    PreparedRollingRun,
    RollingRunOptions,
    RollingTarget,
    options_from_namespace,
    prepare_rolling_run,
    prepared_plan_json,
    render_prepared_plan,
    resolve_workspace_context,
)
from quantpits.rolling.errors import (
    RollingCommandError,
    RollingStatePathError,
    RollingStatePersistenceError,
    RollingStateRepositoryError,
    RollingStateTransitionError,
    RollingWorkspaceRequiredError,
)
from quantpits.rolling.identity import (
    RollingFoldIdentity,
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingWindowIdentity,
    family_for_training_method,
    parse_rolling_window_key,
    training_method_for_family,
    workspace_fingerprint,
)
from quantpits.rolling.legacy import (
    LegacyRollingExecutionAdapter,
    RollingExecutionOutcome,
    recheck_prepared_inputs,
)
from quantpits.rolling.windows import (
    ResolvedRollingRun,
    RollingWindowDescriptor,
    resolve_rolling_run,
)
from quantpits.rolling.state import (
    LegacyRollingStateSnapshot,
    RollingStateExpectation,
    RollingStateInspection,
    RollingStateMigrationProposal,
    RollingStateUnitClaim,
    RollingStateV2Snapshot,
    build_legacy_migration_proposal,
    inspect_rolling_state,
    inspect_rolling_state_bytes,
    parse_rolling_state_v2_bytes,
    serialize_rolling_state_v2,
)
from quantpits.rolling.repository import (
    RollingStateBaseline,
    RollingStateMutationReceipt,
    RollingStateRepository,
    RollingStateRepositoryView,
)

__all__ = [
    "RollingCommandError",
    "LegacyRollingStateInspection",
    "PreparedRollingRun",
    "RollingRunOptions",
    "RollingTarget",
    "RollingTargetIdentity",
    "RollingFoldIdentity",
    "RollingWindowIdentity",
    "RollingRunIdentity",
    "RollingWindowDescriptor",
    "ResolvedRollingRun",
    "RollingExecutionOutcome",
    "LegacyRollingExecutionAdapter",
    "RollingWorkspaceRequiredError",
    "RollingStateRepositoryError",
    "RollingStatePathError",
    "RollingStateTransitionError",
    "RollingStatePersistenceError",
    "RollingStateExpectation",
    "RollingStateInspection",
    "RollingStateMigrationProposal",
    "RollingStateUnitClaim",
    "LegacyRollingStateSnapshot",
    "RollingStateV2Snapshot",
    "RollingStateBaseline",
    "RollingStateRepositoryView",
    "RollingStateMutationReceipt",
    "RollingStateRepository",
    "options_from_namespace",
    "prepare_rolling_run",
    "prepared_plan_json",
    "render_prepared_plan",
    "resolve_workspace_context",
    "resolve_rolling_run",
    "recheck_prepared_inputs",
    "inspect_rolling_state",
    "inspect_rolling_state_bytes",
    "parse_rolling_state_v2_bytes",
    "serialize_rolling_state_v2",
    "build_legacy_migration_proposal",
    "family_for_training_method",
    "parse_rolling_window_key",
    "training_method_for_family",
    "workspace_fingerprint",
]
