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
from quantpits.rolling.errors import RollingCommandError, RollingWorkspaceRequiredError
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
    RollingStateV2Snapshot,
    inspect_rolling_state,
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
    "RollingStateExpectation",
    "RollingStateInspection",
    "LegacyRollingStateSnapshot",
    "RollingStateV2Snapshot",
    "options_from_namespace",
    "prepare_rolling_run",
    "prepared_plan_json",
    "render_prepared_plan",
    "resolve_workspace_context",
    "resolve_rolling_run",
    "recheck_prepared_inputs",
    "inspect_rolling_state",
    "family_for_training_method",
    "parse_rolling_window_key",
    "training_method_for_family",
    "workspace_fingerprint",
]
