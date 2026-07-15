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

__all__ = [
    "RollingCommandError",
    "LegacyRollingStateInspection",
    "PreparedRollingRun",
    "RollingRunOptions",
    "RollingTarget",
    "RollingWindowDescriptor",
    "ResolvedRollingRun",
    "RollingExecutionOutcome",
    "LegacyRollingExecutionAdapter",
    "RollingWorkspaceRequiredError",
    "options_from_namespace",
    "prepare_rolling_run",
    "prepared_plan_json",
    "render_prepared_plan",
    "resolve_workspace_context",
    "resolve_rolling_run",
    "recheck_prepared_inputs",
]
