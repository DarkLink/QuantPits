"""Typed failures for Rolling command planning and execution."""


class RollingCommandError(RuntimeError):
    """Base class for stable Rolling CLI failures."""

    exit_code = 2
    code = "rolling_command_error"

    def to_public_dict(self):
        return {"code": self.code, "message": str(self)}


class RollingWorkspaceRequiredError(RollingCommandError):
    code = "rolling_workspace_required"


class RollingActionConflictError(RollingCommandError):
    code = "rolling_action_conflict"


class RollingConfigMissingError(RollingCommandError):
    code = "rolling_config_missing"


class RollingConfigInvalidError(RollingCommandError):
    code = "rolling_config_invalid"


class RollingTargetSelectionEmptyError(RollingCommandError):
    code = "rolling_target_selection_empty"


class RollingTargetUnknownError(RollingCommandError):
    code = "rolling_target_unknown"


class RollingWorkflowMissingError(RollingCommandError):
    code = "rolling_workflow_missing"


class RollingWorkflowOutsideWorkspaceError(RollingCommandError):
    code = "rolling_workflow_outside_workspace"


class RollingStateCorruptError(RollingCommandError):
    code = "rolling_state_corrupt"


class RollingStateUnsupportedError(RollingCommandError):
    code = "rolling_state_unsupported"


class RollingResumeStateMissingError(RollingCommandError):
    code = "rolling_resume_state_missing"


class RollingStatePreconditionError(RollingCommandError):
    code = "rolling_state_precondition_failed"


class RollingWorkspaceActivationError(RollingCommandError):
    code = "rolling_workspace_activation_failed"


class RollingInputChangedError(RollingCommandError):
    code = "rolling_input_changed"


class RollingWindowResolutionError(RollingCommandError):
    code = "rolling_window_resolution_failed"


class RollingExecutionError(RollingCommandError):
    code = "rolling_execution_failed"
