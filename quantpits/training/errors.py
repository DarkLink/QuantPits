"""Typed failures for training command planning and execution."""

from __future__ import annotations


class TrainingCommandError(RuntimeError):
    exit_code = 1
    code = "training_command_error"


class TrainingPlanError(TrainingCommandError):
    code = "training_plan_error"


class TrainingDatePolicyError(TrainingPlanError):
    code = "training_date_policy_error"


class TrainingSourceRecordError(TrainingPlanError):
    code = "training_source_record_error"


class TrainingExecutionError(TrainingCommandError):
    exit_code = 2
    code = "training_execution_error"


class TrainingRunnerContractError(TrainingExecutionError):
    code = "training_runner_contract_error"


class TrainingRecorderIntegrityError(TrainingExecutionError):
    code = "training_recorder_integrity_error"


class TrainingStateConflictError(TrainingExecutionError):
    code = "training_state_conflict_error"


class TrainingPublicationError(TrainingExecutionError):
    code = "training_publication_error"
