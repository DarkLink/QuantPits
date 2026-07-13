"""Typed failures for training command planning and execution."""

from __future__ import annotations


class TrainingCommandError(RuntimeError):
    exit_code = 1
    code = "training_command_error"


class TrainingPlanError(TrainingCommandError):
    code = "training_plan_error"


class TrainingExecutionError(TrainingCommandError):
    exit_code = 2
    code = "training_execution_error"


class TrainingPublicationError(TrainingExecutionError):
    code = "training_publication_error"
