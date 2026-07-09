"""Bridge workspace validation artifacts into runtime plan refs."""

from __future__ import annotations

from typing import Tuple

from quantpits.config_contracts.core import WorkspaceValidationResult
from quantpits.runtime.command import InputRef


def input_refs_from_validation(result: WorkspaceValidationResult) -> Tuple[InputRef, ...]:
    """Convert config validation artifacts into command plan input refs."""

    refs = []
    for artifact in result.artifacts:
        try:
            display_path = artifact.path.relative_to(result.workspace).as_posix()
        except ValueError:
            display_path = artifact.path.as_posix()
        refs.append(
            InputRef(
                path=display_path,
                kind="config",
                fingerprint=artifact.fingerprint,
                required=artifact.exists,
                description=artifact.name,
            )
        )
    return tuple(refs)
