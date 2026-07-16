"""Hand-authored semantic expectations; no production helper is imported."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TerminalMemberExpectation:
    identity: str
    status: str
    did_execute: bool


@dataclass(frozen=True)
class SemanticScenarioExpectation:
    requested_identities: Tuple[str, ...]
    terminal_members: Tuple[TerminalMemberExpectation, ...]
    authoritative_inputs: Tuple[str, ...]
    allowed_write_paths: Tuple[str, ...]
    forbidden_paths: Tuple[str, ...] = ()

    @property
    def aggregate_status(self):
        return "success" if all(item.status == "success" for item in self.terminal_members) else "failed"
