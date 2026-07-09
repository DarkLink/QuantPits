"""Run id helpers for runtime plans and manifests."""

from __future__ import annotations

import re
import secrets
from datetime import datetime


_RUN_ID_PART_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize_part(value: str) -> str:
    sanitized = _RUN_ID_PART_RE.sub("_", value).strip("_")
    return sanitized or "run"


def generate_run_id(
    command: str,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> str:
    """Generate a stable-format run id for a command execution."""

    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_command = _sanitize_part(command)
    safe_suffix = _sanitize_part(suffix) if suffix is not None else secrets.token_hex(3)
    return f"{timestamp}_{safe_command}_{safe_suffix}"
