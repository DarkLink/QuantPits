"""Read-only workspace configuration validation CLI."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, Optional

from quantpits.config_contracts.core import ValidationMessage, WorkspaceValidationResult
from quantpits.config_contracts.workspace import validate_workspace
from quantpits.utils.workspace import WorkspaceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate QuantPits workspace configuration files.")
    parser.add_argument("--workspace", help="Workspace root path. Defaults to the active QLIB_WORKSPACE_DIR context.")
    parser.add_argument("--read-only", action="store_true", help="Document that validation must not write workspace files.")
    parser.add_argument("--strict", action="store_true", help="Treat missing optional configs as errors.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    parser.add_argument(
        "--include-optional",
        dest="include_optional",
        action="store_true",
        default=True,
        help="Validate optional workspace config files as warnings when missing (default).",
    )
    parser.add_argument(
        "--no-include-optional",
        dest="include_optional",
        action="store_false",
        help="Skip optional workspace config files.",
    )
    return parser


def _resolve_context(workspace: Optional[str]) -> WorkspaceContext:
    if workspace:
        return WorkspaceContext.from_root(workspace)
    from quantpits.utils import env

    return env.get_workspace_context()


def _render_messages(title: str, messages: Iterable[ValidationMessage]) -> list[str]:
    lines = []
    materialized = list(messages)
    if not materialized:
        return lines
    lines.append("")
    lines.append(f"{title}:")
    for message in materialized:
        hint = f" Hint: {message.hint}" if message.hint else ""
        lines.append(f"  [{message.code}] {message.path}: {message.message}{hint}")
    return lines


def render_text(result: WorkspaceValidationResult, *, workspace_label: Optional[str] = None) -> str:
    lines = [
        f"Workspace: {workspace_label or result.workspace}",
        f"Status: {'OK' if result.ok else 'FAILED'}",
        "",
        "Checked:",
    ]
    for artifact in result.artifacts:
        public = artifact.to_public_dict(workspace=result.workspace)
        status = "present" if artifact.exists else "missing"
        fingerprint = public.get("fingerprint") or "-"
        lines.append(f"  - {public['path']} ({status}) sha256={fingerprint}")
    lines.extend(_render_messages("Errors", result.errors))
    lines.extend(_render_messages("Warnings", result.warnings))
    lines.extend(_render_messages("Info", result.infos))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ctx = _resolve_context(args.workspace)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"Workspace resolution failed: {exc}", file=sys.stderr)
        return 2

    result = validate_workspace(ctx, include_optional=args.include_optional, strict=args.strict)
    workspace_label = args.workspace or ctx.root.as_posix()
    if args.json:
        print(json.dumps(result.to_public_dict(workspace_label=workspace_label), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(result, workspace_label=workspace_label))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
