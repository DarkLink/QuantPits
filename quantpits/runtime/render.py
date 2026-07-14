"""Rendering helpers for runtime plans and manifests."""

from __future__ import annotations

from typing import Any, Iterable, List

from quantpits.runtime.command import CommandPlan


def command_plan_to_public_dict(plan: CommandPlan) -> dict[str, Any]:
    return plan.to_public_dict()


def run_manifest_to_public_dict(manifest: Any) -> dict[str, Any]:
    return manifest.to_public_dict()


def _render_refs(title: str, refs: Iterable[Any]) -> List[str]:
    items = list(refs)
    if not items:
        return []
    lines = [f"{title}:"]
    for item in items:
        public = item.to_public_dict()
        suffixes = [f"[{public.get('kind', public.get('action', ''))}]"]
        if public.get("fingerprint"):
            suffixes.append(f"sha256={public['fingerprint']}")
        if public.get("overwrite"):
            suffixes.append("overwrite")
        if public.get("required") is False:
            suffixes.append("optional")
        description = public.get("description", "")
        rendered = f"  - {public['path']} {' '.join(suffixes)}"
        if description:
            rendered += f": {description}"
        lines.append(rendered)
    return lines


def _render_steps(steps: Iterable[Any]) -> List[str]:
    items = list(steps)
    if not items:
        return []
    lines = ["Would run:"]
    for step in items:
        markers = []
        if step.expensive:
            markers.append("[expensive]")
        if step.can_skip:
            markers.append("[can skip]")
        marker_text = f" {' '.join(markers)}" if markers else ""
        rendered = f"  - {step.name}: {step.description}{marker_text}"
        if step.skip_reason:
            rendered += f" ({step.skip_reason})"
        lines.append(rendered)
    return lines


def _render_plan_identity(plan: CommandPlan) -> List[str]:
    """Render stable execution identity carried by typed command metadata."""

    metadata = plan.metadata
    target_keys = metadata.get("target_keys")
    publication_policy = metadata.get("publication_policy")
    source_identities = metadata.get("source_identities")
    if not target_keys and not publication_policy and not source_identities:
        return []

    lines = ["Plan identity:"]
    if target_keys:
        lines.append("  - Target keys: %s" % ", ".join(str(item) for item in target_keys))
    if publication_policy:
        lines.append("  - Publication policy: %s" % publication_policy)
    for source in source_identities or ():
        lines.append(
            "  - Source for %(target_key)s: experiment=%(experiment_name)s "
            "recorder=%(recorder_id)s operation=%(operation)s status=%(status)s" % source
        )
    return lines


def render_command_plan(plan: CommandPlan) -> str:
    """Render a command plan for human dry-run output."""

    lines = [
        "--- Execution Plan (dry run) ---",
        f"Command: {plan.command}",
        f"Workspace: {plan.workspace}",
        f"Run ID: {plan.run_id}",
    ]
    if plan.mode:
        lines.append(f"Mode: {plan.mode}")
    if plan.args:
        lines.append(f"Args: {' '.join(plan.args)}")

    sections: List[List[str]] = [
        _render_plan_identity(plan),
        _render_refs("Inputs", plan.inputs),
        _render_steps(plan.steps),
        _render_refs("Would write", plan.outputs),
        _render_refs("State", plan.states),
    ]

    if plan.config_fingerprints:
        sections.append(
            ["Config fingerprints:"]
            + [
                f"  - {name}: {fingerprint}"
                for name, fingerprint in sorted(plan.config_fingerprints.items())
            ]
        )

    if plan.warnings:
        sections.append(["Warnings:"] + [f"  - {warning}" for warning in plan.warnings])

    for section in sections:
        if section:
            lines.append("")
            lines.extend(section)

    return "\n".join(lines)
