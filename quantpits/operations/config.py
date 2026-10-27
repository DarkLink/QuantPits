"""Small explicit project configuration; account membership belongs to the group."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from quantpits.research.account_comparison import load_account_group
from quantpits.research.paper_accounting import date
from quantpits.research.paper_market import strict_json
from quantpits.utils.workspace import fingerprint_value


class CycleError(ValueError):
    pass


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise CycleError("cycle/member/signal id must use safe filename characters (at most 80)")
    return value


def path_at(base, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def contained(base, value):
    base = Path(base).resolve()
    result = path_at(base, value)
    try:
        result.relative_to(base)
    except ValueError as exc:
        raise CycleError("path escapes its declared directory: %s" % value) from exc
    return result


def read_json(path):
    return strict_json(Path(path).read_bytes())


@dataclass(frozen=True)
class Project:
    path: Path
    group: object
    signals: dict
    accounts: dict
    rules: object
    prepare_rules: object
    identity: str

    @property
    def members(self):
        return {m.id: m for m in self.group.members}

    def cycle_dir(self, cycle):
        return contained(self.path.parent, "cycles/" + identifier(cycle))

    def signal_dir(self, cycle, signal):
        ctx = self.members[self.signals[signal]["producer"]].ctx
        # The project name prevents equal cycle IDs in different projects colliding.
        project_key = self.path.stem + "-" + fingerprint_value(str(self.path))[:8]
        return contained(ctx.root, "output/cycles/%s/%s/%s" % (project_key, identifier(cycle), identifier(signal)))


def load_project(path):
    path = Path(path).expanduser().resolve(strict=True)
    # Duplicate keys must fail just as they do in group files.
    class Loader(yaml.SafeLoader):
        pass
    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise CycleError("project keys must be unique text")
            result[key] = loader.construct_object(value_node)
        return result
    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
    if not isinstance(value, dict) or set(value) - {"version", "group", "signals", "accounts", "rules", "prepare_rules"}:
        raise CycleError("invalid project fields")
    if type(value.get("version")) is not int or value["version"] != 1:
        raise CycleError("project version must be 1")
    identifier(path.stem)
    group = load_account_group(path_at(path.parent, value["group"]))
    members = {identifier(m.id): m for m in group.members}
    roots = [m.ctx.root for m in group.members]
    if len(set(roots)) != len(roots):
        raise CycleError("account workspaces must be distinct")
    signals = value.get("signals")
    accounts = value.get("accounts")
    if not isinstance(signals, dict) or not signals or not isinstance(accounts, dict) or set(accounts) != set(members):
        raise CycleError("signals required; accounts must bind every group member exactly once")
    for key, spec in signals.items():
        identifier(key)
        if not isinstance(spec, dict) or set(spec) - {"producer", "models", "fit_records", "combo", "norm_method"}:
            raise CycleError("invalid signal fields")
        if spec.get("producer") not in members:
            raise CycleError("signal producer must name a group member")
        models = spec.get("models")
        if not isinstance(models, list) or not models or any(not isinstance(m, str) or not m or "," in m or "@" in m for m in models) or len(set(models)) != len(models):
            raise CycleError("signal requires an explicit distinct list of static model names")
        identifier(spec["combo"])
        if spec.get("norm_method", "rank") not in ("rank", "zscore"):
            raise CycleError("invalid normalization")
        contained(members[spec["producer"]].ctx.root, spec["fit_records"])
    producers = [s["producer"] for s in signals.values()]
    if len(set(producers)) != len(producers):
        raise CycleError("one signal group per producer workspace; combine shared consumers in its binding")
    if value.get("prepare_rules") and len({m.ctx.qlib_data_dir for m in members.values()}) != 1:
        raise CycleError("common prepared rules require a shared provider")
    for key, binding in accounts.items():
        if not isinstance(binding, dict) or set(binding) - {"signal", "rules", "pending_rules"} or binding.get("signal") not in signals:
            raise CycleError("invalid account binding: %s" % key)
        if not isinstance(binding.get("pending_rules", {}), dict):
            raise CycleError("pending_rules must map native cycle IDs to rule files")
        for rule in ([binding["rules"]] if binding.get("rules") else []) + list(binding.get("pending_rules", {}).values()):
            path_at(path.parent, rule)
    normalized = dict(value, group_sha256=group.config_sha256, contexts={key: {
        "root": str(m.ctx.root), "mlflow_uri": m.ctx.mlflow_uri,
        "provider": str(m.ctx.qlib_data_dir), "region": m.ctx.qlib_region,
    } for key, m in members.items()})
    return Project(path, group, signals, accounts, value.get("rules"), value.get("prepare_rules"), fingerprint_value(normalized))


def selected_fit_records(project, request, signal):
    """Resolve this request's explicit source, otherwise the project default."""
    spec = project.signals[signal]
    value = request["inputs"].get("fit_records", {}).get(signal, spec["fit_records"])
    return contained(project.members[spec["producer"]].ctx.root, value)


def cycle_request(project, *, cycle, anchor, value_through, trade_date, since, inputs=None,
                  _frozen_sources=False):
    identifier(cycle)
    for selected in (anchor, value_through, trade_date, since):
        date(selected)
    if not since < value_through <= anchor < trade_date:
        raise CycleError("dates require since < value-through <= anchor < trade-date")
    inputs = dict(inputs or {})
    if set(inputs) - {"actual", "paper_cycles", "hold", "skip", "fit_records"}:
        raise CycleError("unknown cycle input fields")
    if "fit_records" in inputs:
        overrides = inputs["fit_records"]
        if not isinstance(overrides, dict):
            raise CycleError("fit_records must map signal IDs to source paths")
        for key, value in overrides.items():
            if key not in project.signals:
                raise CycleError("unknown fit_records signal: %s" % key)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise CycleError("fit_records source must be a nonempty valid path")
            if _frozen_sources:
                ctx = project.members[project.signals[key]["producer"]].ctx
                frozen = contained(ctx.root, project.signal_dir(cycle, key) / "fit_records.json")
                if frozen.is_file():
                    continue  # saved requests use the fixed copy, not a mutable external path
            selected_fit_records(project, {"inputs": inputs}, key)
    hold = inputs.get("hold", [])
    if not isinstance(hold, list) or any(not isinstance(key, str) for key in hold) or len(set(hold)) != len(hold):
        raise CycleError("hold must be a distinct list of member IDs")
    for key in hold:
        if key not in project.members:
            raise CycleError("unknown HOLD member")
    for key, native_cycle in inputs.get("paper_cycles", {}).items():
        if key not in project.members or project.members[key].kind != "paper":
            raise CycleError("paper cycle mapping must name a paper member")
        identifier(native_cycle)
    for key, spec in inputs.get("actual", {}).items():
        if key not in project.members or project.members[key].kind != "actual" or not isinstance(spec, dict) or set(spec) - {"settlement_bundle", "allow_missing_settlement"}:
            raise CycleError("invalid actual cycle inputs")
        if "allow_missing_settlement" in spec and type(spec["allow_missing_settlement"]) is not bool:
            raise CycleError("allow_missing_settlement must be boolean")
        if "settlement_bundle" in spec:
            bundle = Path(spec["settlement_bundle"])
            if bundle.is_absolute() or ".." in bundle.parts:
                raise CycleError("settlement_bundle must be relative to the actual workspace")
    return {"cycle": cycle, "anchor": anchor, "value_through": value_through,
            "trade_date": trade_date, "since": since, "inputs": inputs}
