"""Sequential project cycle with independent branch progress and native recovery."""
from __future__ import annotations

from datetime import datetime, timezone

from quantpits.operations.config import CycleError, contained, cycle_request, path_at, read_json
from quantpits.operations.native import NativeSteps, next_day, ref, save_json, validate_write_context, verify_refs
from quantpits.operations.process import ProcessRunner
from quantpits.research.paper_market import PaperMarket
from quantpits.research.paper_workflow import PaperWorkflow
from quantpits.research.paper_store import latest_state
from quantpits.training.lease import TrainingExecutionLease
from quantpits.utils.workspace import fingerprint_file, fingerprint_value


def now():
    return datetime.now(timezone.utc).isoformat()


def _paper_boundary(workflow):
    journal = workflow._journal()
    last = journal["valuations"][-1]["result"] if journal["valuations"] else journal["bootstrap"]["valuation"]
    return {"valuation_frontier": last["date"], "retry_frontier": last["status"] != "COMPLETE",
            "state_date": latest_state(journal)["as_of_date"],
            "pending": [{"cycle_id": p["cycle_id"], "trade_date": p["trade_date"]}
                        for p in sorted(workflow._pending(journal), key=lambda p: (p["trade_date"], p["cycle_id"]))]}


def _paper_sequence(member, request, boundary, run_prefix):
    """Freeze dates from the account frontier; interleave native close/settle calls."""
    final_id = member.id + "-value"
    def step(key, kind, dependencies=(), **fields):
        return dict(id=key, kind=kind, member=member.id, dependencies=list(dependencies),
                    run_id=run_prefix + "-" + key, **fields)
    through = request["value_through"]
    frontier, state_date = boundary["valuation_frontier"], boundary["state_date"]
    if max(frontier, state_date) > through:
        return [step(final_id, "blocked", reason="account valuation/state frontier is after value-through")]
    market = PaperMarket(member.ctx.qlib_data_dir)
    if market.check_date(through) != "VERIFIED":
        # No account steps have run yet. Resume can resolve the same bounded
        # request when its calendar arrives, retaining the original frontier/IDs.
        return [step(final_id, "paper_schedule", boundary=boundary)]
    calendar = market.calendar if through <= market.calendar[-1] else market.future
    dates = [d for d in calendar if state_date <= d <= through and
             (d > frontier or (d == frontier and boundary["retry_frontier"]))]
    if not dates:
        dates = [through]  # observe an already saved endpoint without replaying it
    pending = [(i, p) for i, p in enumerate(boundary["pending"]) if p["trade_date"] <= through]
    result, dependencies = [], []
    for selected in dates:
        while pending and pending[0][1]["trade_date"] <= selected:
            index, plan = pending.pop(0)
            item = step(member.id + "-settle-" + str(index), "settle", dependencies,
                        native_cycle=plan["cycle_id"])
            result.append(item)
            dependencies = [item["id"]]
        key = final_id if selected == through else final_id + "-" + selected
        item = step(key, "value", dependencies, date=selected)
        result.append(item)
        dependencies = [key]
    return result


def _capture_rules(project, steps):
    for step in steps:
        if step["kind"] not in ("settle", "paper_plan"):
            continue
        binding = project.accounts[step["member"]]
        value = (binding.get("pending_rules", {}).get(step["native_cycle"]) if step["kind"] == "settle"
                 else binding.get("rules", project.rules))
        if value:
            source = path_at(project.path.parent, value)
            step["rules_source"] = str(source)
            step["rules_sha256"] = fingerprint_file(source) if source.is_file() else None


def _resolve_paper_schedules(project, state):
    changed = False
    for previous in tuple(state["steps"]):
        if previous["kind"] != "paper_schedule":
            continue
        try:
            member = project.members[previous["member"]]
            sequence = _paper_sequence(member, state["request"], previous["boundary"],
                                       previous["run_id"].rsplit("-" + previous["id"], 1)[0])
            if sequence[0]["kind"] == "paper_schedule":
                continue
            _capture_rules(project, sequence)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue  # this account keeps waiting; independent branches can run
        index = state["steps"].index(previous)
        state["steps"][index:index + 1] = sequence
        del state["results"][previous["id"]]
        for item in sequence:
            item["skip"] = item["id"] in state["request"]["inputs"].get("skip", [])
            state["results"][item["id"]] = {"status": "pending"}
        changed = True
    if changed:
        state["steps_identity"] = fingerprint_value(state["steps"])
    return changed


def build_steps(project, request):
    steps = []
    prefix = "cycle-" + fingerprint_value({"project": str(project.path), "request": request})[:16]
    held = request["inputs"].get("hold", [])
    old_dependencies = {}

    def add(key, kind, dependencies=(), **fields):
        step = dict(id=key, kind=kind, dependencies=list(dependencies), **fields)
        step.setdefault("run_id", prefix + "-" + key)
        steps.append(step)
        return key

    for member in project.group.members:
        deps = []
        if member.kind == "paper":
            try:
                workflow = PaperWorkflow(member.ctx)
                boundary = _paper_boundary(workflow)
                steps.extend(_paper_sequence(member, request, boundary, prefix))
                deps = [member.id + "-value"]
                if member.id not in held:
                    for plan in boundary["pending"]:
                        if plan["trade_date"] > request["value_through"]:
                            deps.append(add(member.id + "-pending", "blocked", deps, member=member.id,
                                            reason="original pending %s is not due by value-through" % plan["cycle_id"]))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                deps = [add(member.id + "-value", "blocked", member=member.id, reason=str(exc))]
            old_dependencies[member.id] = deps
        else:
            from quantpits.post_trade.transaction import PostTradeTransactionManager
            prod = read_json(member.ctx.config_path("prod_config.json"))
            cursor = prod.get("last_processed_date", prod.get("current_date"))
            active = PostTradeTransactionManager(member.ctx).active()
            if len(active) > 1:
                deps.append(add(member.id + "-account", "blocked", member=member.id,
                                reason="multiple unfinished native transactions require inspection"))
            elif active:
                journal = active[0]
                deps.append(add(member.id + "-recover", "post_recover", member=member.id,
                                run_id=journal.run_id, scope=journal.scope, start=next_day(journal.cursor_before), through=journal.cursor_after))
                cursor = journal.cursor_after
            if cursor > request["value_through"]:
                deps.append(add(member.id + "-frontier", "blocked", member=member.id,
                                reason="account cursor is after value-through"))
            post = add(member.id + "-post", "post_trade", deps, member=member.id,
                       start=next_day(cursor), through=request["value_through"], scope="all",
                       already_current=cursor == request["value_through"])
            old_dependencies[member.id] = [post]
    for key, spec in project.signals.items():
        ctx = project.members[spec["producer"]].ctx
        fit = contained(ctx.root, spec["fit_records"])
        predict = add(key + "-predict", "predict", signal=key, member=spec["producer"],
                      fit_sha256=fingerprint_file(fit) if fit.is_file() else None)
        add(key + "-fuse", "fuse", [predict], signal=key, member=spec["producer"])
    if project.prepare_rules is not None:
        producer = next(iter(project.signals.values()))["producer"]
        add("prepare-rules", "rules", member=producer)
    for member in project.group.members:
        binding = project.accounts[member.id]
        if member.id in held:
            add(member.id + "-hold", "hold", member=member.id, reason="explicit HOLD; no new plan/order requested")
            continue
        deps = old_dependencies[member.id] + [binding["signal"] + "-fuse"]
        if project.prepare_rules is not None and not binding.get("rules", project.rules) and member.kind == "paper":
            deps.append("prepare-rules")
        fields = dict(member=member.id, signal=binding["signal"])
        if member.kind == "paper":
            fields["native_cycle"] = request["inputs"].get("paper_cycles", {}).get(member.id, request["cycle"])
        add(member.id + "-new", "paper_plan" if member.kind == "paper" else "order", deps, **fields)
    add("report", "report")  # reports can describe partial members while another branch waits
    skip = request["inputs"].get("skip", [])
    if not isinstance(skip, list) or len(set(skip)) != len(skip) or set(skip) - {s["id"] for s in steps}:
        raise CycleError("skip must contain distinct known step IDs from plan")
    for step in steps:
        step["skip"] = step["id"] in skip
    _capture_rules(project, steps)
    return steps


def _preflight(project, request, step):
    """Light read-only readiness; missing data is local to its own branch."""
    kind = step["kind"]
    if kind == "blocked":
        return step["reason"]
    if kind in ("report", "hold"):
        return None
    if kind == "paper_schedule":
        return "WAITING_FOR_DATA: valuation calendar through %s" % request["value_through"]
    if step.get("rules_source") and not path_at(project.path.parent, step["rules_source"]).is_file():
        return "dated rules file is missing: " + step["rules_source"]
    member = project.members[step["member"]]
    if kind == "predict":
        frozen = project.signal_dir(request["cycle"], step["signal"]) / "fit_records.json"
        fit = frozen if frozen.is_file() else contained(member.ctx.root, project.signals[step["signal"]]["fit_records"])
        if not fit.is_file():
            return "fixed fit source is missing: %s" % fit
        from quantpits.training.command import TrainingRunOptions, prepare_training_run
        spec = project.signals[step["signal"]]
        prepare_training_run(ctx=member.ctx, options=TrainingRunOptions(family="static", action="predict_only",
            models=tuple(spec["models"]), source_records=str(fit), anchor=request["anchor"],
            run_id=step["run_id"], explain_plan=True))
    market = PaperMarket(member.ctx.qlib_data_dir)
    selected = request["anchor"]
    if kind == "settle":
        plan = PaperWorkflow(member.ctx)._journal()["plans"][step["native_cycle"]]
        selected = plan["trade_date"]
    elif kind in ("value", "post_trade", "post_recover"):
        selected = step.get("date", request["value_through"]) if kind == "value" else step["through"]
    if market.check_date(selected, actual=True) != "VERIFIED":
        return "WAITING_FOR_DATA: %s" % selected
    if kind in ("paper_plan", "order"):
        if market.check_date(request["trade_date"]) != "VERIFIED":
            return "WAITING_FOR_DATA: selected execution date is outside the available future calendar"
    if kind == "post_trade" and not step.get("already_current"):
        from quantpits.post_trade.command import PostTradeRunOptions, prepare_post_trade_run
        options = request["inputs"].get("actual", {}).get(member.id, {})
        bundle = options.get("settlement_bundle")
        if bundle and not contained(member.ctx.root, bundle).is_file():
            return "settlement bundle is missing: %s" % bundle
        prepared = prepare_post_trade_run(member.ctx, PostTradeRunOptions(
            scope="all", start_date=step["start"], end_date=step["through"], run_id=step["run_id"],
            settlement_bundle=bundle,
            allow_missing_settlement=options.get("allow_missing_settlement", False)))
        if not bundle and not options.get("allow_missing_settlement", False):
            sessions = [d for d in market.calendar if step["start"] <= d <= step["through"]]
            present = {s.trade_date for s in prepared.catalog.settlement_sources if s.status in ("present", "already_ingested")}
            missing = sorted(set(sessions) - present)
            if missing:
                return "settlement evidence missing: " + ", ".join(missing)
    return None


def preview(project, request):
    steps = build_steps(project, request)
    rows = []
    for step in steps:
        try:
            reason = _preflight(project, request, step)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            reason = str(exc)
        status = "held" if step["kind"] == "hold" else ("skipped" if step["skip"] else ("waiting" if reason else ("dependency" if step["dependencies"] else "ready")))
        rows.append(dict(step, status=status, reason=reason,
                         source="pending_generation" if step["kind"] in ("predict", "fuse") else None))
    return {"schema_version": 1, "mode": "plan", "project": str(project.path), "request": request,
            "signals": project.signals, "accounts": project.accounts, "contexts": {m.id: {
                "workspace": str(m.ctx.root), "mlflow_uri": m.ctx.mlflow_uri,
                "provider": str(m.ctx.qlib_data_dir), "region": m.ctx.qlib_region,
            } for m in project.group.members}, "steps": rows}


def load_state(project, cycle):
    path = project.cycle_dir(cycle) / "progress.json"
    value = read_json(path)
    if value.get("schema_version") != 1 or value.get("project_identity") != project.identity or value.get("project") != str(project.path):
        raise CycleError("cycle project/group configuration changed; select a new cycle or restore original config")
    request = value["request"]
    normalized = cycle_request(project, cycle=cycle, anchor=request["anchor"], value_through=request["value_through"],
                               trade_date=request["trade_date"], since=request["since"], inputs=request["inputs"])
    if request != normalized or value.get("request_identity") != fingerprint_value(request):
        raise CycleError("saved cycle request identity mismatch")
    ids = [s["id"] for s in value["steps"]]
    if len(ids) != len(set(ids)) or set(value["results"]) != set(ids):
        raise CycleError("cycle progress does not preserve its requested steps")
    if value.get("steps_identity") != fingerprint_value(value["steps"]):
        raise CycleError("saved step requests changed")
    for step in value["steps"]:
        if step["skip"] != (step["id"] in request["inputs"].get("skip", [])) or set(step["dependencies"]) - set(ids):
            raise CycleError("saved step scope/dependencies differ from the cycle request")
        if value["results"][step["id"]].get("status") not in ("pending", "running", "done", "failed", "waiting", "interrupted", "held", "skipped"):
            raise CycleError("invalid saved step status")
    return value


def summarize(state):
    rows = []
    for step in state["steps"]:
        rows.append(dict(step, **state["results"][step["id"]]))
    statuses = [r["status"] for r in rows]
    status = "complete" if all(s in ("done", "held", "skipped") for s in statuses) else (
        "interrupted" if "interrupted" in statuses else ("failed" if "failed" in statuses else "waiting"))
    return {"schema_version": 1, "mode": "status", "project": state["project"], "request": state["request"],
            "status": status, "steps": rows}


def status(project, cycle):
    state = load_state(project, cycle)
    # Never create backend, leases or progress while inspecting saved facts.
    for entry in state["results"].values():
        if entry["status"] == "done":
            verify_refs(entry.get("evidence", {}))
    native = NativeSteps(project, state["request"], None, project.cycle_dir(cycle))
    for step in state["steps"]:
        entry = state["results"][step["id"]]
        if entry["status"] == "done" and step["kind"] != "report":
            observed = native.observe(step, completed=True)
            if observed is None or observed != entry["evidence"]:
                raise CycleError("completed result differs from its saved native proof")
    return summarize(state)


def execute(project, *, request=None, cycle=None, runner=None, fault=None):
    cycle = cycle or request["cycle"]
    directory = project.cycle_dir(cycle)
    for member in project.group.members:
        validate_write_context(member.ctx)
        # Progress/reports must be outside account and provider trees.
        for root in (member.ctx.root, member.ctx.qlib_data_dir):
            try:
                directory.relative_to(root.resolve())
            except ValueError:
                pass
            else:
                raise CycleError("project progress must be outside account/provider directories")
    lease = TrainingExecutionLease(contained(directory, "execution.lock"))
    lease.acquire(run_id=cycle)
    try:
        path = directory / "progress.json"
        if path.exists():
            state = load_state(project, cycle)
            if request is not None and request != state["request"]:
                raise CycleError("cycle request changed; use a new cycle ID; existing paper plans remain frozen")
        elif request is None:
            raise CycleError("cycle does not exist; use run with explicit dates")
        else:
            state = {"schema_version": 1, "project": str(project.path), "project_identity": project.identity,
                     "request": request, "request_identity": fingerprint_value(request), "steps": build_steps(project, request),
                     "results": {}, "created_at": now()}
            state["results"] = {s["id"]: {"status": "pending"} for s in state["steps"]}
            state["steps_identity"] = fingerprint_value(state["steps"])
            save_json(path, state)
        request = state["request"]
        if _resolve_paper_schedules(project, state):
            save_json(path, state)
        logs = contained(directory, "logs")
        logs.mkdir(parents=True, exist_ok=True)
        native = NativeSteps(project, request, runner or ProcessRunner(), directory)
        for step in state["steps"]:
            entry = state["results"][step["id"]]
            if step["kind"] == "report":
                signature = fingerprint_value({key: value for key, value in state["results"].items() if key != step["id"]})
                if entry.get("report_for") != signature:
                    entry.update(status="pending", report_for=signature)
            if step["skip"] or step["kind"] == "hold":
                entry.update(status="skipped" if step["skip"] else "held", reason="explicit cycle selection")
                save_json(path, state)
                continue
            if entry["status"] == "done":
                try:
                    verify_refs(entry["evidence"])
                    if step["kind"] != "report":
                        observed = native.observe(step, completed=True)
                        if observed is None or observed != entry["evidence"]:
                            raise CycleError("completed result differs from its saved native proof")
                except Exception as exc:
                    entry.update(status="failed", reason=str(exc), error_type=type(exc).__name__)
                    save_json(path, state)
                continue
            dependencies = [d for d in step["dependencies"] if state["results"][d]["status"] != "done"]
            if dependencies:
                entry.update(status="waiting", reason="dependencies: " + ", ".join(dependencies))
                save_json(path, state)
                continue
            try:
                evidence = native.observe(step)
                if evidence is None:
                    # Capture the first available fixed fit before native execution.
                    if step["kind"] == "predict" and step["fit_sha256"] is None:
                        source = contained(native.ctx(step).root, project.signals[step["signal"]]["fit_records"])
                        if source.is_file():
                            step["fit_sha256"] = fingerprint_file(source)
                            state["steps_identity"] = fingerprint_value(state["steps"])
                    reason = _preflight(project, request, step)
                    if reason:
                        entry.update(status="waiting", reason=reason)
                        save_json(path, state)
                        continue
                    entry.update(status="running", started_at=now(), reason=None)
                    save_json(path, state)  # intent precedes the native call
                    code = native.execute(step)
                    if fault is not None:
                        fault("after_native", step)
                    evidence = native.observe(step)
                    if step["kind"] == "report" and code == 0:
                        files = sorted((directory / "reports").glob("*"))
                        if not files:
                            raise CycleError("report command returned success without reports")
                        evidence = {"files": [ref(p) for p in files if p.is_file()]}
                    if evidence is None:
                        waiting = code == 2 and step["kind"] in ("settle", "value", "paper_plan", "report", "rules")
                        reason = "native completion not verified; see logs/%s.log" % step["id"]
                        if step["kind"] == "value" and code == 0:
                            selected = step.get("date", request["value_through"])
                            values = [v["result"] for v in native.paper(step)._journal()["valuations"] if v["result"]["date"] == selected]
                            if values and values[-1]["status"] != "COMPLETE":
                                waiting = True
                                reason = "native valuation at %s is %s" % (selected, values[-1]["status"])
                        entry.update(status="waiting" if waiting else "failed", exit_code=code,
                                     reason=reason)
                        save_json(path, state)
                        continue
                entry.update(status="done", evidence=evidence, finished_at=now(), reason=None)
                entry.pop("error_type", None)
                entry.pop("exit_code", None)
                save_json(path, state)
            except (KeyboardInterrupt, SystemExit):
                entry.update(status="interrupted", reason="process control interruption; native recovery facts retained", finished_at=now())
                save_json(path, state)
                raise
            except Exception as exc:
                entry.update(status="failed", reason=str(exc), error_type=type(exc).__name__, finished_at=now())
                save_json(path, state)
                # An ordinary member failure cannot suppress independent branches.
        return summarize(state)
    finally:
        lease.release()
