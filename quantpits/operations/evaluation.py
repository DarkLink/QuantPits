"""Optional saved-signal evaluation, independent of the core cycle state."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile

from quantpits.ensemble.saved_evaluation import evaluation_options
from quantpits.operations.config import contained, read_json
from quantpits.operations.cycle import load_state
from quantpits.operations.native import save_json
from quantpits.operations.process import ProcessRunner


def evaluate(project, cycle, *, start=None, end=None, account=100000, runner=None):
    options = evaluation_options(start, end, account)
    state = load_state(project, cycle)
    parent = contained(project.cycle_dir(cycle), "evaluations")
    for member in project.group.members:
        for root in (member.ctx.root, member.ctx.qlib_data_dir):
            if parent == root.resolve() or root.resolve() in parent.parents:
                raise ValueError("evaluation outputs must be outside account/provider trees")
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="evaluation-", dir=str(parent)))
    result = {"schema_version": 1, "mode": "evaluate", "cycle": cycle,
        "evaluation_id": directory.name, "directory": str(directory),
        "created_at": datetime.now(timezone.utc).isoformat(), "options": options, "groups": []}
    for key, spec in project.signals.items():
        members = [{"name": m + "@static", "kind": "model", "status": "pending"} for m in spec["models"]]
        members.append({"name": spec["combo"], "kind": "ensemble", "status": "pending"})
        result["groups"].append({"signal": key, "producer": spec["producer"], "status": "pending", "members": members})
    process = runner or ProcessRunner()
    try:
        for group in result["groups"]:
            key = group["signal"]
            spec = project.signals[key]
            ctx = project.members[spec["producer"]].ctx
            output = contained(directory, key)
            output.mkdir()
            try:
                predict, fuse = state["results"][key + "-predict"], state["results"][key + "-fuse"]
                if predict["status"] != "done" or fuse["status"] != "done":
                    raise ValueError("fixed prediction and fusion steps are not complete; evaluation cannot generate them")
                request = {"signal": key, "anchor": state["request"]["anchor"], "models": [r["name"] for r in group["members"][:-1]],
                    "combo": spec["combo"], "source_directory": str(project.signal_dir(cycle, key)),
                    "prediction_proof": predict["evidence"], "ensemble_proof": fuse["evidence"], "options": options}
                save_json(output / "request.json", request)
                code = process.run(ctx, "quantpits.scripts.evaluate_signals", [
                    "--workspace", str(ctx.root), "--mlflow-uri", ctx.mlflow_uri,
                    "--qlib-data-dir", str(ctx.qlib_data_dir), "--qlib-region", ctx.qlib_region,
                    "--request", str(output / "request.json"), "--output", str(output)], output / "worker.log")
                group["exit_code"] = code
                if not (output / "result.json").is_file():
                    raise ValueError("evaluation worker exit %s; see %s" % (code, output / "worker.log"))
                saved = read_json(output / "result.json")
                expected = [(m["name"], m["kind"]) for m in group["members"]]
                if saved.get("signal") != key or [(m["name"], m["kind"]) for m in saved["members"]] != expected:
                    raise ValueError("evaluation worker changed the requested group/member scope")
                statuses = [m["status"] for m in saved["members"]]
                if any(value not in ("complete", "failed", "unavailable", "blocked") for value in statuses):
                    raise ValueError("evaluation worker returned unfinished/unknown member states")
                expected_status = "complete" if all(value == "complete" for value in statuses) else (
                    "partial" if "complete" in statuses else "failed")
                if saved["status"] != expected_status or code != (0 if expected_status == "complete" else 2):
                    raise ValueError("evaluation worker completion differs from its member results/exit")
                for row in saved["members"]:
                    if row["status"] == "complete" and (not row.get("metrics") or not contained(output, row["trajectory"]).is_file()):
                        raise ValueError("completed evaluation has no metrics/trajectory")
                group.update(saved, directory=str(output))
            except Exception as exc:
                group.update(status="failed", error=str(exc), error_type=type(exc).__name__)
                for row in group["members"]:
                    row.update(status="failed", error=str(exc))
    except (KeyboardInterrupt, SystemExit):
        result["status"] = "interrupted"
        save_json(directory / "result.json", result)
        raise
    result["status"] = "complete" if all(g["status"] == "complete" for g in result["groups"]) else (
        "partial" if any(g["status"] in ("complete", "partial") for g in result["groups"]) else "failed")
    save_json(directory / "result.json", result)
    # Groups retain their own conditions and windows; there is no cross-group ranking.
    lines = ["# Saved-signal evaluations", "", "Status: " + result["status"], ""]
    for group in result["groups"]:
        path = directory / group["signal"] / "comparison.md"
        lines += ["## " + group["signal"], "", "Status: " + group["status"], ""]
        if path.is_file():
            lines += ["[Comparison](%s/comparison.md)" % group["signal"], ""]
        if group.get("error"):
            lines += [group["error"], ""]
    from quantpits.training.persistence import atomic_write_bytes
    atomic_write_bytes(directory / "comparison.md", "\n".join(lines).encode())
    return result
