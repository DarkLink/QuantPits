import copy
import json
from pathlib import Path

import pytest

from quantpits.operations.config import cycle_request, load_project
from quantpits.operations.cycle import build_steps
from quantpits.operations.evaluation import evaluate
from quantpits.operations.native import save_json
from quantpits.scripts import run_cycle
from quantpits.utils.workspace import fingerprint_value
from tests.quantpits.operations.scenario import scenario


@pytest.fixture
def stored_cycle(tmp_path):
    project = load_project(scenario(tmp_path))
    request = cycle_request(project, cycle="old", anchor="2024-05-17", value_through="2024-05-17",
        trade_date="2024-05-21", since="2024-05-10")
    steps = build_steps(project, request)
    state = {"schema_version": 1, "project": str(project.path), "project_identity": project.identity,
        "request": request, "request_identity": fingerprint_value(request), "steps": steps,
        "steps_identity": fingerprint_value(steps),
        "results": {s["id"]: {"status": "done", "evidence": {"files": []}} for s in steps}}
    path = project.cycle_dir("old") / "progress.json"
    save_json(path, state)
    return project, path


class Worker:
    def __init__(self, error=None, malformed=False):
        self.calls = []
        self.error = error
        self.malformed = malformed

    def run(self, ctx, module, arguments, log):
        self.calls.append((ctx, module, arguments))
        if self.error is not None and len(self.calls) == 1:
            raise self.error
        request = json.loads(Path(arguments[arguments.index("--request") + 1]).read_text())
        output = Path(arguments[arguments.index("--output") + 1])
        members = []
        for name, kind in [(m, "model") for m in request["models"]] + [(request["combo"], "ensemble")]:
            filename = "trajectory-%s.csv" % len(members)
            (output / filename).write_text("nav\n1.0\n")
            members.append({"name": name, "kind": kind, "status": "complete", "metrics": {"final_nav": 1}, "trajectory": filename})
        if self.malformed:
            members.pop()
        save_json(output / "result.json", {"status": "complete", "signal": request["signal"], "members": members,
            "conditions": {"initial_cash": request["options"]["account"]},
            "window": {"start": "2024-05-13" if request["signal"] == "shared" else "2024-05-14", "end": "2024-05-17"}})
        return 0


def test_evaluate_keeps_old_progress_and_new_runs_have_unique_directories(stored_cycle):
    project, path = stored_cycle
    before = path.read_bytes()
    worker = Worker()
    first = evaluate(project, "old", runner=worker)
    second = evaluate(project, "old", runner=worker, account=3210)
    assert first["status"] == second["status"] == "complete"
    assert first["directory"] != second["directory"]
    assert first["groups"][0]["conditions"]["initial_cash"] == 100000
    assert second["groups"][0]["conditions"]["initial_cash"] == 3210
    assert path.read_bytes() == before
    assert [call[0].root for call in worker.calls[:2]] == [project.members[s["producer"]].ctx.root for s in project.signals.values()]
    assert {g["window"]["start"] for g in first["groups"]} == {"2024-05-13", "2024-05-14"}
    assert "leaderboard" not in first


def test_group_failure_retains_scope_and_continues_next_group(stored_cycle):
    project, path = stored_cycle
    before = path.read_bytes()
    worker = Worker(RuntimeError("first group failed"))
    result = evaluate(project, "old", runner=worker)
    assert result["status"] == "partial" and len(worker.calls) == 2
    assert [g["signal"] for g in result["groups"]] == list(project.signals)
    assert [g["status"] for g in result["groups"]] == ["failed", "complete"]
    assert len(result["groups"][0]["members"]) == 2
    assert path.read_bytes() == before


def test_evaluation_never_generates_missing_core_sources(stored_cycle):
    project, path = stored_cycle
    state = json.loads(path.read_text())
    for key in ("shared-predict", "candidate-fuse"):
        state["results"][key]["status"] = "waiting"
    save_json(path, state)
    before = path.read_bytes()
    worker = Worker()
    result = evaluate(project, "old", runner=worker)
    assert result["status"] == "failed" and not worker.calls
    assert path.read_bytes() == before


def test_worker_cannot_shrink_members_and_claim_completion(stored_cycle):
    project, _ = stored_cycle
    result = evaluate(project, "old", runner=Worker(malformed=True))
    assert result["status"] == "failed"
    assert all(len(g["members"]) == 2 for g in result["groups"])


@pytest.mark.parametrize("mismatch", ["aggregate", "exit", "unfinished"])
def test_worker_result_must_match_members_and_exit(stored_cycle, mismatch):
    class InconsistentWorker(Worker):
        def run(self, ctx, module, arguments, log):
            super().run(ctx, module, arguments, log)
            path = Path(arguments[arguments.index("--output") + 1]) / "result.json"
            saved = json.loads(path.read_text())
            if mismatch == "aggregate":
                saved["status"] = "partial"
                for row in saved["members"]:
                    row["status"] = "failed"
            elif mismatch == "unfinished":
                saved["status"] = "failed"
                saved["members"][0]["status"] = "running"
            save_json(path, saved)
            return 1 if mismatch == "exit" else 2
    project, _ = stored_cycle
    result = evaluate(project, "old", runner=InconsistentWorker())
    assert result["status"] == "failed"
    assert all(g["status"] == "failed" for g in result["groups"])


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_evaluation_interrupt_preserves_core_progress(stored_cycle, interrupt):
    project, path = stored_cycle
    before = path.read_bytes()
    with pytest.raises(interrupt):
        evaluate(project, "old", runner=Worker(interrupt("stop")))
    assert path.read_bytes() == before
    result_path = next((path.parent / "evaluations").glob("*/result.json"))
    assert json.loads(result_path.read_text())["status"] == "interrupted"


@pytest.mark.parametrize("action", ["run", "resume"])
def test_attached_evaluation_failure_keeps_core_status_and_exit(stored_cycle, monkeypatch, capsys, action):
    project, path = stored_cycle
    before = path.read_bytes()
    core = {"status": "complete", "request": {"old_field": "unchanged"}}
    monkeypatch.setattr(run_cycle, "execute", lambda *a, **kw: copy.deepcopy(core))
    def fail(*args, **kwargs):
        raise OSError("diagnostic failure")
    arguments = [action, "--project", str(project.path), "--cycle", "old", "--evaluate"]
    if action == "run":
        arguments += ["--anchor", "2024-05-17", "--value-through", "2024-05-17", "--trade-date", "2024-05-21", "--since", "2024-05-10"]
    assert run_cycle.main(arguments, evaluator=fail) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete" and result["request"] == core["request"]
    assert result["evaluation"]["status"] == "failed"
    assert path.read_bytes() == before


def test_standalone_failure_nonzero_and_status_does_not_evaluate(stored_cycle, monkeypatch, capsys):
    project, _ = stored_cycle
    arguments = ["--project", str(project.path), "--cycle", "old"]
    assert run_cycle.main(["evaluate"] + arguments, evaluator=lambda *a, **kw: {"status": "partial"}) == 2
    capsys.readouterr()
    monkeypatch.setattr(run_cycle, "status", lambda *a: {"status": "complete"})
    assert run_cycle.main(["status"] + arguments, evaluator=lambda *a, **kw: pytest.fail("status must not evaluate")) == 0
    assert "evaluation" not in json.loads(capsys.readouterr().out)


def test_diagnostic_cash_without_opt_in_is_rejected_before_core(monkeypatch):
    monkeypatch.setattr(run_cycle, "execute", lambda *a, **kw: pytest.fail("must not run core"))
    with pytest.raises(SystemExit) as error:
        run_cycle.main(["resume", "--project", "unused", "--cycle", "old", "--account", "500"])
    assert error.value.code == 2


def test_attached_evaluation_waits_for_core_completion(stored_cycle, monkeypatch, capsys):
    project, _ = stored_cycle
    monkeypatch.setattr(run_cycle, "execute", lambda *a, **kw: {"status": "waiting"})
    assert run_cycle.main(["resume", "--project", str(project.path), "--cycle", "old", "--evaluate"],
        evaluator=lambda *a, **kw: pytest.fail("incomplete core cannot start evaluation")) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "waiting" and result["evaluation"]["status"] == "not_started"
