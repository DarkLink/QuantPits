"""Original and overridden cycles keep their own source and recovery identities."""

from quantpits.operations.config import load_project
from tests.quantpits.operations.scenario import scenario, write_json
from tests.quantpits.semantic.test_project_cycle import command, files, read, rows


SESSIONS = ["2024-05-10", "2024-05-13", "2024-05-17", "2024-05-20", "2024-05-21", "2024-05-24", "2024-05-27", "2024-05-28"]


def _entry(project, cycle, signal):
    selected = load_project(project)
    return read(selected.signal_dir(cycle, signal) / "predictions.json")["model_records"]["small@static"]


def _business_files(root):
    # Explicit resume acquires the existing execution lease and refreshes its
    # owner metadata. No model/account/progress artifact may change with it.
    return {p: h for p, h in files(root).items() if not p.endswith("/execution.lock")}


def test_new_cycle_override_leaves_project_and_old_completed_cycle_unchanged(tmp_path):
    project = scenario(tmp_path, sessions=SESSIONS)
    original_project = project.read_bytes(), load_project(project).identity
    old = command(project, cycle="old_week")
    assert old["status"] == "complete" and "fit_records" not in old["request"]["inputs"]
    old_progress = (project.parent / "cycles/old_week/progress.json").read_bytes()
    chosen = tmp_path / "paper_b/data/choice.json"
    write_json(chosen, {"experiment_name": "fit", "models": {"small@static": "explicit-B"}})
    inputs = project.parent / "new_inputs.json"
    write_json(inputs, {"fit_records": {"candidate": "data/choice.json"}})
    preview = command(project, "plan", cycle="new_week", inputs=inputs,
                      anchor="2024-05-24", value_through="2024-05-24", trade_date="2024-05-28", since="2024-05-17")
    assert preview["signals"]["candidate"]["fit_records"] == "data/choice.json"
    result = command(project, cycle="new_week", inputs=inputs,
                     anchor="2024-05-24", value_through="2024-05-24", trade_date="2024-05-28", since="2024-05-17")
    assert _entry(project, "old_week", "candidate")["source_recorder_id"] == "original-fit"
    assert _entry(project, "new_week", "candidate")["source_recorder_id"] == "explicit-B"
    assert _entry(project, "new_week", "shared")["source_recorder_id"] == "original-fit"
    assert (project.read_bytes(), load_project(project).identity) == original_project
    before = files(tmp_path)
    assert command(project, "status", cycle="old_week")["status"] == "complete"
    assert files(tmp_path) == before
    business = _business_files(tmp_path)
    assert command(project, "resume", cycle="old_week")["status"] == "complete"
    assert command(project, "resume", cycle="new_week")["status"] == "complete"
    assert _business_files(tmp_path) == business
    assert (project.parent / "cycles/old_week/progress.json").read_bytes() == old_progress
    # The independent evaluation receives the cycle's saved prediction proofs,
    # even if the external source files now describe another model copy.
    from quantpits.operations.evaluation import evaluate
    from tests.quantpits.operations.test_cycle_evaluation import Worker
    write_json(chosen, {"experiment_name": "later", "models": {"small@static": "changed-B"}})
    write_json(tmp_path / "paper_b/fit.json", {"experiment_name": "later", "models": {"small@static": "changed-default"}})
    class CheckEvaluation(Worker):
        def run(self, ctx, module, arguments, log):
            from pathlib import Path
            payload = read(Path(arguments[arguments.index("--request") + 1]))
            if payload["signal"] == "candidate":
                entry = read(Path(payload["source_directory"]) / "predictions.json")["model_records"]["small@static"]
                assert entry["source_recorder_id"] == "explicit-B"
            return super().run(ctx, module, arguments, log)
    core = (project.parent / "cycles/new_week/progress.json").read_bytes()
    assert evaluate(load_project(project), "new_week", runner=CheckEvaluation())["status"] == "complete"
    assert (project.parent / "cycles/new_week/progress.json").read_bytes() == core


def test_interrupted_cycle_resumes_its_saved_override_and_fixed_bytes(tmp_path):
    project = scenario(tmp_path)
    selected = load_project(project)
    chosen = tmp_path / "paper_b/data/choice.json"
    write_json(chosen, {"experiment_name": "fit", "models": {"small@static": "explicit-B"}})
    inputs = project.parent / "inputs.json"
    write_json(inputs, {"fit_records": {"candidate": "data/choice.json"}})
    (project.parent / "interrupt-after").write_text("candidate-predict")
    command(project, code=130, inputs=inputs)
    progress = read(project.parent / "cycles/new_week/progress.json")
    assert rows(progress)["candidate-predict"].get("fit_sha256")
    frozen = selected.signal_dir("new_week", "candidate") / "fit_records.json"
    expected = frozen.read_bytes()
    write_json(inputs, {"fit_records": {"candidate": "fit.json"}})
    chosen.unlink()
    write_json(tmp_path / "paper_b/latest_train_records.json", {"models": {"small@static": "changed-latest"}, "experiment_name": "later"})
    result = command(project, "resume")
    assert result["request"]["inputs"]["fit_records"]["candidate"] == "data/choice.json"
    assert _entry(project, "new_week", "candidate")["source_recorder_id"] == "explicit-B"
    assert frozen.read_bytes() == expected
    before = files(tmp_path)
    assert command(project, "status")["status"] == "complete"
    assert files(tmp_path) == before
    business = _business_files(tmp_path)
    assert command(project, "resume")["status"] == "complete"
    assert _business_files(tmp_path) == business


def test_missing_override_waits_its_branch_then_captures_original_path(tmp_path):
    project = scenario(tmp_path)
    inputs = project.parent / "inputs.json"
    write_json(inputs, {"fit_records": {"candidate": "data/later.json"}})
    waiting = command(project, code=2, inputs=inputs)
    assert rows(waiting)["candidate-predict"]["status"] == "waiting"
    assert rows(waiting)["shared-predict"]["status"] == rows(waiting)["live-new"]["status"] == "done"
    assert rows(waiting)["candidate-predict"]["fit_sha256"] is None
    chosen = tmp_path / "paper_b/data/later.json"
    write_json(chosen, {"experiment_name": "fit", "models": {"small@static": "arrived-B"}})
    write_json(inputs, {"fit_records": {"candidate": "fit.json"}})
    completed = command(project, "resume")
    assert rows(completed)["candidate-predict"]["fit_sha256"] is not None
    assert _entry(project, "new_week", "candidate")["source_recorder_id"] == "arrived-B"


def test_missing_selected_members_never_shrink_or_fall_back(tmp_path):
    project = scenario(tmp_path)
    write_json(tmp_path / "paper_b/data/wrong.json", {"experiment_name": "fit", "models": {"unused@static": "wrong"}})
    inputs = project.parent / "inputs.json"
    write_json(inputs, {"fit_records": {"candidate": "data/wrong.json"}})
    result = command(project, code=2, inputs=inputs)
    assert rows(result)["candidate-predict"]["status"] == "failed"
    assert "missing selected models" in rows(result)["candidate-predict"]["reason"]
    assert rows(result)["shared-predict"]["status"] == "done"
