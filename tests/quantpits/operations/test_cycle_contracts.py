from dataclasses import replace
import json
import signal
import sys
import subprocess

import pandas as pd
import pytest

from quantpits.operations.config import CycleError, contained, load_project
from quantpits.order.command import OrderPlanError, OrderRunOptions, load_order_run_config, prepare_order_run
from quantpits.order.execution import OrderSourceUnavailableError, load_resolved_prediction
from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.resolved import resolve_training_run
from tests.quantpits.operations.scenario import scenario, write_json
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file


def test_training_anchor_overrides_fixed_model_date_without_mutation(tmp_path, monkeypatch):
    project = load_project(scenario(tmp_path))
    ctx = project.members["live"].ctx
    model = ctx.config_path("model_config.json").read_bytes()
    options = TrainingRunOptions(family="static", action="predict_only", models=("small",), source_records="fit.json", anchor="2024-05-17")
    prepared = prepare_training_run(ctx=ctx, options=options)
    assert prepared.date_policy.configured_anchor == "2024-05-17"
    assert prepared.date_policy.resolution == "explicit"
    from qlib.data import D
    from quantpits.utils.train_utils import calculate_dates
    monkeypatch.setattr(D, "calendar", lambda **kw: pd.to_datetime(["2024-05-17"]), raising=False)
    params = calculate_dates(ctx=ctx, anchor="2024-05-17")
    assert params["anchor_date"] == params["test_end_time"] == "2024-05-17"
    initial = resolve_training_run(prepared, params)
    changed = resolve_training_run(prepare_training_run(ctx=ctx, options=replace(options, anchor="2024-05-21")), dict(params, anchor_date="2024-05-21", test_end_time="2024-05-21"))
    assert changed.resume_fingerprint != initial.resume_fingerprint
    assert ctx.config_path("model_config.json").read_bytes() == model
    monkeypatch.setattr(D, "calendar", lambda **kw: [], raising=False)
    with pytest.raises(ValueError, match="observed trading date"):
        calculate_dates(ctx=ctx, anchor="2024-05-18")


def test_explicit_ensemble_selection_never_falls_back_and_checks_bytes(tmp_path):
    project = load_project(scenario(tmp_path))
    ctx = project.members["live"].ctx
    prediction = tmp_path / "signal.csv"
    prediction.write_text("datetime,instrument,score\n2024-05-17,SH600001,1\n")
    fixed = tmp_path / "ensemble.json"
    write_json(fixed, {"default_combo": "selected", "combos": {"selected": "fixed-rid"},
        "predictions": {"selected": {"path": str(prediction), "sha256": fingerprint_file(prediction)}}})
    options = OrderRunOptions(ensemble_record_file=str(fixed), combo="missing", anchor="2024-05-17", trade_date="2024-05-21")
    with pytest.raises(OrderPlanError, match="no requested/default combo"):
        prepare_order_run(ctx=ctx, options=options, cli_args=(), run_config=load_order_run_config(ctx, options))
    options = replace(options, combo="selected")
    prepared = prepare_order_run(ctx=ctx, options=options, cli_args=(), run_config=load_order_run_config(ctx, options))
    assert load_resolved_prediction(prepared.source).data.iloc[0, 0] == 1
    prediction.write_text("datetime,instrument,score\n2024-05-17,SH600001,999\n")
    with pytest.raises(OrderSourceUnavailableError, match="digest mismatch"):
        load_resolved_prediction(prepared.source)


def test_writer_rejects_symlink_escape(tmp_path):
    from quantpits.operations.native import validate_write_context
    project = load_project(scenario(tmp_path))
    ctx = project.members["live"].ctx
    (ctx.root / "output").rmdir()
    (ctx.root / "output").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(CycleError, match="escapes"):
        validate_write_context(ctx)


def test_declared_completion_without_native_evidence_is_rejected_readonly(tmp_path):
    from quantpits.operations.config import cycle_request
    from quantpits.operations.cycle import build_steps, status
    from quantpits.operations.native import save_json
    from quantpits.utils.workspace import fingerprint_value
    from tests.quantpits.semantic.test_project_cycle import files
    project = load_project(scenario(tmp_path))
    request = cycle_request(project, cycle="forged", anchor="2024-05-17", value_through="2024-05-17",
                            trade_date="2024-05-21", since="2024-05-10")
    steps = build_steps(project, request)
    state = {"schema_version": 1, "project": str(project.path), "project_identity": project.identity,
        "request": request, "request_identity": fingerprint_value(request), "steps": steps, "steps_identity": fingerprint_value(steps),
        "results": {s["id"]: {"status": "done", "evidence": {"files": []}} for s in steps}}
    save_json(project.cycle_dir("forged") / "progress.json", state)
    before = files(tmp_path)
    with pytest.raises(CycleError, match="native proof"):
        status(project, "forged")
    assert files(tmp_path) == before


def test_process_interrupt_propagates_and_reaps_child(tmp_path):
    from quantpits.operations.process import ProcessRunner
    package = tmp_path / "child.py"
    # The child requests a parent interrupt only after it has installed its own
    # handler, then proves that forwarded SIGINT reached it before parent returns.
    package.write_text("import os,signal\nfrom pathlib import Path\n"
        "def done(a,b):\n Path(os.environ['PROOF']).write_text('reaped')\n raise SystemExit(130)\n"
        "signal.signal(signal.SIGINT,done)\nos.kill(os.getppid(),signal.SIGINT)\nsignal.pause()\n")
    parent = tmp_path / "parent.py"
    parent.write_text("from quantpits.operations.process import ProcessRunner\nfrom pathlib import Path\n"
        "try:\n ProcessRunner().run(None,'child',[],Path('child.log'))\nexcept KeyboardInterrupt:\n raise SystemExit(130)\n")
    import os
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + os.getcwd(), PROOF=str(tmp_path / "proof"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(parent)], cwd=tmp_path, env=env, capture_output=True, timeout=20)
    assert result.returncode == 130
    assert (tmp_path / "proof").read_text() == "reaped"


def test_unavailable_calendar_keeps_original_frontier_until_schedule_resolution(tmp_path):
    from quantpits.operations.config import cycle_request
    from quantpits.operations.cycle import build_steps, _resolve_paper_schedules
    project = load_project(scenario(tmp_path))
    request = cycle_request(project, cycle="waiting_calendar", anchor="2024-05-23", value_through="2024-05-23",
                            trade_date="2024-05-24", since="2024-05-10")
    steps = build_steps(project, request)
    placeholders = [s for s in steps if s["kind"] == "paper_schedule"]
    assert len(placeholders) == 2
    assert all(s["boundary"]["valuation_frontier"] == "2024-05-10" for s in placeholders)
    state = {"steps": steps, "request": request, "results": {s["id"]: {"status": "waiting"} for s in steps}}
    assert not _resolve_paper_schedules(project, state)
    provider = project.members["live"].ctx.qlib_data_dir
    calendar = ["2024-05-10", "2024-05-13", "2024-05-17", "2024-05-20", "2024-05-21", "2024-05-22", "2024-05-23"]
    (provider / "calendars/day.txt").write_text("\n".join(calendar) + "\n")
    (provider / "calendars/day_future.txt").write_text("\n".join(calendar + ["2024-05-24"]) + "\n")
    assert _resolve_paper_schedules(project, state)
    assert [s["date"] for s in state["steps"] if s["kind"] == "value" and s["member"] == "paper_a"] == calendar[1:]
    assert next(s for s in state["steps"] if s["id"] == "paper_a-settle-0")["native_cycle"] == "old_native_id"
