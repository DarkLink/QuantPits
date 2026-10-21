"""Real cycle CLI → native services → three independent temporary accounts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.quantpits.operations.scenario import broker, scenario


def command(project, action="run", code=0, *, cycle="new_week", inputs=None, parent_env="wrong",
            anchor="2024-05-17", value_through="2024-05-17", trade_date="2024-05-21", since="2024-05-10"):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", QLIB_WORKSPACE_DIR="/must/not/use/inherited",
                       MLFLOW_TRACKING_URI="sqlite:////must/not/use/inherited.db")
    if parent_env == "missing":
        for name in ("QLIB_WORKSPACE_DIR", "MLFLOW_TRACKING_URI", "QLIB_DATA_DIR", "QLIB_REGION"):
            environment.pop(name, None)
    arguments = [action, "--project", str(project), "--cycle", cycle]
    if action in ("run", "plan"):
        arguments += ["--anchor", anchor, "--value-through", value_through, "--trade-date", trade_date, "--since", since]
        if inputs:
            arguments += ["--inputs", str(inputs)]
    result = subprocess.run([sys.executable, "-m", "tests.quantpits.operations.cycle_worker", "cycle"] + arguments,
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, env=environment, timeout=120)
    if result.returncode != code:
        try:
            failed = [(r["id"], r["status"], r.get("reason")) for r in json.loads(result.stdout).get("steps", []) if r["status"] != "done"]
        except ValueError:
            failed = result.stdout
        pytest.fail("exit %s, expected %s: %s\n%s" % (result.returncode, code, failed, result.stderr))
    if code == 130:
        return json.loads(result.stderr.splitlines()[-1])
    payload = json.loads(result.stdout)
    if code == 0 and action != "plan":
        assert payload["status"] == "complete", result.stdout
    return payload


def files(root):
    from quantpits.utils.workspace import fingerprint_file
    return {str(p.relative_to(root)): fingerprint_file(p) for p in root.rglob("*") if p.is_file()}


def read(path):
    return json.loads(path.read_text())


def rows(result):
    return {r["id"]: r for r in result["steps"]}


@pytest.mark.parametrize("parent_env", ["wrong", "missing"])
def test_three_workspace_cycle_and_readonly_cli(tmp_path, monkeypatch, parent_env):
    project = scenario(tmp_path)
    # Full-suite collection can import legacy scripts that change cwd. The
    # subprocess worker must remain importable outside the repository as well.
    monkeypatch.chdir(tmp_path)
    before = files(tmp_path)
    preview = command(project, "plan", parent_env=parent_env)
    assert files(tmp_path) == before
    assert rows(preview)["shared-fuse"]["source"] == "pending_generation"
    assert rows(preview)["paper_a-settle-0"]["native_cycle"] == "old_native_id"
    result = command(project, parent_env=parent_env)
    assert all(r["status"] == "done" for r in result["steps"])
    for producer in ("live", "paper_b"):
        root = tmp_path / producer
        native_runs = list((root / "output/manifests/static_train").glob("*.json"))
        assert len(native_runs) == 1
        manifest = read(native_runs[0])
        assert manifest["records"]["target_keys"] == ["small@static"]
        assert manifest["records"]["anchor_date"] == "2024-05-17"
        assert read(root / "config/model_config.json")["current_date"] == "2024-05-10"
    for name, quantity in (("paper_a", 200), ("paper_b", 400)):
        journal = read(tmp_path / name / "data/paper/account.json")
        assert len([r for r in journal["records"] if r["kind"] == "SETTLEMENT"]) == 1
        assert journal["records"][0]["result"]["cycle_id"] == "old_native_id"
        assert journal["records"][0]["result"]["after"]["positions"][0]["quantity"] == quantity
        assert journal["plans"]["new_week"]["trade_date"] == "2024-05-21"
        assert journal["valuations"][-1]["result"]["date"] == "2024-05-17"
        assert [v["result"]["date"] for v in journal["valuations"]] == ["2024-05-13", "2024-05-17"]
    actual = read(tmp_path / "live/config/prod_config.json")
    assert float(actual["current_cash"]) == 3007
    order = next((tmp_path / "live/output/manifests/order_gen").glob("*.json"))
    assert read(order)["records"]["trade_date"] == "2024-05-21"
    assert (project.parent / "cycles/new_week/reports").is_dir()
    before_status = files(tmp_path)
    command(project, "status", parent_env=parent_env)
    assert files(tmp_path) == before_status
    command(project, "resume", parent_env=parent_env)
    assert Path.cwd() == tmp_path
    assert {k: v for k, v in files(tmp_path).items() if not k.endswith("execution.lock")} == {k: v for k, v in before_status.items() if not k.endswith("execution.lock")}


def test_missing_broker_only_waits_actual_and_resume_keeps_frozen_signal(tmp_path):
    project = scenario(tmp_path, with_broker=False)
    result = command(project, code=2)
    progress = rows(result)
    assert progress["live-post"]["status"] == "waiting"
    assert progress["paper_a-new"]["status"] == progress["paper_b-new"]["status"] == "done"
    saved = files(tmp_path / "paper_a")
    for name in ("live", "paper_b"):
        root = tmp_path / name
        (root / "latest_train_records.json").write_text('{"models":{"other@static":"changed-latest"}}')
        (root / "config/ensemble_records.json").write_text('{"combos":{"other":"changed-latest"}}')
    broker(tmp_path / "live")
    result = command(project, "resume")
    assert rows(result)["live-new"]["status"] == "done"
    assert files(tmp_path / "paper_a") == saved
    order = next((tmp_path / "live/output/manifests/order_gen").glob("*.json"))
    assert read(order)["records"]["source"]["record_id"] != "changed-latest"
    assert len(list((tmp_path / "live/output/manifests/static_train").glob("*.json"))) == 1


@pytest.mark.parametrize("step", ["live-post", "shared-predict", "shared-fuse", "paper_a-settle-0", "paper_a-value-2024-05-13", "paper_a-new", "report"])
def test_native_commit_to_project_progress_gap_is_adopted(tmp_path, step):
    project = scenario(tmp_path)
    (project.parent / "interrupt-after").write_text(step)
    command(project, code=130)
    if step in ("shared-predict", "shared-fuse"):
        # Change the mutable pointer before the outer layer has saved its source.
        root = tmp_path / "live"
        (root / "latest_train_records.json").write_text('{"models":{"other@static":"changed-latest"}}')
        (root / "config/ensemble_records.json").write_text('{"combos":{"other":"changed-latest"}}')
    before = {name: files(tmp_path / name) for name in ("live", "paper_a", "paper_b")}
    command(project, "resume")
    # Native financial writes/producer runs already committed at the interruption
    # are never re-invoked. More dependent steps may legitimately have run.
    after = {name: files(tmp_path / name) for name in before}
    for name in before:
        for path, digest in before[name].items():
            if path.startswith(("data/.post_trade_transactions/", "data/training_transactions/")) or path.endswith("old_signal.csv"):
                assert after[name][path] == digest
    for name in ("live", "paper_b"):
        assert len(list((tmp_path / name / "output/manifests/static_train").glob("*.json"))) == 1


def test_native_training_interruption_resumes_original_run(tmp_path):
    project = scenario(tmp_path)
    (tmp_path / "live/interrupt-training").touch()
    command(project, code=130)
    state = read(tmp_path / "live/data/run_state.json")
    run_id = state["run_id"]
    command(project, "resume")
    assert not (tmp_path / "live/data/run_state.json").exists()
    manifests = list((tmp_path / "live/output/manifests/static_train").glob("*.json"))
    assert [p.stem for p in manifests] == [run_id]


def test_candidate_failure_preserves_shared_producer_branch(tmp_path):
    project = scenario(tmp_path)
    (tmp_path / "paper_b/fail-model").touch()
    result = command(project, code=2)
    assert rows(result)["candidate-predict"]["status"] == "failed"
    assert rows(result)["live-new"]["status"] == rows(result)["paper_a-new"]["status"] == "done"
    (tmp_path / "paper_b/fail-model").unlink()
    command(project, "resume")


def test_report_failure_only_retries_derived_report(tmp_path):
    project = scenario(tmp_path)
    (project.parent / "fail-report").touch()
    result = command(project, code=2)
    assert rows(result)["report"]["status"] == "failed"
    assert all(r["status"] == "done" for r in result["steps"] if r["id"] != "report")
    before = {name: files(tmp_path / name) for name in ("live", "paper_a", "paper_b")}
    (project.parent / "fail-report").unlink()
    command(project, "resume")
    assert before == {name: files(tmp_path / name) for name in before}


def test_native_post_trade_interruption_recovers_same_transaction(tmp_path):
    project = scenario(tmp_path)
    (tmp_path / "live/interrupt-post").touch()
    command(project, code=130)
    transaction = next((tmp_path / "live/data/.post_trade_transactions").glob("*/journal.json"))
    original_id = read(transaction)["run_id"]
    command(project, "resume")
    assert read(transaction)["run_id"] == original_id
    assert read(transaction)["status"] == "completed"
    assert float(read(tmp_path / "live/config/prod_config.json")["current_cash"]) == 3007


def test_common_reference_preparation_and_explicit_hold(tmp_path):
    import yaml
    from tests.quantpits.operations.scenario import write_json
    project = scenario(tmp_path)
    config = yaml.safe_load(project.read_text())
    config["prepare_rules"] = "normal.json"
    project.write_text(yaml.safe_dump(config))
    write_json(project.parent / "normal.json", {"reference_assumption": "anchor_close_for_normal_session",
        "source": "synthetic normal session", "boards": {"SH_MAIN": {"normal_limit": True, "limit_rate": "0.1"}},
        "instruments": {"SH600002": {"reference_close": "9.50"}}})
    inputs = project.parent / "inputs.json"
    write_json(inputs, {"hold": ["live"]})
    result = command(project, inputs=inputs)
    assert rows(result)["live-hold"]["status"] == "held"
    rules = read(project.parent / "cycles/new_week/rules.json")
    assert rules["date"] == "2024-05-21"
    assert rules["instruments"]["SH600001"]["reference_close"] == "10.00"
    assert rules["instruments"]["SH600002"]["reference_close"] == "9.50"
    for name in ("paper_a", "paper_b"):
        assert read(tmp_path / name / "data/paper/account.json")["plans"]["new_week"]["market_rules"] == rules
    assert float(read(tmp_path / "live/config/prod_config.json")["current_cash"]) == 3007


def test_bundle_input_uses_native_workspace_relative_contract(tmp_path):
    from tests.quantpits.operations.scenario import write_json
    project = scenario(tmp_path)
    actual = tmp_path / "live"
    original = actual / "data/2024-05-13-table.xlsx"
    original.rename(actual / "data/2024-05-13-2024-05-17-table.xlsx")
    (actual / "data/2024-05-17-table.xlsx").unlink()
    inputs = project.parent / "inputs.json"
    write_json(inputs, {"actual": {"live": {"settlement_bundle": "data/2024-05-13-2024-05-17-table.xlsx", "allow_missing_settlement": True}}})
    result = command(project, inputs=inputs)
    assert rows(result)["live-post"]["status"] == "done"


WEEK = ["2024-05-10", "2024-05-13", "2024-05-14", "2024-05-15", "2024-05-16", "2024-05-17", "2024-05-20", "2024-05-21"]


@pytest.mark.parametrize("holiday,hold", [(False, False), (True, True)])
def test_daily_closes_use_pre_execution_and_post_execution_positions(tmp_path, holiday, hold):
    from tests.quantpits.operations.scenario import write_json
    sessions = [d for d in WEEK if not (holiday and d == "2024-05-15")]
    project = scenario(tmp_path, sessions=sessions, pending_trade_date="2024-05-14",
                       initial_holding=[{"instrument": "SH600002", "value": 100, "amount": 1000}])
    for name in ("paper_a", "paper_b"):
        old_plan = read(tmp_path / name / "data/paper/account.json")["plans"]["old_native_id"]
        assert [p["instrument"] for p in old_plan["sells"]] == ["SH600002"]
        assert [p["instrument"] for p in old_plan["buys"]] == ["SH600001"]
    inputs = project.parent / "inputs.json"
    write_json(inputs, {"hold": ["paper_a"] if hold else []})
    preview = command(project, "plan", inputs=inputs, parent_env="missing")
    paper = [s for s in preview["steps"] if s.get("member") == "paper_a"]
    assert [(s["kind"], s.get("date")) for s in paper[:3]] == [
        ("value", "2024-05-13"), ("settle", None), ("value", "2024-05-14")]
    result = command(project, inputs=inputs, parent_env="missing")
    for name in ("paper_a", "paper_b"):
        journal = read(tmp_path / name / "data/paper/account.json")
        values = [v["result"] for v in journal["valuations"]]
        assert [v["date"] for v in values] == sessions[1:-2]
        assert [(p["instrument"], p["quantity"]) for p in values[0]["positions"]] == [("SH600002", 100)]
        assert [p["instrument"] for p in values[1]["positions"]] == ["SH600001"]
        positions = [(p["instrument"], p["quantity"]) for p in values[1]["positions"]]
        assert all([(p["instrument"], p["quantity"]) for p in v["positions"]] == positions for v in values[2:])
        assert journal["records"][0]["result"]["cycle_id"] == "old_native_id"
    if hold:
        assert rows(result)["paper_a-hold"]["status"] == "held"
        assert set(read(tmp_path / "paper_a/data/paper/account.json")["plans"]) == {"old_native_id"}


def test_daily_process_failure_waits_before_advancing_and_resume_preserves_completed_dates(tmp_path):
    project = scenario(tmp_path, sessions=WEEK)
    marker = project.parent / "fail-value"
    marker.write_text("paper_a-value-2024-05-14")
    result = command(project, code=2, parent_env="missing")
    entry = rows(result)["paper_a-value-2024-05-14"]
    assert entry["status"] == "failed"
    assert rows(result)["paper_a-new"]["status"] == "waiting"
    journal = read(tmp_path / "paper_a/data/paper/account.json")
    first = journal["valuations"][0]
    assert first["result"]["date"] == "2024-05-13"
    assert all(v["result"]["date"] <= "2024-05-14" for v in journal["valuations"])
    assert rows(result)["paper_b-new"]["status"] == "done"
    marker.unlink()
    before_log = (project.parent / "cycles/new_week/logs/paper_a-value-2024-05-13.log").read_bytes()
    resumed = command(project, "resume", parent_env="missing")
    journal = read(tmp_path / "paper_a/data/paper/account.json")
    assert journal["valuations"][0] == first
    assert [v["result"]["date"] for v in journal["valuations"] if v["result"]["status"] == "COMPLETE"] == WEEK[1:-2]
    assert (project.parent / "cycles/new_week/logs/paper_a-value-2024-05-13.log").read_bytes() == before_log
    assert "exit_code" not in rows(resumed)["paper_a-value-2024-05-14"]


def missing_close(tmp_path, selected):
    import struct
    price = tmp_path / "provider/features/sh600001/close.day.bin"
    raw = price.read_bytes()
    words = list(struct.unpack("<%sf" % (len(raw) // 4), raw))
    words[1 + WEEK.index(selected)] = float("nan")
    price.write_bytes(struct.pack("<%sf" % len(words), *words))
    return price, price.read_bytes()


@pytest.mark.parametrize("legacy_waiting", [False, True])
def test_saved_stale_valuation_advances_without_repairing_history(tmp_path, legacy_waiting):
    from tests.quantpits.operations.scenario import write_json
    project = scenario(tmp_path, sessions=WEEK)
    price, missing = missing_close(tmp_path, "2024-05-14")
    step = "paper_a-value-2024-05-14"
    if legacy_waiting:
        # Commit the real native valuation, then retain the previous candidate's
        # waiting metadata. No native result or completion proof is fabricated.
        (project.parent / "interrupt-after").write_text(step)
        command(project, code=130, parent_env="missing")
        progress_path = project.parent / "cycles/new_week/progress.json"
        progress = read(progress_path)
        progress["results"][step].update(status="waiting", reason="native valuation at 2024-05-14 is STALE", exit_code=0)
        write_json(progress_path, progress)
        saved = read(tmp_path / "paper_a/data/paper/account.json")["valuations"]
        log = project.parent / ("cycles/new_week/logs/" + step + ".log")
        old_log = log.read_bytes()
        before = files(tmp_path)
        waiting = command(project, "status", code=2, parent_env="missing")
        assert files(tmp_path) == before
        assert rows(waiting)[step]["valuation"]["status"] == "STALE"
        result = command(project, "resume", parent_env="missing")
        assert read(progress_path)["steps_identity"] == progress["steps_identity"]
        assert read(tmp_path / "paper_a/data/paper/account.json")["valuations"][:2] == saved
        assert log.read_bytes() == old_log
        assert "exit_code" not in rows(result)[step]
    else:
        result = command(project, parent_env="missing")
    for name in ("paper_a", "paper_b"):
        journal = read(tmp_path / name / "data/paper/account.json")
        values = [v["result"] for v in journal["valuations"]]
        assert [v["date"] for v in values] == WEEK[1:-2]
        assert [v["status"] for v in values] == ["COMPLETE", "STALE", "COMPLETE", "COMPLETE", "COMPLETE"]
        stale = values[1]
        assert stale["nav"] is not None and stale["missing"] == []
        assert stale["positions"][0]["price"] == "10.00"
        assert stale["positions"][0]["price_date"] == "2024-05-13"
        assert stale["positions"][0]["stale"] is True
        assert rows(result)[name + "-value-2024-05-14"]["valuation"] == stale
        assert rows(result)[name + "-new"]["status"] == "done"
        assert journal["plans"]["new_week"]["trade_date"] == "2024-05-21"
    before = files(tmp_path)
    inspected = command(project, "status", parent_env="missing")
    assert files(tmp_path) == before
    assert rows(inspected)[step]["valuation"] == rows(result)[step]["valuation"]
    command(project, "resume", parent_env="missing")
    assert {k: v for k, v in files(tmp_path).items() if not k.endswith("execution.lock")} == {
        k: v for k, v in before.items() if not k.endswith("execution.lock")}
    assert price.read_bytes() == missing


def test_new_cycle_starts_after_saved_stale_frontier(tmp_path):
    from tests.quantpits.operations.scenario import write_json
    project = scenario(tmp_path, sessions=WEEK)
    price, missing = missing_close(tmp_path, "2024-05-14")
    inputs = project.parent / "hold.json"
    write_json(inputs, {"hold": ["live", "paper_a", "paper_b"]})
    first = command(project, code=2, cycle="stale_week", anchor="2024-05-14", value_through="2024-05-14",
                    trade_date="2024-05-15", inputs=inputs, parent_env="missing")
    # The original comparison retains partial quality at a STALE endpoint;
    # both account date steps have nevertheless been processed.
    assert rows(first)["report"]["status"] == "waiting"
    assert rows(first)["paper_a-value"]["status"] == "done"
    assert rows(first)["paper_a-value"]["valuation"]["status"] == "STALE"
    saved = {name: read(tmp_path / name / "data/paper/account.json")["valuations"] for name in ("paper_a", "paper_b")}
    preview = command(project, "plan", cycle="next_week", parent_env="missing")
    for name in saved:
        assert [s["date"] for s in preview["steps"] if s.get("member") == name and s["kind"] == "value"] == WEEK[3:-2]
    command(project, cycle="next_week", parent_env="missing")
    for name, values in saved.items():
        assert read(tmp_path / name / "data/paper/account.json")["valuations"][:2] == values
    before = files(tmp_path)
    inspected = command(project, "status", code=2, cycle="stale_week", parent_env="missing")
    assert rows(inspected)["paper_a-value"]["valuation"]["status"] == "STALE"
    assert files(tmp_path) == before
    assert price.read_bytes() == missing


def test_unknown_nav_stays_partial_and_waits(tmp_path):
    project = scenario(tmp_path, sessions=WEEK)
    price, missing = missing_close(tmp_path, "2024-05-13")
    result = command(project, code=2, parent_env="missing")
    for name in ("paper_a", "paper_b"):
        value = rows(result)[name + "-value-2024-05-13"]
        assert value["status"] == "waiting"
        assert value["valuation"]["status"] == "PARTIAL"
        assert value["valuation"]["nav"] is None
        assert value["valuation"]["positions"][0]["price"] is None
        assert value["valuation"]["missing"] == ["SH600001"]
        journal = read(tmp_path / name / "data/paper/account.json")
        assert len(journal["valuations"]) == 1
        assert "new_week" not in journal["plans"]
    before = files(tmp_path)
    inspected = command(project, "status", code=2, parent_env="missing")
    assert rows(inspected)["paper_a-value-2024-05-13"]["valuation"]["status"] == "PARTIAL"
    assert files(tmp_path) == before
    assert price.read_bytes() == missing


def test_actual_calendar_waits_before_any_stale_observation(tmp_path):
    project = scenario(tmp_path, sessions=WEEK)
    price, missing = missing_close(tmp_path, "2024-05-14")
    calendar = tmp_path / "provider/calendars/day.txt"
    calendar.write_text("\n".join(WEEK[:2]) + "\n")
    result = command(project, code=2, parent_env="missing")
    for name in ("paper_a", "paper_b"):
        entry = rows(result)[name + "-value-2024-05-14"]
        assert entry["status"] == "waiting" and "WAITING_FOR_DATA" in entry["reason"]
        assert "valuation" not in entry
        journal = read(tmp_path / name / "data/paper/account.json")
        assert [v["result"]["date"] for v in journal["valuations"]] == ["2024-05-13"]
    calendar.write_text("\n".join(WEEK[:-2]) + "\n")
    resumed = command(project, "resume", parent_env="missing")
    assert rows(resumed)["paper_a-value-2024-05-14"]["valuation"]["status"] == "STALE"
    assert price.read_bytes() == missing


def test_saved_valuation_frontier_preserves_older_history_gaps(tmp_path):
    from quantpits.operations.config import load_project
    from quantpits.research.paper_workflow import PaperWorkflow
    project = scenario(tmp_path, sessions=WEEK)
    for member in load_project(project).group.members:
        if member.kind == "paper":
            workflow = PaperWorkflow(member.ctx)
            workflow.settle("old_native_id")
            workflow.value("2024-05-16")  # an existing historic gap is outside this cycle's advancement
    command(project, parent_env="missing")
    for name in ("paper_a", "paper_b"):
        assert [v["result"]["date"] for v in read(tmp_path / name / "data/paper/account.json")["valuations"]] == ["2024-05-16", "2024-05-17"]
