"""Real cycle CLI → native services → three independent temporary accounts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.quantpits.operations.scenario import broker, scenario


def command(project, action="run", code=0, *, cycle="new_week", inputs=None):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", QLIB_WORKSPACE_DIR="/must/not/use/inherited",
                       MLFLOW_TRACKING_URI="sqlite:////must/not/use/inherited.db")
    arguments = [action, "--project", str(project), "--cycle", cycle]
    if action in ("run", "plan"):
        arguments += ["--anchor", "2024-05-17", "--value-through", "2024-05-17", "--trade-date", "2024-05-21", "--since", "2024-05-10"]
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


def test_three_workspace_cycle_and_readonly_cli(tmp_path, monkeypatch):
    project = scenario(tmp_path)
    # Full-suite collection can import legacy scripts that change cwd. The
    # subprocess worker must remain importable outside the repository as well.
    monkeypatch.chdir(tmp_path)
    before = files(tmp_path)
    preview = command(project, "plan")
    assert files(tmp_path) == before
    assert rows(preview)["shared-fuse"]["source"] == "pending_generation"
    assert rows(preview)["paper_a-settle-0"]["native_cycle"] == "old_native_id"
    result = command(project)
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
    actual = read(tmp_path / "live/config/prod_config.json")
    assert float(actual["current_cash"]) == 3007
    order = next((tmp_path / "live/output/manifests/order_gen").glob("*.json"))
    assert read(order)["records"]["trade_date"] == "2024-05-21"
    assert (project.parent / "cycles/new_week/reports").is_dir()
    before_status = files(tmp_path)
    command(project, "status")
    assert files(tmp_path) == before_status
    command(project, "resume")
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


@pytest.mark.parametrize("step", ["live-post", "shared-predict", "shared-fuse", "paper_a-settle-0", "paper_a-new", "report"])
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
