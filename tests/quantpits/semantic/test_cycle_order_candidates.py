"""Actual project order invocation exports once and feeds the independent preparer."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from quantpits.operations.config import load_project
from quantpits.operations.cycle import status
from tests.quantpits.operations.scenario import scenario, write_json
from tests.quantpits.semantic.test_project_cycle import command, files, read, rows


def selection(project):
    path = project.parent / "inputs.json"
    write_json(path,{"order_candidates":["live"]})
    return path


def candidate(project,result):
    step=rows(result)["live-new"]
    root=load_project(project).members["live"].ctx.root
    path=root/step["candidate_output"]
    return root,path,read(path)


def test_selected_cycle_completes_without_future_open_and_candidates_feed_preparer(tmp_path):
    project=scenario(tmp_path)
    # Future O is unavailable. Current/pending paper observations remain intact.
    for path in (tmp_path/"provider/features").glob("*/open.day.bin"):
        path.write_bytes(path.read_bytes()[:16])
    inputs=selection(project)
    result=command(project,inputs=inputs)
    assert result["status"] == "complete"
    assert [step["id"] for step in result["steps"] if "candidate_output" in step] == ["live-new"]
    root,path,payload=candidate(project,result)
    assert [row["instrument"] for row in payload["buys"]] == ["SH600001"]
    assert payload["source"]["signal_digest"] == read(load_project(project).signal_dir("new_week","shared")/"ensemble.json")["predictions"]["chosen"]["sha256"]
    evidence=rows(result)["live-new"]["evidence"]
    assert evidence["candidates"]["count"] == 1 and evidence["candidates"]["path"] == str(path)
    manifest=read(Path(evidence["manifest"]))
    assert manifest["records"]["candidates"]["count"] == len(payload["buys"])
    assert any(p["path"].endswith(".csv") for p in manifest["outputs"])
    actual=read(root/"config/prod_config.json")
    observation=dict(schema_version=1,kind="cash_open_observation",trade_date="2024-05-21",price_basis="CASH_OPEN",
        cash=str(actual["current_cash"]),holdings=[dict(instrument=p["instrument"],quantity=int(p["value"])) for p in actual["current_holding"]],
        quotes={row["instrument"]:dict(open="9.60",previous_close="10",factor="1",previous_factor="1") for row in payload["buys"]},
        market_rules=dict(date="2024-05-21",source="synthetic explicit observation",boards={"SH_MAIN":dict(normal_limit=True,limit_rate=".10")},instruments={}),
        execution=dict(base_fee="5",base_rate="0",sell_tax_rate=".0005",transfer_rates=dict(SH=".00001",SZ="0"),lot_size=100))
    write_json(root/"data/current_open.json",observation)
    before=files(tmp_path)
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
    for key in ("QLIB_WORKSPACE_DIR","MLFLOW_TRACKING_URI"): environment.pop(key,None)
    prepared=subprocess.run([sys.executable,"-m","quantpits.scripts.prepare_orders","--workspace",str(root),
        "--candidates",str(path),"--observation","data/current_open.json","--preview"],
        env=environment,cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True,timeout=30)
    assert prepared.returncode == 0,prepared.stderr
    output=json.loads(prepared.stdout)
    assert output["decisions"][0]["quantity"] == 300 and output["decisions"][0]["status"] == "PREPARED"
    assert output["estimated_purchase_cost"] == "2885.03"
    assert files(tmp_path) == before
    assert command(project,"status")["status"] == "complete"
    assert files(tmp_path) == before
    # Completion status must detect damaged candidate bytes without writes.
    path.write_text(path.read_text()+" ")
    before=files(tmp_path)
    with pytest.raises(ValueError,match="artifact changed"):
        status(load_project(project),"new_week")
    assert files(tmp_path) == before


def test_order_native_commit_gap_adopts_saved_selection_without_second_order(tmp_path):
    project=scenario(tmp_path)
    inputs=selection(project)
    (project.parent/"interrupt-after").write_text("live-new")
    command(project,inputs=inputs,code=130)
    progress=read(project.parent/"cycles/new_week/progress.json")
    assert progress["results"]["live-new"]["status"] == "interrupted"
    step=next(s for s in progress["steps"] if s["id"] == "live-new")
    root=tmp_path/"live"
    artifacts=list((root/Path(step["candidate_output"]).parent).glob("*"))
    artifacts += [root/"output/manifests/order_gen"/(step["run_id"]+".json"),
        root/"data/operator_log.jsonl",project.parent/"cycles/new_week/logs/live-new.log"]
    saved={path:path.read_bytes() for path in artifacts}
    inputs.write_text("{external inputs are now invalid and must not be read")
    result=command(project,"resume")
    assert result["request"]["inputs"]["order_candidates"] == ["live"]
    assert rows(result)["live-new"]["evidence"]["candidates"]["count"] == 1
    assert all(path.read_bytes() == raw for path,raw in saved.items())
    assert command(project,"status")["status"] == "complete"


def test_bad_candidate_after_native_success_fails_actual_without_blocking_paper(tmp_path):
    project=scenario(tmp_path)
    inputs=selection(project)
    (project.parent/"damage-order-candidates").write_text("damage actual output")
    result=command(project,inputs=inputs,code=2)
    assert rows(result)["live-new"]["status"] == "failed"
    assert "digest mismatch" in rows(result)["live-new"]["reason"]
    assert rows(result)["paper_a-new"]["status"] == rows(result)["paper_b-new"]["status"] == "done"
    assert rows(result)["shared-predict"]["status"] == rows(result)["shared-fuse"]["status"] == "done"
