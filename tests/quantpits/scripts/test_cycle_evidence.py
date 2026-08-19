import json
import os
import subprocess
import sys
from pathlib import Path

from quantpits.scripts import cycle_evidence
from tests.quantpits.evidence.test_sealing import cycle_factory


def _args(root, qlib, request, *, dry=False):
    values = [
        "capture", "--workspace", str(root), "--qlib-data-dir", str(qlib),
        "--cycle-id", request.cycle_id,
        "--research-epoch-id", request.research_epoch_id,
        "--post-trade-manifest", request.post_trade_manifest,
        "--prediction-manifest", request.prediction_manifest,
        "--ensemble-manifest", request.ensemble_manifest,
        "--order-manifest", request.order_manifest,
    ]
    if request.deep_analysis_run:
        values.extend(["--deep-analysis-run", request.deep_analysis_run])
    if request.decision_event:
        values.extend(["--decision-event", request.decision_event])
    if dry:
        values.append("--dry-run")
    return values


def test_cli_capture_prints_machine_result_and_preserves_cwd(cycle_factory, capsys):
    root, qlib, request = cycle_factory()
    before = Path.cwd()
    assert cycle_evidence.main(_args(root, qlib, request)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "sealed_complete"
    assert payload["did_write"] is True
    assert Path.cwd() == before


def test_cli_dry_run_creates_no_evidence_namespace(cycle_factory, capsys):
    root, qlib, request = cycle_factory()
    assert cycle_evidence.main(_args(root, qlib, request, dry=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "sealed_complete"
    assert payload["did_write"] is False
    assert not (root / "data/evidence").exists()


def test_cli_invalid_epoch_is_blocked_before_write(cycle_factory, capsys):
    root, qlib, request = cycle_factory()
    args = _args(root, qlib, request)
    args[args.index(request.research_epoch_id)] = "not-strict"
    assert cycle_evidence.main(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert not (root / "data/evidence").exists()


def test_import_has_zero_cwd_environment_or_workspace_mutation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment.update({
        "QLIB_WORKSPACE_DIR": str(workspace),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(repository),
    })
    script = (
        "import os; from pathlib import Path; "
        "before=(Path.cwd(),dict(os.environ),tuple(Path(os.environ['QLIB_WORKSPACE_DIR']).iterdir())); "
        "import quantpits.scripts.cycle_evidence; "
        "after=(Path.cwd(),dict(os.environ),tuple(Path(os.environ['QLIB_WORKSPACE_DIR']).iterdir())); "
        "assert before == after"
    )
    subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=environment, check=True)
