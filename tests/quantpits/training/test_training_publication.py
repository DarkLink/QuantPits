import json

import pytest

from quantpits.training.command import TrainingRunOptions, prepare_training_run
from quantpits.training.publication import TrainingPublicationCoordinator
from quantpits.training.records import ModelRecordEntry
from quantpits.training.resolved import resolve_training_run
from quantpits.training.runners import TrainingTargetResult
from quantpits.utils.workspace import WorkspaceContext


def _workspace(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "data").mkdir()
    (root / "config/model_registry.yaml").write_text(
        "models:\n  demo:\n    enabled: true\n    yaml_file: demo.yaml\n"
    )
    (root / "config/model_config.json").write_text('{"freq":"week"}')
    (root / "config/demo.yaml").write_text("model: {}\n")
    return root


def _run_and_result(root):
    prepared = prepare_training_run(
        ctx=WorkspaceContext.from_root(root),
        options=TrainingRunOptions(family="static", action="full", run_id="publication-run"),
    )
    run = resolve_training_run(prepared, {"anchor_date": "2026-07-10", "test_end_time": "2026-07-10"})
    entry = ModelRecordEntry(
        "demo@static", "demo", "static", "train", "ready", "rid", "Prod_Train_WEEK",
        requested_anchor="2026-07-10", prediction_start="2026-07-10",
        prediction_end="2026-07-10", prediction_rows=1,
    )
    return run, TrainingTargetResult("demo@static", "train", "success", entry, {"IC_Mean": 0.1})


def test_publication_commits_record_last_and_writes_verified_receipt(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(run, clock=lambda: __import__("datetime").datetime(2026, 7, 14))
    intent = coordinator.prepare((result,))
    assert [item.kind for item in sorted(intent.members, key=lambda item: item.commit_order)][-1] == "record"
    receipt = coordinator.commit(intent)
    assert receipt.status == "committed"
    assert json.loads((root / "latest_train_records.json").read_text())["models"] == {"demo@static": "rid"}
    assert coordinator.recover().published_keys == ("demo@static",)


def test_publication_rolls_forward_known_mixed_state_without_target_execution(tmp_path):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    coordinator = TrainingPublicationCoordinator(run, clock=lambda: __import__("datetime").datetime(2026, 7, 14))
    intent = coordinator.prepare((result,))
    performance = next(item for item in intent.members if item.kind == "performance")
    target = root / performance.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((coordinator.directory / performance.staged_relative_path).read_bytes())
    receipt = coordinator.recover()
    assert receipt.recovery_action == "roll_forward"
    assert (root / "latest_train_records.json").is_file()


@pytest.mark.parametrize(
    "point,member",
    [
        ("after_member_replace", "output/model_performance_2026-07-10.json"),
        ("after_member_replace", "latest_train_records.json"),
        ("before_receipt_write", None),
    ],
)
def test_publication_recovers_each_current_output_boundary(tmp_path, point, member):
    root = _workspace(tmp_path)
    run, result = _run_and_result(root)
    fired = []

    def fail(actual_point, actual_member=None):
        if not fired and actual_point == point and (member is None or actual_member == member):
            fired.append(True)
            raise RuntimeError("injected publication interruption")

    interrupted = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14), fault_hook=fail,
    )
    intent = interrupted.prepare((result,))
    with pytest.raises(RuntimeError, match="injected"):
        interrupted.commit(intent)
    recovered = TrainingPublicationCoordinator(
        run, clock=lambda: __import__("datetime").datetime(2026, 7, 14)
    ).recover()
    assert recovered.status == "committed"
    assert json.loads((root / "latest_train_records.json").read_text())["models"] == {
        "demo@static": "rid"
    }
