import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from quantpits.ensemble.command import (
    EnsembleCommandDependencies,
    EnsembleCommandRequest,
    EnsembleCommandUsageError,
    build_ensemble_arg_parser,
    run_ensemble_command,
)
from quantpits.ensemble.types import EnsembleRunConfig, EnsembleRunSummary
from quantpits.utils.ensemble_plan import EnsemblePlanError
from quantpits.utils.workspace import WorkspaceContext


def _args(*argv):
    return build_ensemble_arg_parser().parse_args(argv)


def _dependencies(tmp_path, summary=None):
    root = tmp_path / "Demo_Workspace"
    root.mkdir()
    ctx = WorkspaceContext.from_root(root)
    service = Mock()
    service.execute.return_value = summary or EnsembleRunSummary(
        run_id="run-1",
        anchor_date="2026-01-01",
        experiment_name="demo",
        combo_results=(),
        manifest_path=None,
    )
    deps = EnsembleCommandDependencies(
        get_workspace_context=Mock(return_value=ctx),
        load_run_config=Mock(return_value=EnsembleRunConfig({}, {}, {})),
        safeguard=Mock(),
        service_factory=Mock(return_value=service),
    )
    return deps, ctx, service


def _prepared(args, ctx):
    options = SimpleNamespace(
        json_plan=args.json_plan,
        explain_plan=args.explain_plan or args.json_plan,
    )
    return SimpleNamespace(options=options, ctx=ctx, plan_fingerprint="abc123")


def test_missing_selector_fails_before_dependencies(tmp_path):
    deps, _, _ = _dependencies(tmp_path)

    with pytest.raises(EnsembleCommandUsageError, match="必须指定"):
        run_ensemble_command(EnsembleCommandRequest(_args(), ()), deps)

    deps.get_workspace_context.assert_not_called()
    deps.load_run_config.assert_not_called()
    deps.safeguard.assert_not_called()
    deps.service_factory.assert_not_called()


def test_json_plan_is_immutable_dry_run(monkeypatch, tmp_path):
    deps, ctx, _ = _dependencies(tmp_path)
    args = _args("--from-config", "--json-plan", "--run-id", "json-run")
    prepared = _prepared(args, ctx)
    seen = {}

    def prepare(**kwargs):
        seen.update(kwargs)
        return prepared

    monkeypatch.setattr("quantpits.ensemble.command.prepare_ensemble_run", prepare)
    monkeypatch.setattr(
        "quantpits.ensemble.command.prepared_plan_json",
        lambda value: {"schema_version": 1, "plan_fingerprint": value.plan_fingerprint},
    )

    outcome = run_ensemble_command(
        EnsembleCommandRequest(args=args, cli_args=("--from-config", "--json-plan")),
        deps,
    )

    assert outcome.mode == "json-plan"
    assert json.loads(outcome.rendered_output)["plan_fingerprint"] == "abc123"
    assert outcome.summary is None
    assert args.explain_plan is False
    assert seen["options"].explain_plan is True
    assert seen["ctx"] is ctx
    assert seen["cli_args"] == ("--from-config", "--json-plan")
    deps.load_run_config.assert_called_once_with(ctx, "latest_train_records.json")
    deps.safeguard.assert_not_called()
    deps.service_factory.assert_not_called()


def test_explain_plan_does_not_create_service(monkeypatch, tmp_path):
    deps, ctx, _ = _dependencies(tmp_path)
    args = _args("--from-config", "--explain-plan")
    prepared = _prepared(args, ctx)
    monkeypatch.setattr("quantpits.ensemble.command.prepare_ensemble_run", lambda **kwargs: prepared)
    monkeypatch.setattr("quantpits.ensemble.command.render_prepared_plan", lambda value: "PLAN abc123")

    outcome = run_ensemble_command(EnsembleCommandRequest(args, ("--from-config", "--explain-plan")), deps)

    assert outcome.mode == "explain-plan"
    assert outcome.rendered_output == "PLAN abc123"
    assert outcome.summary is None
    deps.safeguard.assert_not_called()
    deps.service_factory.assert_not_called()


def test_execute_calls_safeguard_and_service_once(monkeypatch, tmp_path):
    summary = EnsembleRunSummary("run-1", "2026-01-01", "demo", (), None)
    deps, ctx, service = _dependencies(tmp_path, summary=summary)
    args = _args("--from-config")
    prepared = _prepared(args, ctx)
    monkeypatch.setattr("quantpits.ensemble.command.prepare_ensemble_run", lambda **kwargs: prepared)

    outcome = run_ensemble_command(EnsembleCommandRequest(args, ("--from-config",)), deps)

    assert outcome.mode == "execute"
    assert outcome.summary is summary
    assert outcome.rendered_output is None
    deps.safeguard.assert_called_once_with("Ensemble Fusion")
    deps.service_factory.assert_called_once_with()
    service.execute.assert_called_once_with(prepared)


def test_plan_error_propagates_without_execution(monkeypatch, tmp_path):
    deps, _, _ = _dependencies(tmp_path)
    args = _args("--from-config")
    monkeypatch.setattr(
        "quantpits.ensemble.command.prepare_ensemble_run",
        lambda **kwargs: (_ for _ in ()).throw(EnsemblePlanError("unknown combo")),
    )

    with pytest.raises(EnsemblePlanError, match="unknown combo"):
        run_ensemble_command(EnsembleCommandRequest(args, ("--from-config",)), deps)

    deps.safeguard.assert_not_called()
    deps.service_factory.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ("--models", "m1,m2", "--method", "manual", "--weights", "m1:0.5,m2:0.5"),
        ("--from-config",),
        ("--from-config-all",),
        ("--combo", "combo_A"),
        ("--from-config", "--json-plan"),
        ("--from-config", "--no-backtest", "--no-charts", "--no-manifest"),
        ("--from-config", "--start-date", "2026-01-01", "--end-date", "2026-06-30"),
    ],
)
def test_engine_parser_matches_script_wrapper(argv):
    from quantpits.scripts import ensemble_fusion

    engine_args = vars(build_ensemble_arg_parser().parse_args(argv))
    script_args = vars(ensemble_fusion.build_arg_parser().parse_args(argv))

    assert engine_args == script_args
