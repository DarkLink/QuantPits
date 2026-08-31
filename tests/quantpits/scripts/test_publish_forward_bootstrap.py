import json
import os
import sys

import pytest

from tests.quantpits.research.test_forward_bootstrap import bootstrap_workspace
from tests.quantpits.research.test_forward_definition_evidence import evidence_workspace
from tests.quantpits.research.test_forward_observation import CYCLE as DEFINITION_CYCLE
from tests.quantpits.research.test_forward_portfolio_source import CYCLE as SOURCE_CYCLE
from quantpits.scripts import publish_forward_bootstrap as cli


def _args(value):
    root, activation, definitions, evidence, bootstraps = value
    return [
        "preflight", "--workspace", str(root),
        "--definition-evidence-cycle", DEFINITION_CYCLE,
        "--bootstrap-source-cycle", SOURCE_CYCLE,
        "--activation", str(activation), "--definition-store", str(definitions),
        "--evidence-store", str(evidence), "--bootstrap-store", str(bootstraps),
    ]


def test_cli_requires_explicit_action_and_all_physical_inputs(capsys):
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["error"]["reason_code"] == "ARGUMENT_INVALID"


def test_cli_preflight_is_machine_readable_private_safe_and_zero_write(
    bootstrap_workspace, capsys,
):
    before = sorted(path.relative_to(bootstrap_workspace[0]).as_posix()
                    for path in bootstrap_workspace[0].rglob("*"))
    assert cli.main(_args(bootstrap_workspace)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PREPARED"
    rendered = json.dumps(summary)
    assert "-1.25" not in rendered and "SH600001" not in rendered
    assert str(bootstrap_workspace[0]) not in rendered
    assert before == sorted(path.relative_to(bootstrap_workspace[0]).as_posix()
                            for path in bootstrap_workspace[0].rglob("*"))


def test_cli_publish_requires_exact_digest_and_authorization(bootstrap_workspace, capsys):
    args = _args(bootstrap_workspace)
    assert cli.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    publish = ["publish", *args[1:], "--expected-bootstrap-set-id", plan["bootstrap_set_id"],
               "--expected-bootstrap-request-digest", json.dumps(
                   plan["bootstrap_request_digest"], sort_keys=True, separators=(",", ":")),
               "--authorization-action", "WRONG"]
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    publish[-1] = "PUBLISH_ONE_MATCHED_FORWARD_BOOTSTRAP_V1"
    assert cli.main(publish) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "COMMITTED"
    assert cli.main(publish) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ADOPTED"


def test_cli_unknown_arguments_never_echo_private_tokens(capsys):
    secret = "PRIVATE_BOOTSTRAP_TOKEN_SH600001_-1.25"
    assert cli.main(["preflight", "--workspace", secret, "--unknown", secret]) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and secret not in captured.out


def test_cli_help_import_preserve_environment_cwd_and_optional_modules(capsys):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "preflight" in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cli_process_control_propagates(monkeypatch, exception):
    import quantpits.research.forward_bootstrap as module
    monkeypatch.setattr(module, "prepare_matched_forward_bootstrap",
                        lambda *_: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception):
        cli.main(["preflight", "--workspace", "/private/workspace",
                  "--definition-evidence-cycle", DEFINITION_CYCLE,
                  "--bootstrap-source-cycle", SOURCE_CYCLE,
                  "--activation", "/private/activation", "--definition-store", "/private/definitions",
                  "--evidence-store", "/private/evidence", "--bootstrap-store", "/private/bootstraps"])
