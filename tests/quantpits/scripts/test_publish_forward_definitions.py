import json
import os
import sys

import pytest

from tests.quantpits.research.test_forward_observation import CYCLE, _bundle
from quantpits.scripts import publish_forward_definitions as cli


@pytest.fixture
def cli_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    activation, _ = _bundle(root, frozen=True)
    store = root / "research" / "shadow_v1" / "definitions"
    store.mkdir(mode=0o700)
    return root, activation, store


def _preflight_args(value):
    root, activation, store = value
    return [
        "preflight", "--workspace", str(root), "--evidence-cycle", CYCLE,
        "--activation", str(activation), "--definition-store", str(store),
    ]


def test_cli_requires_explicit_preflight_or_publish_and_all_physical_inputs(capsys):
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "blocked",
        "error": {
            "type": "FrozenDefinitionPublicationArgumentError",
            "reason_code": "ARGUMENT_INVALID",
        },
    }


def test_cli_preflight_is_safe_machine_readable_and_zero_write(cli_workspace, capsys):
    before = sorted(path.relative_to(cli_workspace[0]).as_posix() for path in cli_workspace[0].rglob("*"))
    assert cli.main(_preflight_args(cli_workspace)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PREPARED"
    assert summary["publication_capability"] is False
    assert sorted(path.relative_to(cli_workspace[0]).as_posix() for path in cli_workspace[0].rglob("*")) == before


def test_cli_publish_requires_exact_digest_and_authorization_literal(cli_workspace, capsys):
    args = _preflight_args(cli_workspace)
    assert cli.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    publish = [
        "publish", *args[1:],
        "--expected-definition-set-id", plan["definition_set_id"],
        "--expected-request-digest", json.dumps(plan["request_digest"], sort_keys=True, separators=(",", ":")),
        "--authorization-action", "WRONG",
    ]
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    publish[-3] = json.dumps(plan["request_digest"])
    publish[-1] = "PUBLISH_ONE_FROZEN_DEFINITION_BUNDLE_V1"
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["error"]["reason_code"] == "ARGUMENT_INVALID"
    publish[-3] = json.dumps(plan["request_digest"], sort_keys=True, separators=(",", ":"))
    assert cli.main(publish) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "COMMITTED"
    assert cli.main(publish) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["status"] == "ADOPTED" and replay["did_write"] is False


def test_cli_unknown_or_invalid_arguments_never_echo_private_values(capsys):
    private = "PRIVATE_PATH_TOKEN_MODEL_A_0.001"
    assert cli.main(["preflight", "--workspace", private, "--unknown", private]) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and private not in captured.out


@pytest.mark.parametrize("status,code", [
    ("COMMITTED", 0), ("ADOPTED", 0), ("CONFLICT", 3), ("UNCERTAIN", 4),
])
def test_cli_commit_adopt_conflict_and_uncertain_map_to_stable_exit_codes(monkeypatch, capsys, status, code):
    class Result:
        def to_safe_summary_dict(self):
            return {"status": status, "definition_bytes_durable": status in {"COMMITTED", "ADOPTED"}}

    import quantpits.research.forward_definition_publication as publication
    monkeypatch.setattr(publication, "publish_frozen_shadow_forward_definition_bundle", lambda *_args: Result())
    digest = {"algorithm": "sha256", "domain": "canonical_json", "value": "0" * 64, "size_bytes": 1}
    assert cli.main([
        "publish", "--workspace", "/private/workspace", "--evidence-cycle", CYCLE,
        "--activation", "/private/activation", "--definition-store", "/private/store",
        "--expected-definition-set-id", "definition.id",
        "--expected-request-digest", json.dumps(digest, sort_keys=True, separators=(",", ":")),
        "--authorization-action", "PUBLISH_ONE_FROZEN_DEFINITION_BUNDLE_V1",
    ]) == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_cli_help_and_import_do_not_resolve_workspace_or_load_optional_backends(capsys):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "preflight" in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cli_process_control_is_not_converted_to_a_normal_result(monkeypatch, exception):
    import quantpits.research.forward_definition_publication as publication
    monkeypatch.setattr(
        publication, "prepare_frozen_shadow_forward_definition_publication",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        cli.main([
            "preflight", "--workspace", "/private/workspace",
            "--evidence-cycle", CYCLE, "--activation", "/private/activation",
            "--definition-store", "/private/store",
        ])
