from quantpits.config_contracts.core import ConfigArtifact, WorkspaceValidationResult
from quantpits.config_contracts.runtime_bridge import input_refs_from_validation


def test_input_refs_from_validation_preserves_relative_paths_and_fingerprints(tmp_path):
    result = WorkspaceValidationResult(
        workspace=tmp_path,
        ok=True,
        artifacts=(
            ConfigArtifact(
                name="ensemble_config",
                path=tmp_path / "config" / "ensemble_config.json",
                exists=True,
                raw={},
                normalized={},
                fingerprint="abc",
            ),
            ConfigArtifact(
                name="rolling_config",
                path=tmp_path / "config" / "rolling_config.yaml",
                exists=False,
                raw=None,
                normalized=None,
                fingerprint=None,
            ),
        ),
    )

    refs = input_refs_from_validation(result)

    assert refs[0].path == "config/ensemble_config.json"
    assert refs[0].kind == "config"
    assert refs[0].fingerprint == "abc"
    assert refs[0].required is True
    assert refs[1].path == "config/rolling_config.yaml"
    assert refs[1].required is False
