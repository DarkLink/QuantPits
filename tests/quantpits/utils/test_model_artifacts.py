"""Qlib model reload compatibility at the selected recorder boundary."""

from unittest.mock import Mock

import pytest

from quantpits.utils.model_artifacts import load_owned_model


def test_new_api_trusts_selected_model_on_first_call():
    calls = []
    model = object()

    class Recorder:
        def load_object(self, name, unpickler=None, *, trusted=False):
            calls.append((name, unpickler, trusted))
            assert trusted is True
            return model

    assert load_owned_model(Recorder(), "model_fold_2.pkl") is model
    assert calls == [("model_fold_2.pkl", None, True)]


@pytest.mark.parametrize("with_unpickler", [False, True])
def test_old_api_retains_original_call(with_unpickler):
    load = Mock(return_value=object())

    class OldRecorder:
        if with_unpickler:
            def load_object(self, name, unpickler=None):
                return load(name, unpickler)
        else:
            def load_object(self, name):
                return load(name)

    assert load_owned_model(OldRecorder(), "model.pkl") is load.return_value
    load.assert_called_once_with(*(("model.pkl", None) if with_unpickler else ("model.pkl",)))


@pytest.mark.parametrize("error", [TypeError("bad model"), ImportError("missing model module"),
                                  KeyboardInterrupt("stop"), SystemExit("stop")])
def test_load_failure_or_interrupt_is_not_retried(error):
    load = Mock(side_effect=error)

    class Recorder:
        def load_object(self, name, *, trusted=False):
            return load(name, trusted=trusted)

    with pytest.raises(type(error)) as raised:
        load_owned_model(Recorder(), "model.pkl")
    assert raised.value is error
    load.assert_called_once_with("model.pkl", trusted=True)


def test_unknown_signature_does_not_guess_permissions_or_load():
    class UnknownLoader:
        __signature__ = "invalid"

        def __call__(self, name, **kwargs):
            pytest.fail("an unknown API must not be guessed")

    recorder = Mock(load_object=UnknownLoader())
    with pytest.raises((TypeError, ValueError)):
        load_owned_model(recorder, "model.pkl")


def test_helper_import_requires_no_workspace_or_qlib(tmp_path):
    import os
    import subprocess
    import sys

    process_env = dict(os.environ)
    process_env.pop("QLIB_WORKSPACE_DIR", None)
    process_env.pop("MLFLOW_TRACKING_URI", None)
    process_env["PYTHONDONTWRITEBYTECODE"] = "1"
    process_env["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import quantpits.utils.model_artifacts; "
         "assert 'quantpits.utils.env' not in sys.modules; assert 'qlib' not in sys.modules"],
        cwd=str(tmp_path), env=process_env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
