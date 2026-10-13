"""Reload models saved by this pipeline from an explicitly selected recorder."""

import inspect


def load_owned_model(recorder, name):
    """Use Qlib's trusted loader only for our own selected model artifacts.

    Older Qlib versions have no ``trusted`` argument and retain their original
    loading behavior. Inspect the API before loading; a loading failure must
    never trigger a retry with different permissions or a different source.
    Data artifacts should continue to use ``recorder.load_object`` directly.
    """
    load = recorder.load_object
    trusted = inspect.signature(load).parameters.get("trusted")
    if trusted is not None and trusted.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return load(name, trusted=True)
    return load(name)
