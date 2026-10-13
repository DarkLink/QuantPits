"""Keep CLI countdowns fast while exercising the real safeguard output."""

import sys
import time

import pytest


@pytest.fixture(autouse=True)
def skip_safeguard_countdown(monkeypatch):
    original_sleep = time.sleep
    countdowns = {
        ("quantpits.utils.env", "safeguard"),
        ("quantpits.scripts.rolling_train", "_safeguard_explicit_workspace"),
        ("rolling_train", "_safeguard_explicit_workspace"),
    }

    def sleep(seconds):
        caller = sys._getframe(1)
        if (caller.f_globals.get("__name__"), caller.f_code.co_name) in countdowns:
            return None
        # Lease, concurrency and interruption waits retain their real timing.
        return original_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)
