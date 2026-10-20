"""Isolated native CLI invocation with explicit context and interrupt propagation."""
from __future__ import annotations

import os
import signal
import subprocess
import sys


class ProcessRunner:
    def run(self, ctx, module, arguments, log):
        environment = dict(os.environ)
        # Always override inherited production context, including read-only commands.
        if ctx is not None:
            environment.update(QLIB_WORKSPACE_DIR=str(ctx.root), MLFLOW_TRACKING_URI=ctx.mlflow_uri,
                               QLIB_DATA_DIR=str(ctx.qlib_data_dir), QLIB_REGION=ctx.qlib_region)
        else:
            for key in ("QLIB_WORKSPACE_DIR", "MLFLOW_TRACKING_URI", "QLIB_DATA_DIR", "QLIB_REGION"):
                environment.pop(key, None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with log.open("ab") as stream:
            child = subprocess.Popen([sys.executable, "-m", module] + list(arguments), env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result = child.wait()
            except BaseException:
                # The parent records interruption only after the real child exits.
                previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
                try:
                    def send(sig):
                        try:
                            os.killpg(child.pid, sig)
                        except ProcessLookupError:
                            pass  # the process can exit between poll and signal
                    if child.poll() is None:
                        send(signal.SIGINT)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            send(signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                send(signal.SIGKILL)
                    child.wait()
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
                raise
        if result < 0 or result in (130, 143):
            raise KeyboardInterrupt("native process interrupted (%s)" % result)
        return result
