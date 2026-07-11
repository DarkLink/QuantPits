import json

import pytest

from quantpits.post_trade.contracts import PostTradeStateConflictError, PostTradeStatePersistenceError
from quantpits.post_trade.state_outputs import StateOutputPayloads
from quantpits.post_trade.state_persistence import capture_state_fingerprints, persist_state_outputs
from quantpits.utils.workspace import WorkspaceContext


def _ctx(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    ctx.config_dir.mkdir(); ctx.data_dir.mkdir()
    ctx.config_path("prod_config.json").write_text("{}")
    ctx.config_path("cashflow.json").write_text("{}")
    return ctx


def test_cursor_is_last_and_payload_is_retryable(tmp_path):
    ctx = _ctx(tmp_path)
    expected = capture_state_fingerprints(ctx)
    payloads = StateOutputPayloads((("2026-01-02", b"detail"),), b"trade", b"holding", b"daily", b'{"last_processed_date":"2026-01-02"}')
    result = persist_state_outputs(ctx, payloads, ("2026-01-02",), expected_fingerprints=expected)
    assert result.cursor_committed
    assert result.outputs[-1] == ctx.config_path("prod_config.json")
    assert ctx.config_path("prod_config.json").read_bytes() == payloads.prod_config


def test_conflict_fails_before_first_write(tmp_path):
    ctx = _ctx(tmp_path)
    expected = capture_state_fingerprints(ctx)
    ctx.config_path("cashflow.json").write_text('{"changed":true}')
    payloads = StateOutputPayloads((), b"trade", b"holding", b"daily", b"{}")
    with pytest.raises(PostTradeStateConflictError):
        persist_state_outputs(ctx, payloads, ("2026-01-02",), expected_fingerprints=expected)
    assert not ctx.data_path("trade_log_full.csv").exists()
