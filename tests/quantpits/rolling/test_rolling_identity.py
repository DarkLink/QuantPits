"""Contracts for canonical Rolling identities."""

from dataclasses import replace

import pytest

from quantpits.rolling.errors import RollingIdentityError
from quantpits.rolling.identity import (
    RollingFoldIdentity,
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingWindowIdentity,
    parse_rolling_window_key,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _slide(**updates):
    values = {
        "family": "rolling",
        "train_start": "2020-01-01",
        "train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-03-31",
        "effective_config_fingerprint": DIGEST_A,
    }
    values.update(updates)
    return RollingWindowIdentity(**values)


def _fold(valid_end="2022-03-31", reverse=False):
    segments = (
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )
    return RollingFoldIdentity(
        train_segments=tuple(reversed(segments)) if reverse else segments,
        valid_start="2022-01-01",
        valid_end=valid_end,
    )


def test_target_identity_round_trips_supported_families():
    for key in ("demo@rolling", "demo@cpcv_rolling"):
        identity = RollingTargetIdentity.parse(key)
        assert identity.target_key == key
        assert RollingTargetIdentity.parse(identity.target_key) == identity


def test_target_identity_rejects_bare_alias_embedded_mode_and_control_chars():
    for key in ("demo", "demo@slide", "demo@rolling@extra", "bad/name@rolling"):
        with pytest.raises(RollingIdentityError):
            RollingTargetIdentity.parse(key)
    with pytest.raises(RollingIdentityError):
        RollingTargetIdentity("bad\nname", "rolling")


def test_slide_window_identity_binds_all_dates_family_and_config():
    baseline = _slide()
    assert baseline.window_key.startswith("rolling:2024-01-01:2024-03-31:")
    assert parse_rolling_window_key(baseline.window_key)[:3] == (
        "rolling", "2024-01-01", "2024-03-31",
    )
    changes = (
        {"train_start": "2019-01-01"},
        {"train_end": "2022-11-30"},
        {"valid_start": "2023-02-01"},
        {"valid_end": "2023-11-30"},
        {"test_start": "2024-02-01"},
        {"test_end": "2024-04-30"},
        {"effective_config_fingerprint": DIGEST_B},
    )
    assert all(_slide(**change).window_key != baseline.window_key for change in changes)
    with pytest.raises(RollingIdentityError):
        _slide(family="cpcv_rolling")


def test_cpcv_fold_normalizes_segments_and_rejects_duplicates_or_overlap():
    assert _fold(reverse=True) == _fold()
    with pytest.raises(RollingIdentityError):
        RollingFoldIdentity(
            train_segments=(("2020-01-01", "2020-12-31"),) * 2,
            valid_start="2022-01-01", valid_end="2022-03-31",
        )
    with pytest.raises(RollingIdentityError):
        RollingFoldIdentity(
            train_segments=(("2022-01-15", "2022-02-01"),),
            valid_start="2022-01-01", valid_end="2022-03-31",
        )


def test_cpcv_window_binds_ordered_unique_fold_identities():
    first = _fold()
    second = RollingFoldIdentity(
        train_segments=(("2019-01-01", "2020-12-31"),),
        valid_start="2021-01-01", valid_end="2021-03-31",
    )
    values = {
        "family": "cpcv_rolling",
        "train_start": "2019-01-01", "train_end": "2022-12-31",
        "test_start": "2023-01-01", "test_end": "2023-03-31",
        "effective_config_fingerprint": DIGEST_A,
    }
    ordered = RollingWindowIdentity(folds=(first, second), **values)
    reversed_order = RollingWindowIdentity(folds=(second, first), **values)
    assert ordered.window_key != reversed_order.window_key
    with pytest.raises(RollingIdentityError):
        RollingWindowIdentity(folds=(first, first), **values)


def _run(**updates):
    values = {
        "workspace_fingerprint": DIGEST_A,
        "family": "rolling",
        "action": "daily",
        "plan_fingerprint": DIGEST_B,
        "config_fingerprint": DIGEST_C,
        "anchor_date": "2024-03-31",
        "target_keys": ("demo@rolling",),
        "window_keys": (_slide().window_key,),
        "runtime_params_fingerprint": DIGEST_D,
    }
    values.update(updates)
    return RollingRunIdentity(**values)


def test_run_identity_fingerprint_equals_execution_fingerprint():
    identity = _run()
    assert identity.to_public_dict()["execution_fingerprint"] == identity.fingerprint
    assert len(identity.fingerprint) == 64


@pytest.mark.parametrize("window_key", [
    "arbitrary-window",
    "slide:2024-01-01:2024-03-31:abcdef123456",
    "cpcv_rolling:2024-01-01:2024-03-31:abcdef123456",
])
def test_run_identity_rejects_noncanonical_or_foreign_family_window_keys(window_key):
    with pytest.raises(RollingIdentityError):
        _run(window_keys=(window_key,))


def test_attempt_and_renderer_flags_do_not_change_logical_execution_identity():
    baseline = _run(attempt_id="attempt-a")
    retried = replace(baseline, attempt_id="attempt-b")
    assert baseline.fingerprint == retried.fingerprint
    assert baseline.to_public_dict()["attempt_id"] != retried.to_public_dict()["attempt_id"]
    assert "renderer" not in baseline.to_public_dict()
    assert replace(baseline, action="merge").fingerprint != baseline.fingerprint
