from datetime import datetime

from quantpits.runtime.ids import generate_run_id


def test_generate_run_id_is_deterministic_with_now_and_suffix():
    run_id = generate_run_id(
        "ensemble_fusion",
        now=datetime(2026, 7, 9, 14, 30, 12),
        suffix="ab12",
    )

    assert run_id == "20260709_143012_ensemble_fusion_ab12"


def test_generate_run_id_sanitizes_command_and_suffix():
    run_id = generate_run_id(
        "ensemble fusion/run",
        now=datetime(2026, 7, 9, 14, 30, 12),
        suffix="x@y",
    )

    assert run_id == "20260709_143012_ensemble_fusion_run_x_y"


def test_generate_run_id_default_suffix_is_present():
    run_id = generate_run_id("order_gen", now=datetime(2026, 7, 9, 14, 30, 12))

    assert run_id.startswith("20260709_143012_order_gen_")
    assert len(run_id.rsplit("_", 1)[-1]) >= 4
