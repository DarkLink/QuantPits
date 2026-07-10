from types import SimpleNamespace

from quantpits.ensemble.execution import (
    combo_manifest_records,
    required_models_from_combos,
    success_manifest_records,
    valid_models_for_combo,
)


def test_required_models_from_combos_deduplicates_and_sorts():
    combos = [
        SimpleNamespace(models=("m2@static", "m1@static")),
        SimpleNamespace(models=("m1@static", "m3@rolling")),
    ]

    assert required_models_from_combos(combos) == (
        "m1@static",
        "m2@static",
        "m3@rolling",
    )


def test_required_models_from_empty_combos_returns_empty_tuple():
    assert required_models_from_combos([]) == ()
    assert required_models_from_combos([SimpleNamespace(models=())]) == ()


def test_valid_models_for_combo_preserves_combo_order():
    combo = SimpleNamespace(models=("m2@static", "m1@static", "m3@rolling"))

    assert valid_models_for_combo(combo, ["m1@static", "m2@static"]) == (
        "m2@static",
        "m1@static",
    )


def test_combo_manifest_records_preserve_legacy_defaults():
    records = combo_manifest_records(
        [
            {
                "name": "combo_a",
                "models": ["m1@static"],
                "method": "equal",
                "pred_file": "pred.csv",
            },
            {
                "name": "combo_b",
                "method": "ic_weighted",
            },
        ]
    )

    assert records == [
        {
            "name": "combo_a",
            "models": ["m1@static"],
            "method": "equal",
            "is_default": False,
            "pred_file": "pred.csv",
        },
        {
            "name": "combo_b",
            "models": [],
            "method": "ic_weighted",
            "is_default": False,
            "pred_file": None,
        },
    ]


def test_success_manifest_records_match_current_schema():
    combo_results = [
        {
            "name": "default_combo",
            "models": ["m1@static"],
            "method": "equal",
            "is_default": True,
            "pred_file": "pred.csv",
        }
    ]

    records = success_manifest_records(
        anchor_date="2026-07-09",
        experiment_name="DemoExp",
        combo_results=combo_results,
    )

    assert records == {
        "anchor_date": "2026-07-09",
        "n_combos": 1,
        "experiment_name": "DemoExp",
        "combos": [
            {
                "name": "default_combo",
                "models": ["m1@static"],
                "method": "equal",
                "is_default": True,
                "pred_file": "pred.csv",
            }
        ],
    }
