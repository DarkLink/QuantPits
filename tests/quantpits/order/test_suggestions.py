import subprocess
import sys
from pathlib import Path

import pytest

from quantpits.order.suggestions import (
    SuggestionConflictError, SuggestionSourceError, discover_suggestions,
    load_suggestion, parse_suggestion_name, scan_suggestion_directory,
)


@pytest.mark.parametrize("name,date,direction", [
    ("buy_suggestion_2026-10-27.csv", "2026-10-27", "buy"),
    ("sell_suggestion_ensemble_2026-10-27.csv", "2026-10-27", "sell"),
    ("buy_suggestion_model_static_20261027.csv", "2026-10-27", "buy"),
    ("buy_suggestion_20261027.csv", "2026-10-27", "buy"),
])
def test_execution_date_tail(name, date, direction):
    source = parse_suggestion_name(name)
    assert (source.date, source.direction) == (date, direction)


@pytest.mark.parametrize("name", ["buy_suggestion_2026-02-30.csv", "prediction_2026-10-27.csv",
                                    "buy_suggestion_2026-10-27_extra.csv", "model_opinions_20261027.csv"])
def test_unrelated_or_invalid_names(name):
    assert parse_suggestion_name(name) is None


def _write(root, relative, text="instrument,score\nB,2\nA,1\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_narrow_account_scope_and_direction(tmp_path):
    root = tmp_path / "account"
    wanted = ["output/buy_suggestion_20261027.csv",
              "data/order_history/sell_suggestion_model_2026-10-27.csv",
              "output/cycles/p/week_20261023/orders/buy_suggestion_ensemble_2026-10-27.csv"]
    for path in wanted:
        _write(root, path)
    for path in ["output/ensemble/buy_suggestion_20261027.csv",
                 "output/experiments/buy_suggestion_20261027.csv",
                 "output/cycles/p/week_20261023/signals/buy_suggestion_20261027.csv",
                 "paper/output/buy_suggestion_20261027.csv",
                 "archive/output/buy_suggestion_20261027.csv"]:
        _write(root, path, "instrument\nUNRELATED\n")
    _write(tmp_path / "another_account", "output/buy_suggestion_20261027.csv")
    sources = discover_suggestions(root)
    assert {s.path.relative_to(root).as_posix() for s in sources} == set(wanted)
    assert {s.date for s in sources} == {"2026-10-27"}
    assert list(load_suggestion(sources, "buy", "2026-10-27").instrument) == ["B", "A"]


def test_duplicates_preserve_paths_and_order_conflicts_require_selection(tmp_path):
    a = _write(tmp_path, "output/buy_suggestion_ensemble_2026-10-27.csv")
    b = _write(tmp_path, "data/order_history/buy_suggestion_20261027.csv")
    sources = discover_suggestions(tmp_path)
    frame = load_suggestion(sources, "buy", "2026-10-27")
    assert frame.instrument.tolist() == ["B", "A"]
    assert set(frame.attrs["suggestion_source"]["paths"]) == {str(a), str(b)}
    b.write_text("instrument,score\nA,2\nB,1\n")
    with pytest.raises(SuggestionConflictError) as raised:
        load_suggestion(sources, "buy", "2026-10-27")
    assert set(raised.value.paths) == {str(a), str(b)}
    assert load_suggestion(sources, "buy", "2026-10-27", selected_files=[a]).instrument.tolist() == ["B", "A"]
    assert load_suggestion(sources, "buy", "2026-10-27", selected_files=[b]).instrument.tolist() == ["A", "B"]
    with pytest.raises(SuggestionConflictError):
        load_suggestion(sources, "buy", "2026-10-27", selected_files=[a, b])
    with pytest.raises(SuggestionSourceError, match="not discovered"):
        load_suggestion(sources, "buy", "2026-10-27", selected_files=[tmp_path / "typo.csv"])


@pytest.mark.parametrize("text", ["", "score\n1\n", "instrument,score\n,1\n", 'instrument\n"unterminated'])
def test_malformed_existing_sources_are_not_no_suggestions(tmp_path, text):
    _write(tmp_path, "buy_suggestion_20261027.csv", text)
    with pytest.raises(SuggestionSourceError):
        load_suggestion(scan_suggestion_directory(tmp_path), "buy", "2026-10-27")


def test_external_symlink_is_not_another_account_source(tmp_path):
    root = tmp_path / "account"
    other = _write(tmp_path, "other/buy_suggestion_20261027.csv")
    (root / "output").mkdir(parents=True)
    (root / "output" / other.name).symlink_to(other)
    with pytest.raises(SuggestionSourceError, match="escapes"):
        discover_suggestions(root)


def test_import_requires_no_workspace_or_qlib(tmp_path):
    import os
    environment = {k: v for k, v in os.environ.items() if k not in
                   {"QLIB_WORKSPACE_DIR", "MLFLOW_TRACKING_URI"}}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-c", """
import sys
from quantpits.order.suggestions import discover_suggestions
assert not any(k == 'qlib' or k.startswith('qlib.') or k == 'mlflow' or k.startswith('mlflow.') for k in sys.modules)
"""], env=environment, cwd=str(Path(__file__).resolve().parents[3]), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
