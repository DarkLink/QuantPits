"""Read account order suggestions without initializing a model or workspace runtime."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class SuggestionSourceError(ValueError):
    """An existing suggestion cannot be resolved safely."""


class SuggestionConflictError(SuggestionSourceError):
    def __init__(self, direction, date, paths):
        self.paths = tuple(str(p) for p in paths)
        super().__init__(
            "Conflicting %s suggestions for %s; select one actual file: %s"
            % (direction, date, "; ".join(self.paths))
        )


@dataclass(frozen=True)
class SuggestionFile:
    direction: str
    date: str
    path: Path


def parse_suggestion_name(path):
    """Parse the execution date at the filename tail; preserve buy/sell identity."""
    match = re.fullmatch(
        r"(buy|sell)_suggestion_(?:.+_)?(\d{4}-\d{2}-\d{2}|\d{8})\.csv",
        Path(path).name,
    )
    if not match:
        return None
    direction, value = match.groups()
    try:
        date = datetime.strptime(value, "%Y-%m-%d" if "-" in value else "%Y%m%d")
    except ValueError:
        return None
    return SuggestionFile(direction, date.strftime("%Y-%m-%d"), Path(path).absolute())


def scan_suggestion_directory(order_dir, *, workspace_root=None):
    """Read one explicit directory, never its descendants."""
    root = Path(workspace_root).resolve() if workspace_root is not None else None
    result = []
    directory = Path(order_dir)
    if root is not None:
        _within(directory, root)
    for path in sorted(directory.glob("*_suggestion_*.csv")):
        source = parse_suggestion_name(path)
        if source is None:
            continue
        if root is not None:
            _within(path, root)
        if not path.is_file():
            raise SuggestionSourceError("Suggestion is not a readable file: %s" % path)
        result.append(source)
    return tuple(result)


def _within(path, root):
    try:
        path.resolve().relative_to(root)
    except ValueError:
        raise SuggestionSourceError("Suggestion path escapes account workspace: %s" % path)


def discover_suggestions(workspace_root):
    """Discover only history, output top level and native cycle order directories."""
    root = Path(workspace_root).resolve()
    output = root / "output"
    _within(output, root)
    _within(output / "cycles", root)
    directories = [root / "data/order_history", output]
    directories.extend(sorted(output.glob("cycles/*/*/orders")))
    return tuple(source for directory in directories
                 for source in scan_suggestion_directory(directory, workspace_root=root))


def validate_selection(sources, selected_files):
    """Reject typo/external overrides even if their date has no actual buys."""
    selected = {Path(p).resolve() for p in selected_files}
    unknown = selected - {s.path.resolve() for s in sources}
    if unknown:
        raise SuggestionSourceError("Selected suggestion was not discovered: %s" %
                                    "; ".join(str(p) for p in sorted(unknown)))
    return selected


def load_suggestion(sources, direction, date, *, selected_files=()):
    """Resolve a single day/direction; identical bytes count once, conflicts raise.

    selected_files is a narrow override containing actual discovered paths. It
    cannot introduce an external file or silently choose multiple alternatives.
    Source paths and the digest of the consumed bytes are returned in df.attrs.
    """
    import pandas as pd

    selected = validate_selection(sources, selected_files)
    candidates = [s for s in sources if s.direction == direction and s.date == date]
    chosen = [s for s in candidates if s.path.resolve() in selected]
    if chosen:
        candidates = chosen
    if not candidates:
        return pd.DataFrame()

    contents = {}
    for source in candidates:
        try:
            payload = source.path.read_bytes()
        except OSError as exc:
            raise SuggestionSourceError("Cannot read suggestion %s: %s" % (source.path, exc)) from exc
        contents.setdefault(payload, []).append(str(source.path))
    if len(contents) != 1:
        raise SuggestionConflictError(direction, date, [s.path for s in candidates])
    payload, paths = next(iter(contents.items()))
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype={"instrument": str})
    except (ValueError, pd.errors.ParserError) as exc:
        raise SuggestionSourceError("Invalid suggestion %s: %s" % (paths[0], exc)) from exc
    if "instrument" not in frame.columns or frame["instrument"].isna().any():
        raise SuggestionSourceError("Suggestion requires instrument values: %s" % paths[0])
    frame.attrs["suggestion_source"] = {
        "date": date, "direction": direction, "paths": paths,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return frame
