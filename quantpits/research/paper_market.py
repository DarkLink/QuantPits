"""Read-only signal/backend and cash O/C observations, without Qlib globals."""
from __future__ import annotations

import io
import json
import sqlite3
import math
import struct
from contextlib import closing
from pathlib import Path
from urllib.parse import unquote, urlparse

from quantpits.research.historical_cycle import QlibCashPriceObserver, _stable_read, _digest_payload
from quantpits.research.replay import _prediction_frame, _universe_at, _universe_intervals
from quantpits.research.paper_accounting import PaperError, amount, date, token
from quantpits.training.persistence import sha256_bytes
from quantpits.runtime.mlflow_integrity import inspect_tracking_backend_presence, resolve_mlflow_resource_uri
from quantpits.evidence.contracts import finite_number


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PaperError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise PaperError("nonfinite JSON value: %s" % value)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PaperError("invalid JSON") from exc


def verify_price_row(row, instrument, selected, kind):
    if (row["instrument"], row["observation_date"], row["derived_field"]) != (instrument, selected, kind):
        raise PaperError("price evidence identity mismatch")
    numbers = {}
    for key in ("numerator", "denominator"):
        leaf = row[key]
        if leaf["status"] == "OBSERVED":
            raw = bytes.fromhex(leaf["feature_float32_bits"])
            if len(raw) != 4:
                raise PaperError("price evidence must contain one float32 word")
            value = struct.unpack("<f", raw)[0]
            if not math.isfinite(value) or leaf["feature_text"] != format(float(value), ".17g"):
                raise PaperError("price evidence float32 word mismatch")
            numbers[key] = float(value)
    factor = row["denominator"]["feature_text"] if "denominator" in numbers else None
    numerator, denominator = numbers.get("numerator"), numbers.get("denominator")
    usable = numerator is not None and denominator is not None and numerator > 0 and denominator > 0
    if usable:
        derived = numerator / denominator
        if row["status"] != "OBSERVED" or row["cash_price"] != format(derived, ".17g"):
            raise PaperError("price evidence cash derivation mismatch")
        return amount(derived), factor
    if row["status"] not in ("MISSING", "INVALID") or row["cash_price"] is not None:
        raise PaperError("invalid price evidence status")
    return None, factor


def verify_closes(receipts, selected, expected):
    matches = [r for r in receipts if r["observation_date"] == selected and r["observation_kind"] == "CASH_CLOSE"]
    if not expected and not matches:
        return
    if len(matches) != 1:
        raise PaperError("exact close receipt required")
    receipt = dict(matches[0])
    checksum = receipt.pop("digest")
    if checksum != _digest_payload(receipt):
        raise PaperError("close receipt digest mismatch")
    rows = receipt["rows"]
    if [r["instrument"] for r in rows] != sorted(expected) or receipt["requested_instruments"] != sorted(expected):
        raise PaperError("close receipt requested set mismatch")
    for status, key in (("OBSERVED", "observed_instruments"), ("MISSING", "missing_instruments"), ("INVALID", "invalid_instruments")):
        if receipt[key] != [r["instrument"] for r in rows if r["status"] == status]:
            raise PaperError("close receipt partition mismatch")
    counts = {"requested": len(rows), "raw_sources": 2 * len(rows),
              "observed": len(receipt["observed_instruments"]),
              "missing": len(receipt["missing_instruments"]), "invalid": len(receipt["invalid_instruments"])}
    if receipt["counts"] != counts or sum(counts[k] for k in ("observed", "missing", "invalid")) != len(rows):
        raise PaperError("close receipt counts mismatch")
    actual = {r["instrument"]: verify_price_row(r, r["instrument"], selected, "CASH_CLOSE")[0] for r in rows}
    if actual != expected:
        raise PaperError("observed closes differ from price evidence")


class PaperMarket:
    """Reuse the research float32 observer and its stable-read boundary."""
    def __init__(self, provider_root):
        self.root = Path(provider_root).expanduser().resolve(strict=True)
        self.calendar = self._calendar("calendars/day.txt")
        future_path = self.root / "calendars/day_future.txt"
        self.future = self._calendar("calendars/day_future.txt") if future_path.exists() else self.calendar
        if [d for d in self.future if d <= self.calendar[-1]] != list(self.calendar):
            raise PaperError("actual/future calendars disagree")
        self.observer = QlibCashPriceObserver(self.root, self.calendar)
        self.receipts = []
        self.evidence = {}

    def _calendar(self, logical):
        raw = _stable_read(self.root, logical)
        dates = tuple(date(line.strip()) for line in raw.decode("utf-8").splitlines() if line.strip())
        if not dates or dates != tuple(sorted(set(dates))):
            raise PaperError("invalid trading calendar")
        return dates

    def check_date(self, selected, actual=False):
        date(selected)
        calendar = self.calendar if actual else self.future
        if selected in calendar:
            return "VERIFIED"
        if selected > calendar[-1]:
            return "WAITING_FOR_DATA" if actual else "PENDING_CALENDAR"
        raise PaperError("selected date is not a trading day; choose another date explicitly")

    def universe(self, market, anchor):
        token(market)
        raw = _stable_read(self.root, "instruments/%s.txt" % market)
        members = _universe_at(_universe_intervals(raw), anchor)
        return list(sorted(members)), {"path": str(self.root / "instruments" / (market + ".txt")),
                                       "sha256": sha256_bytes(raw)}

    def _prices(self, selected, instruments, opening=False):
        instruments = tuple(sorted(set(instruments)))
        if not instruments:
            return {}, {}
        if opening:
            receipt = self.observer.observe_next_open(trade_date=selected, requested_instruments=instruments)
        else:
            receipt = self.observer.observe_anchor_close(anchor_date=selected, requested_instruments=instruments)
        self.receipts.append(receipt.to_dict())
        self.evidence[(selected, "open" if opening else "close")] = {r["instrument"]: dict(r) for r in receipt.rows}
        values, factors = {}, {}
        for row in receipt.rows:
            values[row["instrument"]] = amount(row["cash_price"]) if row["status"] == "OBSERVED" else None
            denominator = row["denominator"]
            factors[row["instrument"]] = denominator["feature_text"] if denominator["status"] == "OBSERVED" else None
        return values, factors

    def closes(self, selected, instruments):
        return self._prices(selected, instruments)[0]

    def execution_quotes(self, selected, instruments):
        opening, factors = self._prices(selected, instruments, opening=True)
        position = self.calendar.index(selected)
        if position == 0:
            previous, previous_factors, previous_date = {}, {}, None
        else:
            previous_date = self.calendar[position - 1]
            previous, previous_factors = self._prices(previous_date, instruments)
        return {i: {"open": opening.get(i), "previous_close": previous.get(i),
                    "factor": factors.get(i), "previous_factor": previous_factors.get(i),
                    "previous_date": previous_date,
                    "open_evidence": self.evidence.get((selected, "open"), {}).get(i),
                    "previous_close_evidence": self.evidence.get((previous_date, "close"), {}).get(i)}
                for i in sorted(set(instruments))}


def _read_run(ctx, source):
    """Observe existing local metadata without MLflow initialization/migration."""
    token(str(source["experiment_id"]))
    token(source["recorder_id"])
    presence = inspect_tracking_backend_presence(ctx.mlflow_uri, workspace_root=ctx.root)
    if not presence.metadata_ready:
        raise PaperError("signal metadata backend is missing or outside signal workspace")
    resource = presence.resource
    if resource.scheme == "sqlite":
        uri = resource.canonical_path.as_uri() + "?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                row = connection.execute(
                    "SELECT r.run_uuid, r.experiment_id, r.status, r.artifact_uri, r.lifecycle_stage "
                    "FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id "
                    "WHERE r.run_uuid = ? AND e.lifecycle_stage = 'active'",
                    (source["recorder_id"],),
                ).fetchone()
        except sqlite3.Error as exc:
            raise PaperError("cannot read existing signal SQLite metadata") from exc
        if row is None:
            raise PaperError("signal recorder does not exist in selected backend")
        run_id, experiment_id, status, artifact_uri, lifecycle = row
    else:
        import yaml
        logical = "%s/%s/meta.yaml" % (source["experiment_id"], source["recorder_id"])
        metadata = yaml.safe_load(_stable_read(resource.canonical_path, logical))
        experiment = yaml.safe_load(_stable_read(resource.canonical_path, "%s/meta.yaml" % source["experiment_id"]))
        if experiment.get("lifecycle_stage") != "active":
            raise PaperError("signal experiment is inactive")
        run_id = metadata.get("run_id", metadata.get("run_uuid"))
        experiment_id, status = metadata["experiment_id"], metadata["status"]
        artifact_uri, lifecycle = metadata["artifact_uri"], metadata["lifecycle_stage"]
    if (run_id != source["recorder_id"] or str(experiment_id) != str(source["experiment_id"])
            or status not in ("FINISHED", 3) or lifecycle != "active"):
        raise PaperError("signal run identity/status/experiment mismatch")
    return {"run_id": run_id, "experiment_id": str(experiment_id), "status": "FINISHED",
            "artifact_uri": artifact_uri, "lifecycle_stage": lifecycle}


def load_signal(ctx, source, anchor):
    """Bind to an explicit run or file; never resolve a mutable latest pointer."""
    raw_source = dict(source)
    if "prediction" in source:
        path = Path(source["prediction"])
        path = (ctx.root / path).resolve() if not path.is_absolute() else path.resolve()
        raw = path.read_bytes()
        identity = {"kind": "FILE", "workspace": str(ctx.root), "path": str(path)}
    else:
        run = _read_run(ctx, source)
        resource = resolve_mlflow_resource_uri(run["artifact_uri"], workspace_root=ctx.root)
        if not resource.contained:
            raise PaperError("signal artifact is outside selected signal workspace")
        parsed = urlparse(run["artifact_uri"])
        if parsed.scheme not in ("", "file") or parsed.netloc not in ("", "localhost"):
            raise PaperError("paper requires a local signal artifact URI")
        root = Path(unquote(parsed.path))
        if not root.is_absolute():
            raise PaperError("signal artifact URI must be absolute")
        path = (root / "pred.pkl").resolve(strict=True)
        try:
            path.relative_to(ctx.root)
        except ValueError as exc:
            raise PaperError("signal artifact file escapes signal workspace") from exc
        raw = path.read_bytes()
        identity = {"kind": "MLFLOW", "workspace": str(ctx.root), "mlflow_uri": ctx.mlflow_uri,
                    "experiment_id": run["experiment_id"], "recorder_id": run["run_id"],
                    "artifact_uri": run["artifact_uri"], "path": str(path), "metadata": run}
    if path.suffix.lower() == ".csv":
        import pandas as pd
        frame = pd.read_csv(io.BytesIO(raw))
        if not {"datetime", "instrument", "score"} <= set(frame.columns):
            raise PaperError("signal CSV needs datetime, instrument, score")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index(["datetime", "instrument"])[["score"]]
        stream = io.BytesIO()
        frame.to_pickle(stream)
        decoded = _prediction_frame(stream.getvalue(), "signal")
    else:
        decoded = _prediction_frame(raw, "signal")
    if anchor not in decoded:
        raise PaperError("signal has no rows at selected anchor")
    scores = {i: float(v) if finite_number(v) else None for i, v in decoded[anchor]}
    identity.update(sha256=sha256_bytes(raw), anchor=anchor, requested_source=raw_source)
    # Preserve optional fixed fit/reload source snapshots, not only their mutable paths.
    identity["lineage"] = []
    for filename in source.get("lineage_files", []):
        selected = Path(filename)
        selected = (ctx.root / selected).resolve() if not selected.is_absolute() else selected.resolve()
        contents = selected.read_bytes()
        identity["lineage"].append({"path": str(selected), "sha256": sha256_bytes(contents),
                                     "snapshot": strict_json(contents)})
    return scores, identity
