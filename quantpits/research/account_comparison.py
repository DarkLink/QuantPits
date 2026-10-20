"""Read-only account groups over saved actual ledgers and validated paper journals."""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import yaml

from quantpits.evidence.inspection import SourceMutationObserver
from quantpits.post_trade.state import normalize_date, normalize_instrument
from quantpits.research.historical_cycle import _stable_read
from quantpits.research.paper_accounting import ZERO, amount, cash_context, date, decimal
from quantpits.research.paper_market import strict_json
from quantpits.research.paper_store import PaperStore, latest_state
from quantpits.utils.workspace import WorkspaceContext


class AccountComparisonError(ValueError):
    pass


class UnstableAccountRead(AccountComparisonError):
    pass


@dataclass(frozen=True)
class AccountMember:
    id: str
    label: str
    kind: str
    ctx: WorkspaceContext
    notes: str = ""


@dataclass(frozen=True)
class AccountGroup:
    path: Path
    name: str
    members: tuple
    config_sha256: str


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise AccountComparisonError("%s must be non-empty text" % field)
    return value


def _path(value, base):
    path = Path(_text(value, "path")).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_account_group(path):
    path = Path(path).expanduser().resolve(strict=True)
    raw = _stable_read(path.parent, path.name)

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        value = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in value:
                raise AccountComparisonError("group mapping keys must be unique text")
            value[key] = loader.construct_object(value_node)
        return value

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=Loader)
    except yaml.YAMLError as exc:
        raise AccountComparisonError("invalid group YAML") from exc
    if not isinstance(value, dict) or set(value) != {"name", "members"}:
        raise AccountComparisonError("group requires name and members")
    name = _text(value["name"], "group name")
    if not isinstance(value["members"], list) or not value["members"]:
        raise AccountComparisonError("members must be a non-empty list")
    members, ids = [], set()
    for item in value["members"]:
        if not isinstance(item, dict) or not {"id", "kind", "workspace"} <= set(item) or set(item) - {
            "id", "label", "kind", "workspace", "mlflow_uri", "qlib_data_dir", "notes",
        }:
            raise AccountComparisonError("invalid account member fields")
        member_id = _text(item["id"], "member id")
        if member_id in ids or item["kind"] not in ("actual", "paper"):
            raise AccountComparisonError("member ids must be unique and kind must be actual or paper")
        ids.add(member_id)
        uri = _text(item["mlflow_uri"], "mlflow_uri") if "mlflow_uri" in item else None
        provider = _path(item["qlib_data_dir"], path.parent) if "qlib_data_dir" in item else Path("~/.qlib/qlib_data/cn_data").expanduser().resolve()
        ctx = WorkspaceContext.from_root(_path(item["workspace"], path.parent), mlflow_uri=uri,
                                         qlib_data_dir=provider, qlib_region="cn")
        notes = item.get("notes", "")
        if not isinstance(notes, str):
            raise AccountComparisonError("member notes must be text")
        members.append(AccountMember(member_id, _text(item.get("label", member_id), "label"), item["kind"], ctx, notes))
    return AccountGroup(path, name, tuple(members), _sha(raw))


class _AccountReader:
    """Reuse stable reads and mutation observation; no account locks or writes."""
    def __init__(self, root):
        self.root = root
        self.observer = SourceMutationObserver(root, ())
        self.files, self.directories = {}, {}

    def read(self, logical):
        if logical not in self.files:
            self.observer.add_paths((logical,))
            raw = _stable_read(self.root, logical, allow_missing=True)
            self.files[logical] = (raw, self._identity(logical, raw))
        return self.files[logical][0]

    def _identity(self, logical, raw):
        if raw is None:
            return None
        info = (self.root / logical).stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def json(self, logical, optional=False):
        raw = self.read(logical)
        if raw is None:
            if optional:
                return None
            raise AccountComparisonError("missing %s" % logical)
        value = strict_json(raw)
        if not isinstance(value, dict):
            raise AccountComparisonError("%s must contain a JSON object" % logical)
        return value

    def csv(self, logical):
        raw = self.read(logical)
        if raw is None:
            return None, []
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        names = reader.fieldnames
        if not names or len(names) != len(set(names)):
            raise AccountComparisonError("invalid CSV header: %s" % logical)
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise AccountComparisonError("invalid CSV row: %s" % logical)
        return set(names), rows

    def manifests(self):
        logical = "output/manifests/order_gen"
        self.observer.add_paths((logical,))
        paths = tuple(sorted(p.relative_to(self.root).as_posix() for p in (self.root / logical).glob("*.json")))
        self.directories[logical] = paths
        return paths

    def verify(self):
        try:
            for logical, (raw, identity) in self.files.items():
                observed = _stable_read(self.root, logical, allow_missing=True)
                if observed != raw or self._identity(logical, observed) != identity:
                    raise UnstableAccountRead("source changed during comparison: %s" % logical)
            for logical, names in self.directories.items():
                observed = tuple(sorted(p.relative_to(self.root).as_posix() for p in (self.root / logical).glob("*.json")))
                if observed != names:
                    raise UnstableAccountRead("manifest inventory changed during comparison")
            if self.observer.mutated():
                raise UnstableAccountRead("source mutation observed during comparison")
        except UnstableAccountRead:
            raise
        except Exception as exc:
            raise UnstableAccountRead("source no longer stable: %s" % exc) from exc

    def close(self):
        self.observer.close()

    def inputs(self):
        return [{"path": name, "sha256": None if raw is None else _sha(raw)}
                for name, (raw, _identity) in self.files.items()]


def _gap(result, code, detail):
    result["gaps"].append({"code": code, "detail": detail})


def _number(value):
    return format(decimal(value), "f")


def _known_sum(rows, column):
    try:
        return amount(sum((decimal(row[column]) for row in rows), ZERO))
    except (ValueError, KeyError, TypeError):
        return None


def _endpoint(selected):
    return {"date": selected, "status": "missing", "nav": None, "cash": None,
            "position_count": None, "shares": None, "cash_ratio": None,
            "positions": None, "holdings_available": False, "reliable_nav": False}


def _finish_endpoint(endpoint):
    if endpoint["positions"] is not None:
        endpoint["position_count"] = len(endpoint["positions"])
        endpoint["shares"] = _number(sum((decimal(p["quantity"]) for p in endpoint["positions"].values()), ZERO))
    if endpoint["nav"] is not None and decimal(endpoint["nav"]) > ZERO:
        if endpoint["cash"] is not None:
            endpoint["cash_ratio"] = str(decimal(endpoint["cash"]) / decimal(endpoint["nav"]))
        for position in (endpoint["positions"] or {}).values():
            position["weight"] = (str(decimal(position["market_value"]) / decimal(endpoint["nav"]))
                                  if position["market_value"] is not None and endpoint["holdings_available"] else None)


def _period(result, since, selected, start, end, external, fees):
    period = {"since": since, "date": selected, "start": start, "end": end,
              "net_external_cashflow": external, "pnl_excluding_external": None,
              "return_excluding_external": None, "fees": fees,
              "return_method": "PNL_EXCLUDING_NET_EXTERNAL_OVER_PERIOD_START_NAV"}
    if external is None:
        _gap(result, "cashflow_unknown", "net external funding is not fully recorded")
    if fees is None:
        _gap(result, "fees_unknown", "period fees are not fully recorded")
    if start["reliable_nav"] and end["reliable_nav"] and external is not None:
        pnl = decimal(end["nav"]) - decimal(start["nav"]) - decimal(external)
        period["pnl_excluding_external"] = amount(pnl)
        if decimal(start["nav"]) > ZERO:
            period["return_excluding_external"] = str(pnl / decimal(start["nav"]))
        else:
            _gap(result, "nonpositive_start_nav", "period return requires positive starting NAV")
    return period


def _dated(rows):
    for row in rows:
        row["_date"] = normalize_date(row["成交日期"])
        date(row["_date"])
    return rows


def _actual_endpoint(result, selected, daily, holdings):
    endpoint = _endpoint(selected)
    rows = [r for r in daily if r["_date"] == selected]
    if not rows:
        _gap(result, "nav_missing", "no daily NAV at %s" % selected)
    else:
        endpoint.update(nav=amount(rows[0]["收盘价值"]), status="recorded", reliable_nav=True)
    rows = [r for r in holdings if r["_date"] == selected]
    if rows:
        positions, cash = {}, None
        for row in rows:
            instrument = row["证券代码"].strip().upper()
            value = amount(row["收盘价值"])
            if instrument == "CASH":
                if cash is not None:
                    raise AccountComparisonError("duplicate cash holding at %s" % selected)
                cash = value
                continue
            instrument = normalize_instrument(instrument)
            quantity = decimal(row["持仓数量"])
            if instrument in positions or quantity <= ZERO or quantity != int(quantity) or decimal(value) < ZERO:
                raise AccountComparisonError("invalid or duplicate holding at %s" % selected)
            positions[instrument] = {"quantity": _number(quantity), "market_value": value, "weight": None}
        endpoint.update(cash=cash, positions=positions)
        if cash is not None and endpoint["reliable_nav"]:
            total = decimal(cash) + sum((decimal(p["market_value"]) for p in positions.values()), ZERO)
            endpoint["holdings_available"] = abs(total - decimal(endpoint["nav"])) <= decimal("0.01")
    if not endpoint["holdings_available"]:
        _gap(result, "holdings_incomplete", "cash/holdings/NAV unavailable or inconsistent at %s" % selected)
    _finish_endpoint(endpoint)
    return endpoint


def _source(source):
    if source is None:
        return {"status": "not_recorded"}
    if not isinstance(source, dict):
        raise AccountComparisonError("invalid frozen source")
    result = {"status": "recorded"}
    for key in ("kind", "workspace", "anchor", "path", "recorder_id", "experiment_id", "mlflow_uri", "sha256"):
        if key in source:
            result[key] = source[key]
    lineage = source.get("lineage", [])
    if not isinstance(lineage, list):
        raise AccountComparisonError("invalid frozen lineage")
    result["lineage"] = []
    for item in lineage:
        if not isinstance(item, dict) or not isinstance(item.get("snapshot"), dict):
            raise AccountComparisonError("invalid frozen lineage snapshot")
        snapshot = item["snapshot"]
        facts = {key: snapshot[key] for key in ("command", "run_id", "status", "anchor_date", "models") if key in snapshot}
        result["lineage"].append({"path": item.get("path"), "sha256": item.get("sha256"), "summary": facts})
    return result


def _order_sources(member, reader, result, since, selected, cursor):
    summaries = []
    for logical in reader.manifests():
        manifest = reader.json(logical)
        if manifest.get("command") != "order_gen" or manifest.get("workspace") not in (member.ctx.root.name, str(member.ctx.root)):
            raise AccountComparisonError("order manifest identity mismatch: %s" % logical)
        if manifest.get("status") != "success":
            continue
        records = manifest["records"]
        anchor, trade = date(records["anchor_date"]), date(records["trade_date"])
        if anchor >= trade:
            raise AccountComparisonError("order signal anchor must precede trade date")
        if not (since < trade <= selected or cursor is not None and trade > cursor):
            continue
        source = records.get("source")
        if source is not None and not isinstance(source, dict):
            raise AccountComparisonError("invalid order manifest source")
        artifacts = []
        for output in manifest["outputs"]:
            path = output["path"]
            if Path(path).name.startswith(("buy_suggestion_", "sell_suggestion_")):
                if reader.read(path) is not None:
                    columns, _rows = reader.csv(path)
                    if not ({"instrument", "证券代码"} & columns):
                        raise AccountComparisonError("invalid saved order suggestion columns")
                    artifacts.append(path)
        summaries.append({"manifest": logical, "run_id": manifest["run_id"], "anchor": anchor,
                          "trade_date": trade, "status": "suggestions_generated" if artifacts else "outputs_unavailable",
                          "relationship": "suggested_source_not_fill_proof",
                          "artifacts": artifacts, "source": {"status": "recorded", "workspace": str(member.ctx.root),
                          "anchor": anchor, "reference": source} if source else {"status": "not_recorded"}})
    summaries.sort(key=lambda s: (s["trade_date"], s["manifest"]))
    result["sources"] = {"settled_period": [s for s in summaries if since < s["trade_date"] <= selected],
                         "pending_current": [], "order_suggestions": summaries}
    future = [s for s in summaries if cursor is not None and s["trade_date"] > cursor]
    result["current"]["suggestions"] = future
    result["current"]["next_signal"] = future[-1]["source"] if future else {"status": "not_recorded"}
    result["notes"].append("Order manifests record suggestions; broker order/fill status is unknown.")


def _read_actual(member, reader, result, since, selected):
    daily_columns, daily = reader.csv("data/daily_amount_log_full.csv")
    holding_columns, holdings = reader.csv("data/holding_log_full.csv")
    trade_columns, trades = reader.csv("data/trade_log_full.csv")
    for columns, required, name in (
        (daily_columns, {"成交日期", "收盘价值"}, "daily"),
        (holding_columns, {"成交日期", "证券代码", "持仓数量", "收盘价值"}, "holdings"),
        (trade_columns, {"成交日期"}, "trades"),
    ):
        if columns is not None and not required <= columns:
            raise AccountComparisonError("invalid %s ledger columns" % name)
    daily, holdings, trades = _dated(daily), _dated(holdings), _dated(trades)
    available = sorted(r["_date"] for r in daily)
    if len(available) != len(set(available)):
        raise AccountComparisonError("duplicate daily NAV dates")
    result["available_dates"] = available
    config = reader.json("config/prod_config.json", optional=True)
    result["protected_roots"].extend(_provider_roots(reader.json("config/model_config.json", optional=True), member.ctx.root))
    cursor = (config.get("last_processed_date", config.get("current_date")) if config else None)
    if cursor is not None:
        date(cursor)
    ids = sorted({r.get("资金账号", "").strip() for r in trades} - {"", "nan", "None"})
    result["account_id"] = ids[0] if len(ids) == 1 else None
    result["account_id_source"] = "trade_log.资金账号" if len(ids) == 1 else "not_recorded_or_ambiguous"
    result["current"] = {"last_processed_date": cursor, "last_valuation_date": available[-1] if available else None,
                         "pending": None, "next_signal": {"status": "not_recorded"}}
    if cursor is None:
        _gap(result, "cursor_unknown", "account processing cursor is not recorded")
    start, end = (_actual_endpoint(result, d, daily, holdings) for d in (since, selected))
    rows = [r for r in daily if since < r["_date"] <= selected]
    external = None
    if daily_columns and "CASHFLOW" in daily_columns and rows:
        external = _known_sum(rows, "CASHFLOW")
    fees = None
    if trade_columns and "费用合计" in trade_columns:
        rows = [r for r in trades if since < r["_date"] <= selected]
        fees = _known_sum(rows, "费用合计")
    result["period"] = _period(result, since, selected, start, end, external, fees)
    _order_sources(member, reader, result, since, selected, cursor)


def _paper_endpoint(result, selected, values):
    endpoint = _endpoint(selected)
    match = [v for v in values if v["date"] == selected]
    if not match:
        _gap(result, "valuation_missing", "no saved paper valuation at %s" % selected)
        return endpoint
    value = match[-1]
    reliable = value["status"] == "COMPLETE" and value["nav"] is not None
    endpoint.update(status=value["status"], nav=value["nav"], cash=value["cash"], reliable_nav=reliable,
                    positions={p["instrument"]: {"quantity": _number(p["quantity"]), "market_value": p["market_value"],
                                               "weight": None} for p in value["positions"]}, holdings_available=reliable,
                    net_external_cashflow=value["net_external_cashflow"])
    if not reliable:
        _gap(result, "valuation_unreliable", "saved paper valuation is %s at %s" % (value["status"], selected))
    _finish_endpoint(endpoint)
    return endpoint


def _provider_roots(value, base):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("provider_root", "qlib_data_dir", "provider_uri") and isinstance(item, str) and item:
                yield str(_path(item, base))
            elif isinstance(item, (dict, list)):
                yield from _provider_roots(item, base)
    elif isinstance(value, list):
        for item in value:
            yield from _provider_roots(item, base)


def _read_paper(member, reader, result, since, selected):
    config = reader.json("config/paper_config.json")
    result["protected_roots"].extend(_provider_roots(config, member.ctx.root))
    raw = reader.json("data/paper/account.json")
    result["protected_roots"].extend(_provider_roots(raw, member.ctx.root))
    journal = PaperStore(member.ctx, config["account_id"]).read()
    if journal is None or raw["account_id"] != journal["account_id"]:
        raise AccountComparisonError("paper journal unavailable or account identity changed")
    result["account_id"] = journal["account_id"]
    result["account_id_source"] = "validated_paper_journal"
    values = [journal["bootstrap"]["valuation"]] + [v["result"] for v in journal["valuations"]]
    result["available_dates"] = sorted({v["date"] for v in values})
    settled = {r["result"]["cycle_id"]: r["result"] for r in journal["records"] if r["kind"] == "SETTLEMENT"}
    pending = [p for key, p in journal["plans"].items() if key not in settled and key not in journal["cancelled_cycles"]]
    current_pending = [{"cycle_id": p["cycle_id"], "anchor": p["anchor"], "trade_date": p["trade_date"],
                        "source": _source(p["source"])} for p in pending]
    result["current"] = {"last_processed_date": latest_state(journal)["as_of_date"],
                         "last_valuation_date": result["available_dates"][-1], "pending": current_pending,
                         "next_signal": current_pending[0]["source"] if current_pending else {"status": "not_recorded"}}
    result["sources"] = {"settled_period": [
        {"cycle_id": cycle, "trade_date": row["after"]["as_of_date"],
         "anchor": journal["plans"][cycle]["anchor"], "source": _source(journal["plans"][cycle]["source"])}
        for cycle, row in settled.items() if since < row["after"]["as_of_date"] <= selected
    ], "pending_current": current_pending, "order_suggestions": [],
        "bootstrap": {"date": journal["bootstrap"]["date"], "path": journal["bootstrap"]["source_path"],
                      "sha256": journal["bootstrap"]["source_sha256"]}}
    start, end = (_paper_endpoint(result, d, values) for d in (since, selected))
    external = (amount(decimal(end["net_external_cashflow"]) - decimal(start["net_external_cashflow"]))
                if "net_external_cashflow" in start and "net_external_cashflow" in end else None)
    fees = amount(sum((decimal(r["total_fees"]) for r in settled.values()
                       if since < r["after"]["as_of_date"] <= selected), ZERO))
    result["period"] = _period(result, since, selected, start, end, external, fees)
    result["notes"].append("Paper cash distributions/taxes require explicit events; additional slippage is zero.")


def _invalidate(result, exc):
    result.update(status="unstable" if isinstance(exc, UnstableAccountRead) else "error",
                  current=None, period=None, sources=None, available_dates=[], account_id=None)
    error = {"type": type(exc).__name__, "message": str(exc)}
    if error not in result["errors"]:
        result["errors"].append(error)


def _read_member(member, since, selected, readers):
    result = {"id": member.id, "label": member.label, "kind": member.kind, "workspace": str(member.ctx.root),
              "account_id": None, "status": "complete", "current": None, "period": None, "sources": None,
              "available_dates": [], "gaps": [], "errors": [], "notes": [member.notes] if member.notes else [],
              "inputs": [], "protected_roots": [str(member.ctx.root), str(member.ctx.qlib_data_dir)]}
    reader = None
    try:
        reader = _AccountReader(member.ctx.root)
        readers.append((result, reader))
        try:
            (_read_paper if member.kind == "paper" else _read_actual)(member, reader, result, since, selected)
        except Exception as exc:
            result["status"] = "error"
            result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        reader.verify()
        if result["status"] == "complete" and result["gaps"]:
            result["status"] = "partial"
    except Exception as exc:
        _invalidate(result, exc)
    finally:
        result["protected_roots"] = list(dict.fromkeys(result["protected_roots"]))
        if reader is not None:
            result["inputs"] = reader.inputs()
    return result


def _comparable(member):
    endpoint = (member.get("period") or {}).get("end")
    return bool(endpoint and endpoint["holdings_available"] and endpoint["reliable_nav"]
                and decimal(endpoint["nav"]) > ZERO)


def _pair(left, right):
    pair = {"left": left["id"], "right": right["id"], "status": "unavailable", "common_holdings": None,
            "only_left": None, "only_right": None, "stock_weight_overlap": None, "quantity_differences": None}
    if not (_comparable(left) and _comparable(right)):
        return pair
    a, b = left["period"]["end"]["positions"], right["period"]["end"]["positions"]
    common = set(a) & set(b)
    pair.update(status="complete", common_holdings=len(common), only_left=sorted(set(a) - set(b)),
                only_right=sorted(set(b) - set(a)), quantity_differences=sorted(
                    i for i in common if decimal(a[i]["quantity"]) != decimal(b[i]["quantity"])),
                stock_weight_overlap=str(sum((min(decimal(a[i]["weight"]), decimal(b[i]["weight"])) for i in common), ZERO)))
    return pair


@cash_context
def compare_accounts(group, *, since, selected):
    """Return every requested member, including errors and exact-date gaps."""
    date(since)
    date(selected)
    if since >= selected:
        raise AccountComparisonError("--since must precede --date")
    readers = []
    try:
        members = [_read_member(member, since, selected, readers) for member in group.members]
        # Keep earlier accounts under observation while reading later members.
        for result, reader in readers:
            try:
                reader.verify()
            except Exception as exc:
                _invalidate(result, exc)
        pairs = [_pair(left, right) for left, right in combinations(members, 2)]
        return {"schema_version": 1, "name": group.name, "group": str(group.path), "group_sha256": group.config_sha256,
                "since": since, "date": selected, "status": "complete" if all(m["status"] == "complete" for m in members) else "partial",
                "members": members, "pairs": pairs,
                "notes": ["Return is P&L excluding net external funding divided by starting NAV, not TWR.",
                          "Fees are displayed separately and are already included in NAV; do not subtract them twice.",
                          "Actual/paper differences include execution friction and cash entitlements, not just model effects.",
                          "Accounts are compared separately; their assets are not added into an actual total."]}
    finally:
        for _result, reader in readers:
            reader.close()
