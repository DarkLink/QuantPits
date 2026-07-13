import pandas as pd
from decimal import Decimal
from pathlib import Path
from .base import BaseBrokerAdapter
from quantpits.post_trade.contracts import BrokerParseError


class GtjaAdapter(BaseBrokerAdapter):
    """
    国泰君安 (GTJA) 交割单适配器
    格式特点：
    - 读取 Sheet1
    - 跳过前 5 行无关表头
    - 列名自带 `证券代码`, `交易类别`, `成交价格`, `成交数量`, `成交金额`, `资金发生数`, `交收日期`，无需重命名
    - 交易类别字符串也正好等于系统标准类别，无需转换
    """
    
    @property
    def name(self) -> str:
        return "gtja"

    def _read_excel(self, file_path) -> pd.DataFrame:
        try:
            return pd.read_excel(file_path, sheet_name="Sheet1", skiprows=5, dtype={"证券代码": str})
        except Exception as exc:
            raise BrokerParseError("[%s] cannot parse %s: %s" % (self.name, file_path, exc)) from exc

    @staticmethod
    def _clean(frame: pd.DataFrame, *, filter_codes: bool) -> pd.DataFrame:
        df = frame.copy()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.lstrip("\t").str.strip()
        if filter_codes and "证券代码" in df.columns:
            codes = df["证券代码"].astype(str).str.lstrip("\t").str.strip()
            valid = ~codes.str.lower().isin(["nan", "none", ""])
            df = df.loc[valid].copy()
            df["证券代码"] = codes.loc[valid].apply(lambda value: value.split(".")[0].zfill(6))
            df = df[df["证券代码"].str.startswith(("6", "0", "3"))].copy()
        return df

    def parse_settlement(self, file_path) -> pd.DataFrame:
        return self.validate_stream(self._clean(self._read_excel(file_path), filter_codes=True), "settlement")

    def parse_orders(self, file_path) -> pd.DataFrame:
        return self.validate_stream(self._clean(self._read_excel(file_path), filter_codes=True), "order")

    def parse_trades(self, file_path) -> pd.DataFrame:
        return self.validate_stream(self._clean(self._read_excel(file_path), filter_codes=True), "trade")

    def read_settlement(self, file_path: str) -> pd.DataFrame:
        """
        读取并清洗国泰君安交割单
        """
        try:
            return self.parse_settlement(file_path)
            
        except Exception as e:
            print(f"  [WARN] [{self.name}] Error loading {file_path}: {e}")
            return pd.DataFrame()

    def _read_and_filter(self, file_path: str) -> pd.DataFrame:
        try:
            return self._clean(self._read_excel(file_path), filter_codes=True)
        except Exception as e:
            print(f"  [WARN] [{self.name}] Error loading {file_path}: {e}")
            return pd.DataFrame()

    def read_orders(self, file_path: str) -> pd.DataFrame:
        """读取并清洗国泰君安委托单"""
        return self._read_and_filter(file_path)

    def read_trades(self, file_path: str) -> pd.DataFrame:
        """读取并清洗国泰君安成交单"""
        return self._read_and_filter(file_path)

    def parse_account_snapshot(
        self, file_path, *, effective_date=None, market_date=None
    ):
        """Parse a GTJA asset export and immediately discard private fields."""
        from quantpits.post_trade.account_snapshot import (
            BrokerAccountSnapshot, position_from_mapping, source_fingerprint,
            validate_snapshot,
        )
        from quantpits.post_trade.contracts import BrokerAccountSnapshotError
        path = Path(file_path).expanduser().resolve()
        try:
            raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object)
        except Exception as exc:
            raise BrokerAccountSnapshotError("Cannot parse GTJA account snapshot") from exc
        labels = {
            "cash": ("可用资金", "资金余额", "现金"),
            "equity": ("证券市值", "股票市值", "持仓市值"),
            "total": ("总资产", "资产总值"),
            "observed": ("查询时间", "数据时间", "导出时间"),
        }
        summary = {}
        for row in raw.itertuples(index=False, name=None):
            cells = [str(value).strip() if value is not None else "" for value in row]
            for key, candidates in labels.items():
                for index, cell in enumerate(cells):
                    if any(candidate in cell for candidate in candidates) and index + 1 < len(row):
                        value = row[index + 1]
                        if value not in (None, "", "nan"):
                            summary.setdefault(key, value)
        aliases = {
            "instrument": ("证券代码",), "quantity": ("证券数量", "当前拥股", "持仓数量"),
            "available_quantity": ("可用数量", "可卖数量"),
            "display_price": ("当前价", "现价", "最新价"),
            "market_value": ("证券市值", "股票市值", "市值"),
            "corporate_action_note": ("备注", "提示"),
        }
        header_index, columns = None, None
        for idx, row in raw.iterrows():
            cells = [str(value).strip() for value in row.tolist()]
            mapped = {}
            for target, names in aliases.items():
                match = next((pos for pos, value in enumerate(cells) if value in names), None)
                if match is not None: mapped[target] = match
            if all(key in mapped for key in ("instrument", "quantity", "display_price", "market_value")):
                header_index, columns = idx, mapped; break
        if header_index is None or not all(key in summary for key in ("cash", "equity", "total", "observed")):
            raise BrokerAccountSnapshotError("GTJA asset snapshot has an unrecognized schema")
        positions = []
        for _, row in raw.iloc[header_index + 1:].iterrows():
            instrument = row.iloc[columns["instrument"]]
            text = str(instrument).strip()
            if not text or text.lower() in {"nan", "none", "合计"}:
                continue
            try:
                mapping = {key: row.iloc[index] for key, index in columns.items()}
                mapping["instrument"] = text
                positions.append(position_from_mapping(mapping))
            except Exception as exc:
                raise BrokerAccountSnapshotError("Invalid GTJA position row") from exc
        try:
            observed = pd.Timestamp(summary["observed"]).isoformat()
        except Exception as exc:
            raise BrokerAccountSnapshotError("GTJA snapshot has an invalid observation time") from exc
        def number(value):
            return Decimal(str(value).replace(",", "").strip())
        snapshot = BrokerAccountSnapshot(
            self.name, path.name, source_fingerprint(path), observed,
            effective_date, market_date,
            number(summary["cash"]), number(summary["equity"]),
            number(summary["total"]), tuple(sorted(positions, key=lambda item: item.instrument)),
            "operator_assertion" if effective_date or market_date else "broker_observation",
        )
        return validate_snapshot(snapshot)
