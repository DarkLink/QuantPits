import pandas as pd
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
