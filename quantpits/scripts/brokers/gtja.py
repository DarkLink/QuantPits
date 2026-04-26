import pandas as pd
from .base import BaseBrokerAdapter


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

    def read_settlement(self, file_path: str) -> pd.DataFrame:
        """
        读取并清洗国泰君安交割单
        """
        try:
            # 读取文件，强制把证券代码读成字符串防止前导0丢失
            df = pd.read_excel(
                file_path, 
                sheet_name="Sheet1", 
                skiprows=5, 
                dtype={"证券代码": str}
            )
            
            # 清洗字符串，剥离系统导出的尾随或前导制表符 '\t'
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.lstrip("\t")
            
            # 校验并返回
            return self.validate(df)
            
        except Exception as e:
            print(f"  [WARN] [{self.name}] Error loading {file_path}: {e}")
            return pd.DataFrame()

    def _read_and_filter(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(
                file_path, 
                sheet_name="Sheet1", 
                skiprows=5, 
                dtype={"证券代码": str}
            )
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.lstrip("\t").str.strip()
            
            if "证券代码" in df.columns:
                df = df[df["证券代码"].notna() & (df["证券代码"] != "nan")].copy()
                df["证券代码"] = df["证券代码"].apply(lambda x: str(x).split(".")[0].zfill(6))
                df = df[df["证券代码"].str.startswith(("6", "0"))].copy()
            
            return df
        except Exception as e:
            print(f"  [WARN] [{self.name}] Error loading {file_path}: {e}")
            return pd.DataFrame()

    def read_orders(self, file_path: str) -> pd.DataFrame:
        """读取并清洗国泰君安委托单"""
        return self._read_and_filter(file_path)

    def read_trades(self, file_path: str) -> pd.DataFrame:
        """读取并清洗国泰君安成交单"""
        return self._read_and_filter(file_path)
