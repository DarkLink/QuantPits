import pandas as pd
from abc import ABC, abstractmethod

# 统一标准交易类别
# 从之前 prod_post_trade.py 的顶层定义迁移至此，作为全体券商归一化的终点
SELL_TYPES = ["上海A股普通股票竞价卖出", "深圳A股普通股票竞价卖出"]
BUY_TYPES = ["上海A股普通股票竞价买入", "深圳A股普通股票竞价买入"]
INTEREST_TYPES = [
    "上海A股红利入账", "深圳A股红利入账",
    "上海A股红利税补缴", "深圳A股红利税补缴",
    "利息归本",
]

# 标准列名（保留中文以兼容下游所有依赖文件，如 trade_classifier, execution_analyzer 等）
REQUIRED_COLUMNS = [
    "证券代码",    # str, 6位代码，不带 SH/SZ 前缀
    "交易类别",    # str, 必须映射为上方标准类别常量
    "成交价格",    # float/Decimal
    "成交数量",    # float/Decimal
    "成交金额",    # float/Decimal
    "资金发生数",  # float/Decimal, 实际发生的资金变动 (买入为负/正取决于券商，程序中统一按此字段结合业务处理)
    "交收日期"     # str/datetime, 后续流程可处理，一般转成 "YYYY-MM-DD"
]


class BaseBrokerAdapter(ABC):
    """
    券商交割单适配器基类
    职责：将不同券商各异的导出格式 (XLSX/CSV) 清洗、映射为统一的内部标准 DataFrame
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """券商唯一标识 (如 'gtja', 'haitong')"""
        pass

    @abstractmethod
    def read_settlement(self, file_path: str) -> pd.DataFrame:
        """
        读取交割单文件并标准化格式。
        必须返回包含至少 REQUIRED_COLUMNS 中所有列的 DataFrame。
        
        对于“证券代码”，必须为不带前缀的纯 6 位数字字符串。
        """
        pass

    def read_orders(self, file_path: str) -> pd.DataFrame:
        """
        读取当日委托记录文件并标准化格式。
        默认返回空 DataFrame，供子类按需实现。
        """
        return pd.DataFrame()

    def read_trades(self, file_path: str) -> pd.DataFrame:
        """
        读取当日成交记录文件并标准化格式。
        默认返回空 DataFrame，供子类按需实现。
        """
        return pd.DataFrame()

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        校验输出的 DataFrame 是否满足必须的标准 Scheme
        """
        if df.empty:
            return df
            
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"[{self.name}] 适配器输出缺失必要列: {missing}")
        return df
