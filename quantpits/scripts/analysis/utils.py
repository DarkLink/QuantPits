import os
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
import env

ROOT_DIR = env.ROOT_DIR
CONFIG_FILE = os.path.join(ROOT_DIR, "config", "prod_config.json")
MODEL_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "model_config.json")
PREDICTION_DIR = os.path.join(ROOT_DIR, "output", "predictions")

# 默认市场和基准（当配置文件不存在时使用）
DEFAULT_MARKET = "csi300"
DEFAULT_BENCHMARK = "SH000300"


def load_market_config():
    """
    从 model_config.json 读取 market 和 benchmark 配置。
    返回 (market, benchmark) 元组。
    """
    market = DEFAULT_MARKET
    benchmark = DEFAULT_BENCHMARK
    if os.path.exists(MODEL_CONFIG_FILE):
        try:
            with open(MODEL_CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            market = cfg.get("market", DEFAULT_MARKET)
            benchmark = cfg.get("benchmark", DEFAULT_BENCHMARK)
        except (json.JSONDecodeError, IOError):
            pass
    return market, benchmark


def init_qlib():
    """Initialize Qlib environment."""
    import qlib
    from qlib.constant import REG_CN
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)


def get_trading_dates(start_date, end_date):
    """Get market trading dates between start_date and end_date."""
    from qlib.data import D
    cal = D.calendar(start_time=start_date, end_time=end_date)
    return [d.strftime('%Y-%m-%d') for d in cal]


def get_daily_features(start_date, end_date, market=None, features=None):
    """
    Fetch daily features from Qlib. Default fetches price information.
    market 默认从 model_config.json 读取，未配置时 fallback 到 csi300。
    """
    from qlib.data import D
    if market is None:
        market, _ = load_market_config()
    if features is None:
        features = {
            'close': '$close',
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'vwap': '$vwap',
            'factor': '$factor',
            'adj_close': '$close / $factor',
            'adj_open': '$open / $factor',
        }
    instruments = D.instruments(market=market)
    fields = list(features.values())
    names = list(features.keys())
    
    df = D.features(instruments, fields, start_time=start_date, end_time=end_date)
    df.columns = names
    return df


def get_forward_returns(start_date, end_date, market=None, n_days=1):
    """
    Fetch N-day forward returns from Qlib.
    Return from T's close to T+N's close.
    market 默认从 model_config.json 读取。
    """
    from qlib.data import D
    if market is None:
        market, _ = load_market_config()
    instruments = D.instruments(market=market)
    field = f"Ref($close, -{n_days}) / $close - 1"
    df = D.features(instruments, [field], start_time=start_date, end_time=end_date)
    df.columns = [f'return_{n_days}d']
    return df


def load_trade_log():
    """Load the realistic full trade execution log."""
    trade_log_path = os.path.join(ROOT_DIR, "data", "trade_log_full.csv")
    if not os.path.exists(trade_log_path):
        print(f"Warning: Trade log not found at {trade_log_path}")
        return pd.DataFrame()
    df = pd.read_csv(trade_log_path)
    if '成交日期' in df.columns:
        df['成交日期'] = pd.to_datetime(df['成交日期'])
    return df


def load_daily_amount():
    """Load the daily NAV and portfolio amount data."""
    daily_amount_path = os.path.join(ROOT_DIR, "data", "daily_amount_log_full.csv")
    if not os.path.exists(daily_amount_path):
        print(f"Warning: Daily amount log not found at {daily_amount_path}")
        return pd.DataFrame()
    df = pd.read_csv(daily_amount_path)
    if '成交日期' in df.columns:
        df['成交日期'] = pd.to_datetime(df['成交日期'])
    return df

def load_holding_log():
    """Load the daily holding log."""
    holding_log_path = os.path.join(ROOT_DIR, "data", "holding_log_full.csv")
    if not os.path.exists(holding_log_path):
        print(f"Warning: Holding log not found at {holding_log_path}")
        return pd.DataFrame()
    df = pd.read_csv(holding_log_path)
    if '成交日期' in df.columns:
        df['成交日期'] = pd.to_datetime(df['成交日期'])
    return df


def load_model_predictions(model_name, start_date=None, end_date=None):
    """
    Load a model's prediction series from CSVs spanning the requested dates.
    Here we expect output/predictions/model_YYYY-MM-DD.csv to exist.
    """
    import glob
    pattern = os.path.join(PREDICTION_DIR, f"{model_name}_*.csv")
    files = sorted(glob.glob(pattern))
    
    dfs = []
    for f in files:
        base = os.path.basename(f)
        date_str = base.replace(f"{model_name}_", "").replace(".csv", "")
        # Very simple date filtering (lexicographical)
        if start_date and date_str < start_date:
            continue
        # We MUST NOT skip if date_str > end_date, because newer files 
        # still contain the historical predictions up to end_date.
        
        df = pd.read_csv(f)
        if '0' in df.columns and 'score' not in df.columns:
            df = df.rename(columns={'0': 'score'})
        # Normalize the column names
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if 'score' not in df.columns and num_cols:
            df = df.rename(columns={num_cols[0]: 'score'})
            
        dfs.append(df)
        
    if not dfs:
        return pd.DataFrame()
        
    concat_df = pd.concat(dfs)
    if 'datetime' in concat_df.columns:
        concat_df['datetime'] = pd.to_datetime(concat_df['datetime'])
        concat_df = concat_df.set_index(['datetime', 'instrument']).sort_index()
    else:
        # If it doesn't have datetime as column, it might be in index
        if concat_df.index.nlevels == 2 and 'datetime' in concat_df.index.names:
            concat_df = concat_df.sort_index()
            
    # Drop duplicates if multiple prediction files have overlapping days
    concat_df = concat_df[~concat_df.index.duplicated(keep='last')]
    
    if start_date:
        concat_df = concat_df[concat_df.index.get_level_values('datetime') >= pd.to_datetime(start_date)]
    if end_date:
        concat_df = concat_df[concat_df.index.get_level_values('datetime') <= pd.to_datetime(end_date)]
            
    return concat_df
