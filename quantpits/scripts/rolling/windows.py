"""
Rolling Windows 生成模块

根据 rolling_config.yaml 的参数生成滚动训练窗口。
"""

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from quantpits.utils.constants import MONTHS_PER_YEAR


def parse_step_to_relativedelta(step_str):
    """将 '1M', '3M', '6M', '1Y' 等字符串转为 relativedelta"""
    step_str = step_str.strip().upper()
    if step_str.endswith('M'):
        months = int(step_str[:-1])
        return relativedelta(months=months), months
    elif step_str.endswith('Y'):
        years = int(step_str[:-1])
        return relativedelta(years=years), years * MONTHS_PER_YEAR
    else:
        raise ValueError(f"不支持的 test_step 格式: {step_str}，请使用 nM 或 nY")


def generate_rolling_windows(rolling_start, train_years, valid_years,
                             test_step, anchor_date):
    """
    生成滚动训练窗口列表。

    Args:
        rolling_start: 滚动起点 (如 '2012-01-01')
        train_years: 训练集年数
        valid_years: 验证集年数
        test_step: 滚动步长 (如 '1M', '3M', '1Y')
        anchor_date: 当前锚点日期 (决定最后一个窗口的截止)

    Returns:
        list of dict: 每个 dict 包含 window_idx, train_start/end,
                       valid_start/end, test_start/end
    """
    step_delta, step_months = parse_step_to_relativedelta(test_step)
    anchor = pd.Timestamp(anchor_date)
    T = pd.Timestamp(rolling_start)

    windows = []
    widx = 0

    while True:
        # 当前 window 的起点偏移
        offset = relativedelta(months=step_months * widx)

        train_start = T + offset
        train_end = train_start + relativedelta(years=train_years) - relativedelta(days=1)

        valid_start = train_end + relativedelta(days=1)
        valid_end = valid_start + relativedelta(years=valid_years) - relativedelta(days=1)

        test_start = valid_end + relativedelta(days=1)
        test_end = test_start + step_delta - relativedelta(days=1)

        # 如果 test_start 超过 anchor_date，停止生成
        if test_start > anchor:
            break

        # 如果 test_end 超过 anchor，截断到 anchor
        if test_end > anchor:
            test_end = anchor

        windows.append({
            'window_idx': widx,
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'valid_start': valid_start.strftime('%Y-%m-%d'),
            'valid_end': valid_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })

        widx += 1

    return windows
