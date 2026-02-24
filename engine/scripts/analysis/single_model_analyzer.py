import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from .utils import get_forward_returns

class SingleModelAnalyzer:
    def __init__(self, pred_df):
        """
        pred_df should have multi-index (datetime, instrument) and 'score' column.
        """
        if 'score' not in pred_df.columns:
            raise ValueError("Predictions must contain a 'score' column.")
            
        self.pred_df = pred_df.dropna(subset=['score'])
        
    def calculate_rank_ic(self, next_returns_df):
        """
        Calculate daily Rank IC and IC Win Rate.
        next_returns_df should have multi-index (datetime, instrument) and a return column.
        """
        merged = self.pred_df.join(next_returns_df, how='inner').dropna()
        if merged.empty:
            return pd.Series(dtype=float), 0.0
            
        ret_col = next_returns_df.columns[0]
        
        def _ic(df):
            if len(df) < 2:
                return np.nan
            return spearmanr(df['score'], df[ret_col])[0]
            
        daily_ic = merged.groupby(level='datetime').apply(_ic).dropna()
        ic_win_rate = (daily_ic > 0).mean() if not daily_ic.empty else 0.0
        icir = (daily_ic.mean() / daily_ic.std()) if not daily_ic.empty and daily_ic.std() != 0 else 0.0
        
        return daily_ic, ic_win_rate, icir
        
    def calculate_ic_decay(self, market="csi300", max_days=5):
        """
        Calculate IC for T+1 up to T+max_days.
        """
        dates = self.pred_df.index.get_level_values('datetime').unique()
        if len(dates) == 0:
            return {}
            
        start_date = dates.min().strftime('%Y-%m-%d')
        end_date = dates.max().strftime('%Y-%m-%d')
        
        ic_decay = {}
        for n in range(1, max_days + 1):
            fwd_ret = get_forward_returns(start_date, end_date, market=market, n_days=n)
            fwd_ret = fwd_ret.dropna()
            
            daily_ic, ic_win, icir = self.calculate_rank_ic(fwd_ret)
            ic_decay[f"T+{n}"] = daily_ic.mean() if not daily_ic.empty else np.nan
            
        return ic_decay

    def calculate_cusum(self, series, target_mean=0.0, k=0.5):
        """
        Calculate Tabular CUSUM for a metric series (e.g., daily returns or daily IC).
        Detects structural breaks.
        Returns a DataFrame with CUSUM_POS and CUSUM_NEG.
        """
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.DataFrame({'CUSUM_POS': 0, 'CUSUM_NEG': 0}, index=series.index)
            
        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))
        
        for i, val in enumerate(series):
            if i == 0:
                continue
            z = (val - target_mean) / std
            cusum_pos[i] = max(0, cusum_pos[i-1] + z - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - z - k)
            
        return pd.DataFrame({'CUSUM_POS': cusum_pos, 'CUSUM_NEG': cusum_neg}, index=series.index)

    def calculate_psi(self, baseline_dates, current_dates, bins=10):
        """
        Population Stability Index to monitor score distribution shifts.
        """
        base_scores = self.pred_df.loc[self.pred_df.index.get_level_values('datetime').isin(baseline_dates)]['score']
        curr_scores = self.pred_df.loc[self.pred_df.index.get_level_values('datetime').isin(current_dates)]['score']
        
        if base_scores.empty or curr_scores.empty:
            return np.nan
            
        base_hist, bin_edges = np.histogram(base_scores, bins=bins)
        curr_hist, _ = np.histogram(curr_scores, bins=bin_edges)
        
        base_pct = base_hist / len(base_scores)
        curr_pct = curr_hist / len(curr_scores)
        
        base_pct = np.where(base_pct == 0, 0.0001, base_pct)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
        
        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return psi

    def calculate_quantile_spread(self, next_returns_df, top_q=0.2, bottom_q=0.2):
        """
        Compute average returns for top N% vs bottom N% and their spread.
        next_returns_df: dataframe with future returns
        """
        merged = self.pred_df.join(next_returns_df, how='inner').dropna()
        if merged.empty:
            return pd.DataFrame()
            
        ret_col = next_returns_df.columns[0]
        
        def _spread(df):
            q_top = df['score'].quantile(1 - top_q)
            q_bot = df['score'].quantile(bottom_q)
            
            top_ret = df[df['score'] >= q_top][ret_col].mean()
            bot_ret = df[df['score'] <= q_bot][ret_col].mean()
            
            return pd.Series({'Top_Ret': top_ret, 'Bottom_Ret': bot_ret, 'Spread': top_ret - bot_ret})
            
        return merged.groupby(level='datetime').apply(_spread).dropna()

    def calculate_long_only_ic(self, next_returns_df, top_k=22):
        """
        Calculate IC strictly on the top `top_k` ranked stocks per day.
        """
        merged = self.pred_df.join(next_returns_df, how='inner').dropna()
        if merged.empty:
            return pd.Series(dtype=float), 0.0
            
        ret_col = next_returns_df.columns[0]
        
        def _top_ic(df):
            if len(df) < 2:
                return np.nan
            top_df = df.nlargest(top_k, 'score')
            if len(top_df) < 2:
                return np.nan
            # Using Spearman rank correlation
            return spearmanr(top_df['score'], top_df[ret_col])[0]
            
        daily_ic = merged.groupby(level='datetime').apply(_top_ic).dropna()
        ic_mean = daily_ic.mean() if not daily_ic.empty else 0.0
        return daily_ic, ic_mean
