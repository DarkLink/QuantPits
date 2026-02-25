import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings

class EnsembleAnalyzer:
    def __init__(self, models_preds):
        """
        models_preds: dict {model_name: pred_df}
        Each pred_df should have multi-index (datetime, instrument) and 'score' column.
        """
        self.models_preds = {}
        for name, df in models_preds.items():
            if 'score' in df.columns:
                self.models_preds[name] = df.dropna(subset=['score'])
                
    def calculate_signal_correlation(self):
        """
        Calculate the Spearman correlation matrix between models' cross-sectional rankings per day,
        then average them over time.
        """
        if len(self.models_preds) < 2:
            return pd.DataFrame()
            
        dfs = []
        for name, df in self.models_preds.items():
            _df = df[['score']].rename(columns={'score': name})
            dfs.append(_df)
            
        merged = pd.concat(dfs, axis=1, join='inner').dropna()
        if merged.empty:
            return pd.DataFrame()
            
        def _cross_sectional_corr(df):
            if len(df) < 2:
                return pd.DataFrame(np.nan, index=merged.columns, columns=merged.columns)
            corr_matrix = df.corr(method='spearman')
            return corr_matrix
            
        # Suppress warnings inside apply
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            daily_corr = merged.groupby(level='datetime').apply(_cross_sectional_corr).dropna(how='all')
            
        if daily_corr.empty:
            return pd.DataFrame()
            
        # The index will be (datetime, model_name). We want to average over datetime.
        return daily_corr.groupby(level=1).mean()

    def calculate_marginal_contribution(self, next_returns_df, top_q=0.2):
        """
        Calculate Marginal Contribution to Sharpe (Proxy using TopQ Return Sharpe).
        Evaluate the impact of dropping each specific model.
        """
        if len(self.models_preds) < 2:
            return {}
            
        dfs = []
        for name, df in self.models_preds.items():
            _df = df[['score']].rename(columns={'score': name})
            dfs.append(_df)
            
        merged = pd.concat(dfs, axis=1, join='inner').dropna()
        if merged.empty:
            return {}
            
        # Standardize scores cross-sectionally per day to simulate equal-weight combo
        def _zscore(df):
            std = df.std()
            std = std.replace(0, 1) # prevent division by zero
            return (df - df.mean()) / std
        
        merged_z = merged.groupby(level='datetime', group_keys=False).apply(_zscore)
        full_score = merged_z.mean(axis=1)
        
        def _score_to_sharpe(score_series):
            # Combine score with returns
            df = score_series.to_frame('score').join(next_returns_df, how='inner').dropna()
            if df.empty:
                return 0.0
            ret_col = next_returns_df.columns[0]
            
            def _top_ret(x):
                if len(x) == 0:
                    return np.nan
                q = x['score'].quantile(1 - top_q)
                return x[x['score'] >= q][ret_col].mean()
                
            daily_returns = df.groupby(level='datetime').apply(_top_ret).dropna()
            if len(daily_returns) < 2 or daily_returns.std() == 0:
                return 0.0
            rf_daily = 0.0135 / 252
            return ((daily_returns.mean() - rf_daily) / daily_returns.std()) * np.sqrt(252)

        full_sharpe = _score_to_sharpe(full_score)
        marginal_contributions = {}
        
        for model in self.models_preds.keys():
            without_model = merged_z.drop(columns=[model]).mean(axis=1)
            wo_sharpe = _score_to_sharpe(without_model)
            marginal_contributions[model] = full_sharpe - wo_sharpe
            
        return {
            'Full_Ensemble_Sharpe': full_sharpe,
            'Marginal_Contributions': marginal_contributions
        }

    def track_oos_vs_is_drift(self, full_backtest_returns_series, realtime_returns_series):
        """
        Compare In-Sample (backtest) vs Out-of-Sample (real-time) Sharpe Ratio.
        Series should be daily portfolio returns.
        """
        def _sharpe(ret_series):
            ret_series = ret_series.dropna()
            if len(ret_series) < 2 or ret_series.std() == 0:
                return 0.0
            rf_daily = 0.0135 / 252
            return ((ret_series.mean() - rf_daily) / ret_series.std()) * np.sqrt(252)
            
        is_sharpe = _sharpe(full_backtest_returns_series)
        oos_sharpe = _sharpe(realtime_returns_series)
        
        decay_ratio = 1.0
        if is_sharpe > 0:
            decay_ratio = oos_sharpe / is_sharpe
            
        return {
            'IS_Sharpe': is_sharpe,
            'OOS_Sharpe': oos_sharpe,
            'Decay_Ratio': decay_ratio,
            'Warning': "Strong Warning: Sharpe dropped > 50%" if decay_ratio < 0.5 else "OK"
        }

    def calculate_ensemble_ic_metrics(self, next_returns_df, top_k=22, top_q=0.1, bottom_q=0.1):
        """
        Calculate metrics for the full ensemble (equal weighted z-score).
        """
        if len(self.models_preds) == 0:
            return {}
            
        dfs = []
        for name, df in self.models_preds.items():
            _df = df[['score']].rename(columns={'score': name})
            dfs.append(_df)
            
        merged = pd.concat(dfs, axis=1, join='inner').dropna()
        if merged.empty:
            return {}
            
        def _zscore(df):
            std = df.std()
            std = std.replace(0, 1) # prevent division by zero
            return (df - df.mean()) / std
        
        merged_z = merged.groupby(level='datetime', group_keys=False).apply(_zscore)
        full_score = merged_z.mean(axis=1).to_frame('score')
        
        from .single_model_analyzer import SingleModelAnalyzer
        sma = SingleModelAnalyzer(full_score)
        
        daily_ic, ic_win_rate, icir = sma.calculate_rank_ic(next_returns_df)
        spread_df = sma.calculate_quantile_spread(next_returns_df, top_q=top_q, bottom_q=bottom_q)
        long_ic_series, long_ic_mean = sma.calculate_long_only_ic(next_returns_df, top_k=top_k)
        
        return {
            'Rank_IC_Mean': daily_ic.mean() if not daily_ic.empty else np.nan,
            'IC_Win_Rate': ic_win_rate,
            'ICIR': icir,
            'Spread_Mean': spread_df['Spread'].mean() if not spread_df.empty else np.nan,
            'Long_Only_IC_Mean': long_ic_mean
        }
