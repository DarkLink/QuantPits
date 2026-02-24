import pandas as pd
import numpy as np
from .utils import load_trade_log, get_daily_features

class ExecutionAnalyzer:
    def __init__(self, trade_log_df=None, start_date=None, end_date=None):
        """
        trade_log_df: dataframe from load_trade_log(), contains all real trades
        """
        if trade_log_df is None:
            trade_log_df = load_trade_log()
        self.trade_log = trade_log_df
        
        if not self.trade_log.empty:
            if start_date:
                self.trade_log = self.trade_log[self.trade_log['成交日期'] >= pd.to_datetime(start_date)]
            if end_date:
                self.trade_log = self.trade_log[self.trade_log['成交日期'] <= pd.to_datetime(end_date)]
        
    def calculate_slippage_and_delay(self, market="csi300"):
        """
        Delay Cost: Close(T-1) -> Open(T)
        Execution Slippage: Open(T) -> Actual Execution Price(T)
        Returns a DataFrame of trades with slippage metrics added.
        """
        if self.trade_log.empty:
            return pd.DataFrame()
            
        df = self.trade_log.copy()
        
        min_date = df['成交日期'].min()
        max_date = df['成交日期'].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return pd.DataFrame()
            
        min_date_str = min_date.strftime('%Y-%m-%d')
        max_date_str = max_date.strftime('%Y-%m-%d')
        
        # Get price features
        features_dict = {
            'close': '$close',
            'open': '$open',
            'unadj_open': '$open / $factor',
            'unadj_close': '$close / $factor',
            'volume': '$volume',
            'vwap': '$vwap'
        }
        features = get_daily_features(min_date_str, max_date_str, market=market, features=features_dict)
        if features.empty:
            return pd.DataFrame()
            
        features = features.reset_index()
        features['datetime'] = pd.to_datetime(features['datetime'])
        
        # Calculate prev_close
        features = features.sort_values(['instrument', 'datetime'])
        features['prev_close'] = features.groupby('instrument')['close'].shift(1)
        
        merged = pd.merge(
            df, 
            features, 
            left_on=['证券代码', '成交日期'], 
            right_on=['instrument', 'datetime'], 
            how='inner'
        )
        
        if merged.empty:
            return pd.DataFrame()
            
        # Parse Buy/Sell
        is_buy = merged['交易类别'].str.contains('买入', na=False)
        is_sell = merged['交易类别'].str.contains('卖出', na=False)
        
        # Buy friction (negative means loss/friction)
        merged.loc[is_buy, 'Delay_Cost'] = (merged['prev_close'] - merged['open']) / merged['prev_close']
        merged.loc[is_buy, 'Exec_Slippage'] = (merged['unadj_open'] - merged['成交价格']) / merged['unadj_open']
        
        # Sell friction (negative means loss/friction)
        merged.loc[is_sell, 'Delay_Cost'] = (merged['open'] - merged['prev_close']) / merged['prev_close']
        merged.loc[is_sell, 'Exec_Slippage'] = (merged['成交价格'] - merged['unadj_open']) / merged['unadj_open']
        
        merged['Total_Friction'] = merged['Delay_Cost'] + merged['Exec_Slippage']
        
        # Absolute Slippage Monetary Amount (Loss if sliding backwards)
        # Note: Slippage is negative percent when bad, so if we take it * Trade Amount it's already properly signed
        merged['Absolute_Slippage_Amount'] = merged['Total_Friction'] * merged['成交金额']
        
        # ADV Participation Rate
        # Qlib's $amount is usually in thousands or scaled down. To reconstruct the True Daily Market Turnover in RMB:
        # True Volume in shares = $volume * $factor * 100 (since volume is adjusted and in lots)
        # True Price = $vwap / $factor
        # True Amount = True Volume * True Price = $volume * $vwap * 100 (factors cancel out perfectly)
        merged['Market_Turnover_RMB'] = merged['volume'] * merged['vwap'] * 100
        merged['ADV_Participation_Rate'] = (merged['成交金额'] / merged['Market_Turnover_RMB']).replace([np.inf, -np.inf], np.nan)
        
        return merged

    def calculate_path_dependency(self, market="csi300"):
        """
        Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
        Intra-day relative to the execution price.
        """
        if self.trade_log.empty:
            return pd.DataFrame()
            
        df = self.trade_log.copy()
        
        min_date = df['成交日期'].min()
        max_date = df['成交日期'].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return pd.DataFrame()
            
        features_dict = {
            'unadj_high': '$high / $factor',
            'unadj_low': '$low / $factor'
        }
        features = get_daily_features(min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'), market=market, features=features_dict)
        if features.empty:
            return pd.DataFrame()
            
        features = features.reset_index()
        features['datetime'] = pd.to_datetime(features['datetime'])
        
        merged = pd.merge(
            df, 
            features, 
            left_on=['证券代码', '成交日期'], 
            right_on=['instrument', 'datetime'], 
            how='inner'
        )
        
        if merged.empty:
            return pd.DataFrame()
            
        is_buy = merged['交易类别'].str.contains('买入', na=False)
        is_sell = merged['交易类别'].str.contains('卖出', na=False)
        
        # Buy Excursions
        merged.loc[is_buy, 'MFE'] = (merged['unadj_high'] - merged['成交价格']) / merged['成交价格']
        merged.loc[is_buy, 'MAE'] = (merged['unadj_low'] - merged['成交价格']) / merged['成交价格']
        
        # Sell Excursions
        merged.loc[is_sell, 'MFE'] = (merged['成交价格'] - merged['unadj_low']) / merged['成交价格']
        merged.loc[is_sell, 'MAE'] = (merged['成交价格'] - merged['unadj_high']) / merged['成交价格']
        
        return merged
        
    def analyze_explicit_costs(self):
        """
        Calculate total explicit fee ratio, and sum of dividends.
        """
        if self.trade_log.empty:
            return {'fee_ratio': 0.0, 'total_fees': 0.0, 'total_dividend': 0.0}
            
        df = self.trade_log.copy()
        
        # explicit fees are typically when 交易类别 contains 买入 or 卖出
        trades = df[df['交易类别'].str.contains('买入|卖出', na=False)]
        total_volume = trades['成交金额'].sum()
        total_fees = trades['费用合计'].sum()
        fee_ratio = total_fees / total_volume if total_volume > 0 else 0
        
        # dividends
        div_in = df[df['交易类别'].str.contains('红利入账', na=False)]['资金发生数'].sum()
        div_tax = df[df['交易类别'].str.contains('红利税补缴', na=False)]['资金发生数'].sum()
        total_dividend = div_in + div_tax # div_tax is usually negative
        
        return {
            'fee_ratio': fee_ratio,
            'total_fees': total_fees,
            'total_dividend': total_dividend
        }

    def analyze_order_discrepancies(self, order_dir, market="csi300"):
        """
        Analyze substitution bias (missed buys vs actual buys).
        order_dir: Path to {workspace}/data/order_suggestions or similar.
        Returns a dict of metrics.
        """
        import os
        import glob
        from .utils import get_forward_returns
        
        if self.trade_log.empty or not os.path.exists(order_dir):
            return {}
            
        buy_files = sorted(glob.glob(os.path.join(order_dir, "buy_suggestion_*.csv")))
        
        df_trade = self.trade_log.copy()
        df_trade['date_str'] = df_trade['成交日期'].dt.strftime('%Y-%m-%d')
        df_trade_buy = df_trade[df_trade['交易类别'].str.contains('买入', na=False)]
        
        min_date = df_trade['date_str'].min()
        max_date = df_trade['date_str'].max()
        if pd.isna(min_date):
             return {}
             
        # Use 1-day or 5-day proxy for missed opportunities. Using 5-day for structural impact checking
        fwd_ret_5d = get_forward_returns(min_date, max_date, market=market, n_days=5)
        
        missed_buys_return = []
        actual_buys_return = []
        total_missed_count = 0
        total_days_with_misses = 0
        
        for f in buy_files:
            date_str = os.path.basename(f).replace("buy_suggestion_", "").replace(".csv", "")
            if len(date_str) == 8 and "-" not in date_str:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                
            day_log = df_trade_buy[df_trade_buy['date_str'] == date_str]
            if day_log.empty:
                continue
                
            actual_instruments = set(day_log['证券代码'].unique())
            k = len(actual_instruments)
            if k == 0:
                continue
                
            sugg_df = pd.read_csv(f)
            if sugg_df.empty:
                continue
            
            if 'action' in sugg_df.columns:
                sugg_df = sugg_df[sugg_df['action'] == 'BUY']
            top_k_sugg = sugg_df.head(k)['instrument'].tolist()
            missed_targets = set(top_k_sugg) - actual_instruments
            substitute_targets = actual_instruments - set(top_k_sugg)
            
            if missed_targets:
                total_missed_count += len(missed_targets)
                total_days_with_misses += 1
            
            if not fwd_ret_5d.empty:
                day_ret = fwd_ret_5d[fwd_ret_5d.index.get_level_values('datetime') == pd.to_datetime(date_str)]
                if not day_ret.empty:
                    for t in missed_targets:
                        try:
                            val = day_ret.loc[(t, pd.to_datetime(date_str)), 'return_5d']
                            if isinstance(val, pd.Series): val = val.iloc[0]
                            missed_buys_return.append(val)
                        except KeyError:
                            pass
                    for t in substitute_targets:
                        try:
                            val = day_ret.loc[(t, pd.to_datetime(date_str)), 'return_5d']
                            if isinstance(val, pd.Series): val = val.iloc[0]
                            actual_buys_return.append(val)
                        except KeyError:
                            pass

        avg_missed_return = float(np.nanmean(missed_buys_return)) if len(missed_buys_return) > 0 else 0.0
        avg_substitute_return = float(np.nanmean(actual_buys_return)) if len(actual_buys_return) > 0 else 0.0
        # Positive value means actual substitute performed BETTER than missed (Gain)
        substitute_bias_impact = avg_substitute_return - avg_missed_return

        return {
            'substitute_bias_impact': substitute_bias_impact,
            'avg_missed_buys_return': avg_missed_return,
            'avg_substitute_buys_return': avg_substitute_return,
            'total_missed_count': total_missed_count,
            'total_days_with_misses': total_days_with_misses
        }
