import pandas as pd
import numpy as np
import statsmodels.api as sm
from .utils import load_daily_amount, get_daily_features, load_holding_log, load_market_config

class PortfolioAnalyzer:
    def __init__(self, daily_amount_df=None, trade_log_df=None, holding_log_df=None, start_date=None, end_date=None):
        self.start_date = None
        self.end_date = None
        if daily_amount_df is None:
            daily_amount_df = load_daily_amount()
        if not daily_amount_df.empty:
            self.daily_amount = daily_amount_df.sort_values('成交日期').set_index('成交日期')
            if start_date:
                self.start_date = pd.to_datetime(start_date)
            if end_date:
                self.end_date = pd.to_datetime(end_date)
            # DO NOT FILTER DATES IN __init__ to preserve history for shifting
            # We will filter it AFTER shifting in calculate_daily_returns
        else:
            self.daily_amount = pd.DataFrame()
            
        if trade_log_df is None:
            from .utils import load_trade_log
            trade_log_df = load_trade_log()
            
        if not trade_log_df.empty:
            self.trade_log = trade_log_df
        else:
            self.trade_log = pd.DataFrame()

        if holding_log_df is None:
            holding_log_df = load_holding_log()
            
        if not holding_log_df.empty:
            self.holding_log = holding_log_df
            if start_date:
                self.holding_log = self.holding_log[self.holding_log['成交日期'] >= self.start_date]
            if end_date:
                self.holding_log = self.holding_log[self.holding_log['成交日期'] <= self.end_date]
        else:
            self.holding_log = pd.DataFrame()
        
    def calculate_daily_returns(self):
        """
        Calculate daily portfolio returns.
        Calculate purely from '收盘价值' (NAV) and 'CASHFLOW'.
        """
        if self.daily_amount.empty:
            return pd.Series(dtype=float)
            
        df = self.daily_amount
        
        # Parse based on strictly allowed columns by user
        nav_col = '收盘价值' if '收盘价值' in df.columns else '当日结存'
        if nav_col in df.columns:
            nav = df[nav_col].astype('float')
            flow = pd.Series(0, index=nav.index)
            
            if 'CASHFLOW' in df.columns:
                flow = df['CASHFLOW'].fillna(0).astype('float')
            elif '今日出入金' in df.columns:
                flow = df['今日出入金'].fillna(0).astype('float')
            elif '资金发生数' in df.columns:
                flow = df['资金发生数'].fillna(0).astype('float')
                
            prev_nav = nav.shift(1)
            # Ret = (NAV(t) - NAV(t-1) - CF(t)) / NAV(t-1)
            ret = (nav - prev_nav - flow) / prev_nav
            ret = ret.fillna(0)
            
            # Now filter the dates returning only what the user requested
            if getattr(self, 'start_date', None) is not None:
                ret = ret[ret.index >= self.start_date]
            if getattr(self, 'end_date', None) is not None:
                ret = ret[ret.index <= self.end_date]
            return ret
            
        return pd.Series(dtype=float)

    def calculate_traditional_metrics(self):
        """
        Calculate CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Profit Factor.
        """
        returns = self.calculate_daily_returns()
        if returns.empty:
            return {}
            
        cum_ret = (1.0 + returns).cumprod()
        
        # Max Drawdown & TUW
        rolling_max = cum_ret.cummax()
        drawdowns = cum_ret / rolling_max - 1.0
        max_dd = drawdowns.min()
        
        days = len(returns)
        if days < 2:
            return {}
        years = days / 252.0
        
        abs_ret = float(cum_ret.iloc[-1] - 1.0) if cum_ret.iloc[-1] > 0 else 0.0
        cagr = (cum_ret.iloc[-1]) ** (1 / years) - 1 if cum_ret.iloc[-1] > 0 else -1
        
        benchmark_cagr = np.nan
        excess_cagr = np.nan
        tracking_error = np.nan
        information_ratio = np.nan
        rf_daily = 0.0135 / 252
        benchmark_abs_ret = np.nan
        benchmark_volatility = np.nan
        benchmark_sharpe = np.nan
        benchmark_max_dd = np.nan
        benchmark_calmar = np.nan
        benchmark_max_tuw = np.nan
        benchmark_avg_tuw = np.nan
        
        if 'CSI300' in self.daily_amount.columns:
            market_close = self.daily_amount['CSI300'].astype(float)
            market_close = market_close.loc[returns.index].dropna()
            if len(market_close) >= 2:
                bench_ret = market_close.pct_change().fillna(0)
                bench_cum = (1.0 + bench_ret).cumprod()
                benchmark_cagr = (bench_cum.iloc[-1]) ** (1 / years) - 1 if bench_cum.iloc[-1] > 0 else -1
                excess_cagr = ((1 + cagr) / (1 + benchmark_cagr)) - 1 if benchmark_cagr != -1 else np.nan
                
                # Active return = Portfolio return - Benchmark return
                active_return_daily = returns - bench_ret
                tracking_error = float(active_return_daily.std() * np.sqrt(252))
                if tracking_error != 0 and not pd.isna(tracking_error):
                    # IR = annualized active return mean / tracking error
                    information_ratio = float((active_return_daily.mean() * 252) / tracking_error)
                    
                # Calculate Benchmark Metrics
                benchmark_abs_ret = float(bench_cum.iloc[-1] - 1.0) if bench_cum.iloc[-1] > 0 else 0.0
                
                bench_rolling_max = bench_cum.cummax()
                bench_drawdowns = bench_cum / bench_rolling_max - 1.0
                benchmark_max_dd = bench_drawdowns.min()
                
                benchmark_volatility = bench_ret.std() * np.sqrt(252)
                benchmark_sharpe = ((bench_ret.mean() - rf_daily) / bench_ret.std()) * np.sqrt(252) if bench_ret.std() != 0 else 0
                benchmark_calmar = benchmark_cagr / abs(benchmark_max_dd) if benchmark_max_dd < 0 and benchmark_max_dd != 0 else np.nan
                
                bench_is_under_water = bench_drawdowns < 0
                bench_under_water_blocks = bench_is_under_water.ne(bench_is_under_water.shift()).cumsum()
                bench_under_water_lengths = bench_is_under_water.groupby(bench_under_water_blocks).sum()
                benchmark_max_tuw = float(bench_under_water_lengths.max()) if not bench_under_water_lengths.empty else 0
                
                bench_under_water_only = bench_under_water_lengths[bench_under_water_lengths > 0]
                benchmark_avg_tuw = float(bench_under_water_only.mean()) if not bench_under_water_only.empty else 0
        

        volatility = returns.std() * np.sqrt(252)
        sharpe = ((returns.mean() - rf_daily) / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        calmar = cagr / abs(max_dd) if max_dd < 0 and max_dd != 0 else np.nan
        
        downside = returns[returns < 0]
        sortino = ((returns.mean() - rf_daily) / downside.std()) * np.sqrt(252) if not downside.empty and downside.std() != 0 else 0
        
        win_rate = (returns > 0).mean()
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else np.nan
        
        avg_win = returns[returns > 0].mean() if not returns[returns > 0].empty else 0
        avg_loss = abs(returns[returns < 0].mean()) if not returns[returns < 0].empty else 1e-9
        win_loss_ratio = avg_win / avg_loss
        
        is_under_water = drawdowns < 0
        under_water_blocks = is_under_water.ne(is_under_water.shift()).cumsum()
        under_water_lengths = is_under_water.groupby(under_water_blocks).sum()
        max_time_under_water = float(under_water_lengths.max()) if not under_water_lengths.empty else 0
        
        # for avg, we must only avg the blocks that actually represent time under water
        # those are the ones where lengths > 0 and is_under_water was TRUE in that block
        under_water_only = under_water_lengths[under_water_lengths > 0]
        avg_time_under_water = float(under_water_only.mean()) if not under_water_only.empty else 0
        
        days_below_initial_capital = int((cum_ret < 1.0).sum())
        
        turnover_annual = 0.0
        if not self.trade_log.empty and not self.daily_amount.empty:
            
            t_log = self.trade_log.copy()
            if getattr(self, 'start_date', None) is not None:
                t_log = t_log[t_log['成交日期'] >= self.start_date]
            if getattr(self, 'end_date', None) is not None:
                t_log = t_log[t_log['成交日期'] <= self.end_date]
                
            # Matches daily amount index (which is dates) with trade sum
            nav_col = '收盘价值' if '收盘价值' in self.daily_amount.columns else '当日结存'
            if nav_col in self.daily_amount.columns and not t_log.empty:
                daily_trade = t_log.groupby('成交日期')['成交金额'].sum()
                # filter the self.daily_amount to ONLY valid dates 
                # returns is already filtered, so we can use its index
                daily_nav = self.daily_amount.loc[returns.index, nav_col].astype(float)
                # align indices
                aligned = pd.concat([daily_nav, daily_trade], axis=1).fillna(0)
                aligned.columns = ['NAV', 'Trade_Amount']
                # Avoid div by zero
                aligned['NAV'] = aligned['NAV'].replace(0, 1e-9)
                daily_turnover = (aligned['Trade_Amount'] / 2) / aligned['NAV']
                turnover_annual = float(daily_turnover.mean() * 252)
        
        return {
            'Absolute_Return': abs_ret,
            'Benchmark_Absolute_Return': float(benchmark_abs_ret),
            'CAGR': float(cagr),
            'Benchmark_CAGR': float(benchmark_cagr),
            'Excess_Return_CAGR': float(excess_cagr),
            'Volatility': float(volatility),
            'Benchmark_Volatility': float(benchmark_volatility),
            'Tracking_Error': float(tracking_error),
            'Sharpe': float(sharpe),
            'Benchmark_Sharpe': float(benchmark_sharpe),
            'Sortino': float(sortino),
            'Calmar': float(calmar),
            'Benchmark_Calmar': float(benchmark_calmar),
            'Information_Ratio': float(information_ratio),
            'Max_Drawdown': float(max_dd),
            'Benchmark_Max_Drawdown': float(benchmark_max_dd),
            'Max_Time_Under_Water_Days': float(max_time_under_water),
            'Benchmark_Max_Time_Under_Water_Days': float(benchmark_max_tuw),
            'Avg_Time_Under_Water_Days': float(avg_time_under_water),
            'Benchmark_Avg_Time_Under_Water_Days': float(benchmark_avg_tuw),
            'Days_Below_Initial_Capital': int(days_below_initial_capital),
            'Realized_Trade_Win_Rate': float(win_rate),
            'Win/Loss_Ratio': float(win_loss_ratio),
            'Profit_Factor': float(profit_factor),
            'Turnover_Rate_Annual': turnover_annual
        }

    def calculate_factor_exposure(self, market=None):
        """
        Regress daily returns against market benchmark returns.
        market 默认从 model_config.json 读取。
        """
        if market is None:
            market, _ = load_market_config()
        returns = self.calculate_daily_returns()
        if returns.empty:
            return {}
            
        min_date = returns.index.min().strftime('%Y-%m-%d')
        max_date = returns.index.max().strftime('%Y-%m-%d')
        
        # Priority: use pre-existing 'CSI300' in daily_amount to guarantee correct date alignment
        if 'CSI300' in self.daily_amount.columns:
            market_close = self.daily_amount['CSI300'].astype(float)
            # align to our returns index specifically
            market_close = market_close.loc[returns.index].dropna()
            market_return = market_close.pct_change().fillna(0)
            aligned = pd.concat([returns, market_return], axis=1).dropna()
            aligned.columns = ['Portfolio', 'Market']
        else:
            # Fallback to Get market close prices from Qlib
            features = get_daily_features(min_date, max_date, market=market, features={'close': '$close'})
            if features.empty:
                return {}
                
            features = features.reset_index()
            features['datetime'] = pd.to_datetime(features['datetime'])
            
            # Market return = mean of all instruments returns in market
            features = features.sort_values(['instrument', 'datetime'])
            features['prev_close'] = features.groupby('instrument')['close'].shift(1)
            features['ret'] = (features['close'] - features['prev_close']) / features['prev_close']
            
            market_return = features.groupby('datetime')['ret'].mean()
            
            # Align
            aligned = pd.concat([returns, market_return], axis=1).dropna()
            aligned.columns = ['Portfolio', 'Market']
        
        if len(aligned) < 2:
            return {}
            
        X = sm.add_constant(aligned['Market'])
        model = sm.OLS(aligned['Portfolio'], X).fit()
        
        alpha = (1.0 + model.params['const']) ** 252 - 1.0
        beta = model.params['Market']
        
        return {
            'Beta_Market': float(beta),
            'Annualized_Alpha': float(alpha),
            'R_Squared': float(model.rsquared)
        }

    def calculate_style_exposures(self, market=None):
        """
        Regress daily returns against proxy style factors (Size, Momentum, Volatility).
        market 默认从 model_config.json 读取。
        """
        if market is None:
            market, _ = load_market_config()
        returns = self.calculate_daily_returns()
        if returns.empty:
            return {}
            
        min_date_ts = returns.index.min()
        max_date_ts = returns.index.max()
        
        # Buffer min_date to avoid NaN drops when calculating 20-day rolling momentum and volatility
        buffer_date = min_date_ts - pd.Timedelta(days=45)
        
        features_dict = {
            'close': '$close',
            'volume': '$volume'
        }
        features = get_daily_features(buffer_date.strftime('%Y-%m-%d'), max_date_ts.strftime('%Y-%m-%d'), market=market, features=features_dict)
        if features.empty:
            return {}
            
        features = features.reset_index()
        features['datetime'] = pd.to_datetime(features['datetime'])
        
        features = features.sort_values(['instrument', 'datetime'])
        features['size'] = np.log(features['close'] * features['volume'] + 1e-9)
        features['momentum'] = features.groupby('instrument')['close'].pct_change(20)
        
        features['prev_close'] = features.groupby('instrument')['close'].shift(1)
        features['ret'] = (features['close'] - features['prev_close']) / features['prev_close']
        
        features['volatility'] = features.groupby('instrument')['ret'].rolling(20, min_periods=5).std().reset_index(0, drop=True)
        
        features = features.dropna(subset=['ret', 'size', 'momentum', 'volatility'])
        
        factor_returns = {}
        for factor in ['size', 'momentum', 'volatility']:
            # top 20% minus bottom 20%
            def _factor_ret(df):
                if len(df) < 5:
                    return 0.0
                q_top = df[factor].quantile(0.8)
                q_bot = df[factor].quantile(0.2)
                ret_top = df[df[factor] >= q_top]['ret'].mean()
                ret_bot = df[df[factor] <= q_bot]['ret'].mean()
                if pd.isna(ret_top): ret_top = 0
                if pd.isna(ret_bot): ret_bot = 0
                return ret_top - ret_bot
                
            factor_returns[factor] = features.groupby('datetime').apply(_factor_ret)
            
            
        factor_df = pd.DataFrame(factor_returns)
        
        # Calculate annualized returns for each factor to use in Performance Attribution
        # Note: Zero-investment long-short factor returns cannot be geometrically compounded (CAGR).
        # We must use arithmetic annualized returns (mean daily return * 252) mathematically.
        factor_annualized = {}
        for col in factor_df.columns:
            factor_annualized[col] = float(factor_df[col].mean() * 252)
        
        # We need Market Return. 
        if 'CSI300' in self.daily_amount.columns:
            market_close = self.daily_amount['CSI300'].astype(float)
            market_close = market_close.loc[returns.index].dropna()
            market_return = market_close.pct_change().fillna(0)
        else:
            market_return = features.groupby('datetime')['ret'].mean()
            
        aligned = pd.concat([returns, market_return, factor_df], axis=1).dropna()
        aligned.columns = ['Portfolio', 'Market'] + list(factor_df.columns)
        
        if len(aligned) < 2:
            return {}
            
        X = sm.add_constant(aligned[['Market', 'size', 'momentum', 'volatility']])
        model = sm.OLS(aligned.iloc[:, 0], X).fit()
        
        return {
            'Multi_Factor_Intercept': float(model.params.get('const', 0)) * 252,
            'Multi_Factor_Beta': float(model.params.get('Market', 0)),
            'Barra_Size_Exp': float(model.params.get('size', 0)),
            'Barra_Momentum_Exp': float(model.params.get('momentum', 0)),
            'Barra_Volatility_Exp': float(model.params.get('volatility', 0)),
            'Barra_Style_R_Squared': float(model.rsquared),
            'Factor_Annualized': factor_annualized,
            'Market_Annualized': float(aligned['Market'].mean() * 252)
        }

    def calculate_holding_metrics(self):
        """
        Calculate metrics based on daily holdings (excluding CASH).
        - Avg Daily Count of Holdings
        - Avg Concentration of Top 1 Stock
        - Avg Floating Return of Positional Stocks
        - Position Win Rate (Percentage of holdings with floating profit > 0)
        """
        if getattr(self, 'holding_log', None) is None or self.holding_log.empty:
            return {}
            
        # Exclude CASH
        df = self.holding_log[self.holding_log['证券代码'] != 'CASH'].copy()
        if df.empty:
            return {}
            
        # Group by date
        daily_groups = df.groupby('成交日期')
        
        # 1. Avg count
        daily_count = daily_groups.size()
        avg_count = float(daily_count.mean())
        
        # 2. Avg top 1 concentration
        def top1_conc(g):
            total_val = g['收盘价值'].sum()
            if total_val == 0: return 0
            max_val = g['收盘价值'].max()
            return max_val / total_val
            
        concentration = daily_groups.apply(top1_conc).mean()
        
        # 3. Avg floating return
        if '浮盈收益率' in df.columns:
            avg_float_ret = float(df['浮盈收益率'].mean())
            win_rate = float((df['浮盈收益率'] > 0).mean())
        else:
            avg_float_ret = np.nan
            win_rate = np.nan
            
        return {
            'Avg_Daily_Holdings_Count': float(avg_count),
            'Avg_Top1_Concentration': float(concentration),
            'Avg_Floating_Return': float(avg_float_ret),
            'Daily_Holding_Win_Rate': float(win_rate)
        }

    def calculate_classified_returns(self):
        """
        Calculates separate returns for Quant-generated vs Manual trades.
        Returns a dictionary with quant metrics and a dataframe of manual trades.
        """
        import os
        from .utils import ROOT_DIR
        
        # Load trade classification
        class_path = os.path.join(ROOT_DIR, "data", "trade_classification.csv")
        if not os.path.exists(class_path) or self.trade_log is None or self.trade_log.empty:
            return None
            
        class_df = pd.read_csv(class_path)
        
        # Merge with holding log to get PnL details
        if '成交日期' not in class_df.columns and 'trade_date' in class_df.columns:
            class_df['成交日期'] = pd.to_datetime(class_df['trade_date'])
        if '证券代码' not in class_df.columns and 'instrument' in class_df.columns:
            class_df['证券代码'] = class_df['instrument']
            
        # We need the sell trades from the trade log to see the realized PnL of manual trades
        df = self.trade_log[self.trade_log['交易类别'].str.contains('卖出', na=False)].copy()
        if df.empty:
            return None
            
        df['成交日期'] = pd.to_datetime(df['成交日期'])
        class_df['成交日期'] = pd.to_datetime(class_df['成交日期'])
        
        merged_sells = pd.merge(df, class_df[['成交日期', '证券代码', 'trade_class']], on=['成交日期', '证券代码'], how='left')
        merged_sells['trade_class'] = merged_sells['trade_class'].fillna('U')
        
        if getattr(self, 'start_date', None) is not None:
            merged_sells = merged_sells[merged_sells['成交日期'] >= pd.to_datetime(self.start_date)]
        if getattr(self, 'end_date', None) is not None:
            merged_sells = merged_sells[merged_sells['成交日期'] <= pd.to_datetime(self.end_date)]
            
        # PnL logic - approximate using sell events
        # Note: True quant vs manual portfolio separation requires two separate holding ledgers.
        # Here we approximate by looking at the generated PnL on sell events.
        
        # Get manual trades details
        manual_trades = merged_sells[merged_sells['trade_class'] == 'M'].copy()
        manual_pnl = 0.0
        manual_details = []
        
        # Check actual buys to see if manual occurred
        df_buys = self.trade_log[self.trade_log['交易类别'].str.contains('买入', na=False)].copy()
        df_buys['成交日期'] = pd.to_datetime(df_buys['成交日期'])
        merged_buys = pd.merge(df_buys, class_df[['成交日期', '证券代码', 'trade_class']], on=['成交日期', '证券代码'], how='left')
        merged_buys['trade_class'] = merged_buys['trade_class'].fillna('U')
        
        if getattr(self, 'start_date', None) is not None:
            merged_buys = merged_buys[merged_buys['成交日期'] >= pd.to_datetime(self.start_date)]
        if getattr(self, 'end_date', None) is not None:
            merged_buys = merged_buys[merged_buys['成交日期'] <= pd.to_datetime(self.end_date)]
            
        manual_buys = merged_buys[merged_buys['trade_class'] == 'M'].copy()
        
        # We need to construct a comprehensive view. The easiest way is to re-calculate PnL.
        import warnings
        warnings.filterwarnings('ignore')
        
        return {
            'quant_only_metrics': {}, # Placeholder, full rigorous separation needs dual ledgers
            'manual_buys_count': len(manual_buys),
            'manual_sells_count': len(manual_trades),
            'manual_buys': manual_buys,
            'manual_sells': manual_trades,
            'class_df': class_df
        }
