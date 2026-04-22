import pandas as pd
import numpy as np
import statsmodels.api as sm
from .utils import load_daily_amount, get_daily_features, load_holding_log, load_market_config
from quantpits.utils.constants import TRADING_DAYS_PER_YEAR, TRADING_WEEKS_PER_YEAR, CALENDAR_DAYS_PER_YEAR, RISK_FREE_RATE_ANNUAL

# Metric Keys Contract
BARRA_LIQD_KEY = 'Barra_Liquidity_Exp_(High-Low)'
BARRA_MOMT_KEY = 'Barra_Momentum_Exp_(High-Low)'
BARRA_VOLA_KEY = 'Barra_Volatility_Exp_(High-Low)'

class PortfolioAnalyzer:
    def __init__(self, daily_amount_df=None, trade_log_df=None, holding_log_df=None, start_date=None, end_date=None, benchmark_col='CSI300', freq='day'):
        self.start_date = None
        self.end_date = None
        self.benchmark_col = benchmark_col
        self.freq = freq
        self.periods_per_year = TRADING_DAYS_PER_YEAR if freq == 'day' else TRADING_WEEKS_PER_YEAR
        if daily_amount_df is None:
            daily_amount_df = load_daily_amount()
        if not daily_amount_df.empty:
            self.daily_amount = daily_amount_df.copy().sort_values('成交日期').set_index('成交日期')
            
            # Ensure benchmark is NAV (Value/Price) internally.
            # If values start near 0, they are returns.
            if self.benchmark_col in self.daily_amount.columns:
                bench_series = self.daily_amount[self.benchmark_col].dropna()
                if not bench_series.empty:
                    # Adaptive check: if any value is negative OR (mean < 0.2 and max < 0.5) it's considered returns
                    # This guarantees it doesn't fail based solely on the first day's outlier value
                    if (bench_series.min() < 0) or (bench_series.mean() < 0.2 and bench_series.max() < 0.5):
                        # Convert Returns to cumulative NAV
                        self.daily_amount[self.benchmark_col] = (1 + self.daily_amount[self.benchmark_col].fillna(0)).cumprod()
                        # NOTE: We DO NOT force the first element to 1.0 (iloc[0] = 1.0)
                        # because it breaks the return chain for the second element when using pct_change().
                        # If bench_returns[0] = r1, then iloc[0] = 1 + r1 is correct.

            if start_date:
                self.start_date = pd.to_datetime(start_date)
            if end_date:
                self.end_date = pd.to_datetime(end_date)
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
            # Ret = (NAV(t) - NAV(t-1) - CF(t)) / (NAV(t-1) + CF(t))
            # 采用“资金发生于期初并参与全天交易”口径 (Beginning of Day CF assumption)
            ret = (nav - prev_nav - flow) / (prev_nav + flow)
            # Drop the very first NaN (the very first day in history has no return)
            # This ensures len(ret) correctly reflects the number of return intervals.
            ret = ret.dropna()
            
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
        
        abs_ret = float(cum_ret.iloc[-1] - 1.0) if cum_ret.iloc[-1] > 0 else 0.0
        
        days = len(returns)
        if days < 2:
            return {}
        
        # Consistent with project's '252-day basis' philosophy:
        # years are calculated based on the number of trading days.
        # We always expect daily frequency for the records internally.
        years_252 = days / float(TRADING_DAYS_PER_YEAR)
        
        # Arithmetic annual return: independently-computed constant for attribution display
        portfolio_arith_annual = float(returns.mean() * self.periods_per_year)
        
        # Calculate calendar years if dates are available
        # Use the first NAV date (base of the first return) rather than returns.index[0]
        # to avoid shortening the period by weekends/holidays at the start.
        years_calendar = years_252 # Fallback
        if len(returns) >= 2:
            nav_dates_before_first_ret = self.daily_amount.index[self.daily_amount.index < returns.index[0]]
            if len(nav_dates_before_first_ret) > 0:
                start_nav_date = nav_dates_before_first_ret[-1]
            else:
                start_nav_date = returns.index[0]
            calendar_days = (returns.index[-1] - start_nav_date).days
            if calendar_days > 0:
                years_calendar = calendar_days / float(CALENDAR_DAYS_PER_YEAR)
        
        cagr_252 = (cum_ret.iloc[-1]) ** (1 / years_252) - 1 if cum_ret.iloc[-1] > 0 else -1
        cagr_calendar = (cum_ret.iloc[-1]) ** (1 / years_calendar) - 1 if cum_ret.iloc[-1] > 0 else -1
        
        benchmark_cagr_252 = np.nan
        benchmark_cagr_calendar = np.nan
        excess_cagr_252 = np.nan
        excess_cagr_calendar = np.nan
        tracking_error = np.nan
        information_ratio_arithmetic = np.nan
        information_ratio_log = np.nan
        tracking_error_geometric = np.nan
        annualized_active_return_arithmetic = np.nan
        if self.benchmark_col in self.daily_amount.columns:
            market_close = self.daily_amount[self.benchmark_col].astype(float)
            
            # CRITICAL: We must compute returns on the FULL market series FIRST,
            # then slice by the portfolio returns' index. 
            # This ensures the first day of the slice correctly captures the return from the previous day.
            market_return_full = market_close.pct_change().dropna()
            bench_ret = market_return_full.loc[market_return_full.index.intersection(returns.index)]
            
            if len(bench_ret) >= 1:
                # Use returns to build a consistent cumulative series starting from 1.0
                # at (Start Date - 1 day), effectively.
                bench_cum = (1 + bench_ret).cumprod()
                benchmark_abs_ret = float(bench_cum.iloc[-1] - 1) if not bench_cum.empty else 0.0
                
                benchmark_cagr_252 = (bench_cum.iloc[-1]) ** (1 / years_252) - 1 if (not bench_cum.empty and bench_cum.iloc[-1] > 0) else -1
                benchmark_cagr_calendar = (bench_cum.iloc[-1]) ** (1 / years_calendar) - 1 if (not bench_cum.empty and bench_cum.iloc[-1] > 0) else -1
                
                excess_cagr_252 = ((1 + cagr_252) / (1 + benchmark_cagr_252)) - 1 if benchmark_cagr_252 != -1 else np.nan
                excess_cagr_calendar = ((1 + cagr_calendar) / (1 + benchmark_cagr_calendar)) - 1 if benchmark_cagr_calendar != -1 else np.nan
                
                # Active return = Portfolio return - Benchmark return
                active_return_daily = returns.loc[bench_ret.index] - bench_ret
                tracking_error = float(active_return_daily.std() * np.sqrt(self.periods_per_year))
                if tracking_error != 0 and not pd.isna(tracking_error):
                    annualized_active_return_arithmetic = float(active_return_daily.mean() * self.periods_per_year)
                    # IR (Arithmetic) = annualized arithmetic mean of daily active returns / tracking error
                    information_ratio_arithmetic = float(annualized_active_return_arithmetic / tracking_error)
                    
                    # IR (Log-Based) = annualized mean of log-active-returns / geometric tracking error
                    # This resolves the dimension mismatch between CAGR and arithmetic tracking error.
                    log_active_ret = np.log((1 + returns) / (1 + bench_ret))
                    ann_log_active_ret = float(log_active_ret.mean() * self.periods_per_year)
                    tracking_error_geometric = float(log_active_ret.std() * np.sqrt(self.periods_per_year))
                    
                    if tracking_error_geometric != 0 and not pd.isna(tracking_error_geometric):
                        information_ratio_log = float(ann_log_active_ret / tracking_error_geometric)
                    else:
                        information_ratio_log = np.nan
                    
                # Calculate Benchmark Metrics
                bench_rolling_max = bench_cum.cummax()
                bench_drawdowns = bench_cum / bench_rolling_max - 1.0
                benchmark_max_dd = bench_drawdowns.min()
                
                benchmark_volatility = bench_ret.std() * np.sqrt(self.periods_per_year)
                rf_daily = RISK_FREE_RATE_ANNUAL / self.periods_per_year
                benchmark_sharpe = ((bench_ret.mean() - rf_daily) / bench_ret.std()) * np.sqrt(self.periods_per_year) if bench_ret.std() != 0 else 0
                benchmark_calmar = benchmark_cagr_252 / abs(benchmark_max_dd) if benchmark_max_dd < 0 and benchmark_max_dd != 0 else np.nan
                
                bench_is_under_water = bench_drawdowns < 0
                bench_under_water_blocks = bench_is_under_water.ne(bench_is_under_water.shift()).cumsum()
                bench_under_water_lengths = bench_is_under_water.groupby(bench_under_water_blocks).sum()
                benchmark_max_tuw = float(bench_under_water_lengths.max()) if not bench_under_water_lengths.empty else 0
                
                bench_under_water_only = bench_under_water_lengths[bench_under_water_lengths > 0]
                
                # EXCLUDE the final block for the AVERAGE calculation if it is unfinished 
                # (i.e. the benchmark is still in a drawdown at the end of the timeseries)
                if len(bench_is_under_water) > 0 and bench_is_under_water.iloc[-1]:
                    bench_last_block_id = bench_under_water_blocks.iloc[-1]
                    if bench_last_block_id in bench_under_water_only.index:
                        bench_under_water_only = bench_under_water_only[bench_under_water_only.index != bench_last_block_id]
                        
                benchmark_avg_tuw = float(bench_under_water_only.mean()) if not bench_under_water_only.empty else 0
        

        volatility = returns.std() * np.sqrt(self.periods_per_year)
        sharpe = ((returns.mean() - rf_daily) / returns.std()) * np.sqrt(self.periods_per_year) if returns.std() != 0 else 0
        calmar = cagr_252 / abs(max_dd) if max_dd < 0 and max_dd != 0 else np.nan
        
        downside_dev = np.sqrt(np.mean(np.minimum(0, returns)**2))
        sortino = ((returns.mean() - rf_daily) / downside_dev) * np.sqrt(self.periods_per_year) if downside_dev != 0 else 0
        
        win_rate = (returns > 0).mean()
        
        # Compute exact daily PnL for Profit Factor
        nav_col = '收盘价值' if '收盘价值' in self.daily_amount.columns else '当日结存'
        if nav_col in self.daily_amount.columns:
            nav = self.daily_amount[nav_col].astype('float')
            flow = pd.Series(0, index=nav.index)
            if 'CASHFLOW' in self.daily_amount.columns: flow = self.daily_amount['CASHFLOW'].fillna(0).astype('float')
            elif '今日出入金' in self.daily_amount.columns: flow = self.daily_amount['今日出入金'].fillna(0).astype('float')
            elif '资金发生数' in self.daily_amount.columns: flow = self.daily_amount['资金发生数'].fillna(0).astype('float')
            
            nav_aligned = nav.loc[returns.index]
            flow_aligned = flow.loc[returns.index]
            prev_nav_aligned = nav.shift(1).loc[returns.index]
            daily_pnl = nav_aligned - prev_nav_aligned - flow_aligned
            gross_profit = daily_pnl[daily_pnl > 0].sum()
            gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
        else:
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            
        daily_profit_factor_pnl = float(gross_profit / gross_loss) if gross_loss != 0 else np.nan
        
        # Calculate Return-Based Profit Factor (for Signal Quality Evaluation)
        # Gross profit sum from returns / abs(Gross loss sum from returns)
        # This is AUM-agnostic.
        gross_profit_ret = returns[returns > 0].sum()
        gross_loss_ret = abs(returns[returns < 0].sum())
        daily_profit_factor_returns = float(gross_profit_ret / gross_loss_ret) if gross_loss_ret != 0 else np.nan
        
        # ----------------------------------------------------
        # Calculate Trade-Based Profit Factor (Closed-Loop FIFO)
        # ----------------------------------------------------
        trade_profit_factor = np.nan
        if not self.trade_log.empty:
            t_log = self.trade_log.copy()
            if getattr(self, 'start_date', None) is not None:
                t_log = t_log[t_log['成交日期'] >= pd.to_datetime(self.start_date)]
            if getattr(self, 'end_date', None) is not None:
                t_log = t_log[t_log['成交日期'] <= pd.to_datetime(self.end_date)]
                
            gross_trade_profit = 0.0
            gross_trade_loss = 0.0
            trade_wins = 0
            trade_losses = 0
            positions = {} # instrument -> list of {'price': p, 'qty': q, 'fees': f}
            
            t_log = t_log.sort_values('成交日期')
            for _, row in t_log.iterrows():
                inst = row.get('证券代码', '')
                if inst == 'CASH' or not pd.notna(inst): continue
                
                trade_type = str(row.get('交易类别', ''))
                qty = float(row.get('成交数量', 0))
                price = float(row.get('成交价格', 0))
                fees = float(row.get('费用合计', 0))
                
                if pd.isna(qty) or qty == 0: continue
                
                if '买入' in trade_type:
                    if inst not in positions: positions[inst] = []
                    positions[inst].append({'price': price, 'qty': qty, 'fees': fees})
                elif '卖出' in trade_type:
                    if inst not in positions: continue
                    sell_qty_remaining = qty
                    realized_pnl = 0.0
                    
                    while sell_qty_remaining > 0 and len(positions[inst]) > 0:
                        lot = positions[inst][0]
                        matched_qty = min(lot['qty'], sell_qty_remaining)
                        
                        buy_val = matched_qty * lot['price']
                        sell_val = matched_qty * price
                        
                        buy_fee = lot['fees'] * (matched_qty / lot['qty']) if lot['qty'] > 0 else 0
                        sell_fee = fees * (matched_qty / qty) if qty > 0 else 0
                        
                        chunk_pnl = sell_val - buy_val - buy_fee - sell_fee
                        realized_pnl += chunk_pnl
                        
                        lot['qty'] -= matched_qty
                        lot['fees'] -= buy_fee
                        sell_qty_remaining -= matched_qty
                        
                        if lot['qty'] <= 1e-5:
                            positions[inst].pop(0)
                            
                    if realized_pnl > 0:
                        gross_trade_profit += realized_pnl
                        trade_wins += 1
                    elif realized_pnl < 0:
                        gross_trade_loss += abs(realized_pnl)
                        trade_losses += 1
                        
            if gross_trade_loss != 0:
                trade_profit_factor = float(gross_trade_profit / gross_trade_loss)
            elif gross_trade_profit > 0:
                trade_profit_factor = np.inf
                
            # ----------------------------------------------------
            # Calculate MTM Profit Factor (Includes Open Positions)
            # ----------------------------------------------------
            gross_trade_profit_mtm = gross_trade_profit
            gross_trade_loss_mtm = gross_trade_loss
            
            if not self.holding_log.empty:
                # Use the last day of the analysis period to value open positions
                last_date = returns.index[-1] if not returns.empty else None
                if last_date:
                    for inst, lots in positions.items():
                        total_qty = sum(l['qty'] for l in lots)
                        if total_qty <= 1e-5: continue
                        
                        # Find terminal price from holding log
                        # We look for the price on last_date, fallback to the latest available
                        inst_holdings = self.holding_log[self.holding_log['证券代码'] == inst]
                        if not inst_holdings.empty:
                            last_price_row = inst_holdings[inst_holdings['成交日期'] <= last_date].sort_values('成交日期').iloc[-1]
                            # Use '收盘价格' if available, otherwise derive from '收盘价值' / '持仓数量'
                            terminal_price = last_price_row.get('收盘价格')
                            if pd.isna(terminal_price):
                                val = last_price_row.get('收盘价值', 0)
                                q = last_price_row.get('持仓数量', 0)
                                terminal_price = val / q if q > 0 else 0
                            
                            unrealized_pnl = 0.0
                            for lot in lots:
                                if lot['qty'] <= 1e-5: continue
                                # Simplified MTM PnL (ignoring proportional fees from buy as they were already deducted if we follow standard PnL)
                                # Actually, realized_pnl = sell_val - buy_val - buy_fee - sell_fee
                                # So unrealized_pnl = virtual_sell_val - buy_val - buy_fee
                                unrealized_pnl += (lot['qty'] * terminal_price - lot['qty'] * lot['price'] - lot['fees'])
                                
                            if unrealized_pnl > 0:
                                gross_trade_profit_mtm += unrealized_pnl
                            elif unrealized_pnl < 0:
                                gross_trade_loss_mtm += abs(unrealized_pnl)
                                
            trade_profit_factor_mtm = float(gross_trade_profit_mtm / gross_trade_loss_mtm) if gross_trade_loss_mtm != 0 else np.nan
            if gross_trade_loss_mtm == 0 and gross_trade_profit_mtm > 0:
                trade_profit_factor_mtm = np.inf

            trade_win_rate = float(trade_wins / (trade_wins + trade_losses)) if (trade_wins + trade_losses) > 0 else np.nan
            trade_avg_win = float(gross_trade_profit / trade_wins) if trade_wins > 0 else 0.0
            trade_avg_loss = float(gross_trade_loss / trade_losses) if trade_losses > 0 else 1e-9
            trade_win_loss_ratio = float(trade_avg_win / trade_avg_loss) if trade_avg_loss > 0 else np.nan
        else:
            trade_win_rate = np.nan
            trade_win_loss_ratio = np.nan
            trade_profit_factor_mtm = np.nan
        
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
        
        # EXCLUDE the final block for the AVERAGE calculation if it is unfinished 
        # (i.e. the strategy is still in a drawdown at the end of the timeseries)
        if len(is_under_water) > 0 and is_under_water.iloc[-1]:
            last_block_id = under_water_blocks.iloc[-1]
            if last_block_id in under_water_only.index:
                under_water_only = under_water_only[under_water_only.index != last_block_id]
                
        avg_time_under_water = float(under_water_only.mean()) if not under_water_only.empty else 0
        
        days_below_initial_capital = int((cum_ret < 1.0).sum())

        max_daily_drop = float(returns.min())
        
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
                full_nav = self.daily_amount[nav_col].astype(float)
                # Use Average NAV: (NAV_t + NAV_{t-1}) / 2
                avg_nav = (full_nav + full_nav.shift(1).fillna(full_nav)) / 2
                daily_avg_nav = avg_nav.loc[returns.index]
                
                # align indices
                aligned = pd.concat([daily_avg_nav, daily_trade], axis=1).fillna(0)
                aligned.columns = ['NAV', 'Trade_Amount']
                # Avoid div by zero
                aligned['NAV'] = aligned['NAV'].replace(0, 1e-9)
                # Note: Trade_Amount integrates both buy and sell legs, effectively doubling execution value.
                # Standard Turnover implies dividing the double-side volume by 2 to represent a 1-way equivalent.
                daily_turnover = (aligned['Trade_Amount'] / 2) / aligned['NAV']
                turnover_annual = float(daily_turnover.mean() * self.periods_per_year)
        
        return {
            'Absolute_Return': abs_ret,
            'Benchmark_Absolute_Return': float(benchmark_abs_ret),
            'Portfolio_Arithmetic_Annual_Return': portfolio_arith_annual,
            'CAGR_252': float(cagr_252),
            'CAGR_Calendar': float(cagr_calendar),
            'Benchmark_CAGR_252': float(benchmark_cagr_252),
            'Benchmark_CAGR_Calendar': float(benchmark_cagr_calendar),
            'Excess_Return_CAGR_252': float(excess_cagr_252),
            'Excess_Return_CAGR_Calendar': float(excess_cagr_calendar),
            'Volatility': float(volatility),
            'Benchmark_Volatility': float(benchmark_volatility),
            'Tracking_Error': float(tracking_error),
            'Sharpe': float(sharpe),
            'Benchmark_Sharpe': float(benchmark_sharpe),
            'Sortino': float(sortino),
            'Calmar': float(calmar),
            'Benchmark_Calmar': float(benchmark_calmar),
            'Information_Ratio_(Arithmetic)': float(information_ratio_arithmetic),
            'Information_Ratio_(Log_Based)': float(information_ratio_log),
            'Tracking_Error_(Geometric)': float(tracking_error_geometric),
            'Annualized_Active_Return_(Arithmetic)': float(annualized_active_return_arithmetic),
            'Max_Drawdown': float(max_dd),
            'Benchmark_Max_Drawdown': float(benchmark_max_dd),
            'Max_Time_Under_Water_Days': float(max_time_under_water),
            'Benchmark_Max_Time_Under_Water_Days': float(benchmark_max_tuw),
            'Avg_Time_Under_Water_Days': float(avg_time_under_water),
            'Benchmark_Avg_Time_Under_Water_Days': float(benchmark_avg_tuw),
            'Days_Below_Initial_Capital': int(days_below_initial_capital),
            'Max_Daily_Drop': max_daily_drop,
            'Realized_Trade_Win_Rate': float(trade_win_rate),
            'Trade_Win/Loss_Ratio': float(trade_win_loss_ratio),
            'Daily_Return_Win_Rate': float(win_rate),
            'Daily_Return_Win/Loss_Ratio': float(win_loss_ratio),
            'Daily_Profit_Factor_(Returns)': float(daily_profit_factor_returns),
            'Daily_Profit_Factor_(PnL)': float(daily_profit_factor_pnl),
            'Trade_Profit_Factor': float(trade_profit_factor),
            'Trade_Profit_Factor_MTM': float(trade_profit_factor_mtm),
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
        
        # Priority: use pre-existing benchmark in daily_amount to guarantee correct date alignment
        if self.benchmark_col in self.daily_amount.columns:
            market_close = self.daily_amount[self.benchmark_col].astype(float)
            market_return_full = market_close.pct_change().dropna()
            # Align to our returns index specifically
            market_return = market_return_full.loc[market_return_full.index.intersection(returns.index)]
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
            
        # Deduct risk-free rate for true excess-return CAPM regression
        rf_daily = RISK_FREE_RATE_ANNUAL / self.periods_per_year
        market_ret_total = aligned['Market'].copy()
        # Capture the aligned portfolio total return BEFORE subtracting rf
        aligned_portfolio_total = aligned['Portfolio'].copy()
        aligned['Portfolio'] = aligned['Portfolio'] - rf_daily
        aligned['Market'] = aligned['Market'] - rf_daily
            
        X = sm.add_constant(aligned['Market'])
        model = sm.OLS(aligned['Portfolio'], X).fit()
        
        # OLS intercept is an arithmetic daily mean; annualize arithmetically
        # to stay consistent with the multi-factor model.
        alpha = model.params['const'] * self.periods_per_year
        beta = model.params['Market']
        
        # OLS inference statistics for alpha and beta
        # t-stat for alpha is scale-invariant (daily or annualized gives same t),
        # so we use the raw daily intercept's t-stat directly.
        alpha_tstat = float(model.tvalues['const'])
        beta_tstat  = float(model.tvalues['Market'])
        alpha_pvalue = float(model.pvalues['const'])
        beta_pvalue  = float(model.pvalues['Market'])
        
        return {
            'Beta_Market': float(beta),
            'Beta_Market_t': beta_tstat,
            'Beta_Market_p': beta_pvalue,
            'Annualized_Alpha': float(alpha),
            'Annualized_Alpha_t': alpha_tstat,
            'Annualized_Alpha_p': alpha_pvalue,
            'R_Squared': float(model.rsquared),
            'Market_Total_Return_Annualized': float(market_ret_total.mean() * self.periods_per_year),
            'Market_Excess_Return_Annualized': float(aligned['Market'].mean() * self.periods_per_year),
            'Portfolio_Arithmetic_Annual_Return': float(aligned_portfolio_total.mean() * self.periods_per_year),
            'Aligned_Sample_Size': len(aligned)
        }

    def calculate_style_exposures(self, market=None):
        """
        Regress daily returns against proxy style factors (Liquidity, Momentum, Volatility).
        Factor values are lagged by 1 day (T-1) to avoid lookahead bias.
        Note: 'Liquidity' uses log(close*volume) as a proxy; this is turnover/amount,
        not market capitalization (which requires total_shares data).
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
        
        # Calculate daily returns first (no lag needed, this is the dependent variable)
        features['prev_close'] = features.groupby('instrument')['close'].shift(1)
        features['ret'] = (features['close'] - features['prev_close']) / features['prev_close']
        
        # Factor values use T-1 data (shift(1)) to avoid lookahead bias:
        # We use yesterday's factor scores to explain today's returns.
        features['liquidity'] = np.log(features['close'] * features['volume'] + 1e-9)
        features['liquidity'] = features.groupby('instrument')['liquidity'].shift(1)
        features['momentum'] = features.groupby('instrument')['close'].pct_change(20).shift(1)
        features['volatility'] = features.groupby('instrument')['ret'].rolling(20, min_periods=5).std().reset_index(0, drop=True)
        features['volatility'] = features.groupby('instrument')['volatility'].shift(1)
        
        features = features.dropna(subset=['ret', 'liquidity', 'momentum', 'volatility'])
        
        factor_returns = {}
        for factor in ['liquidity', 'momentum', 'volatility']:
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
        
        # We need Market Return. 
        if self.benchmark_col in self.daily_amount.columns:
            market_close = self.daily_amount[self.benchmark_col].astype(float)
            market_return_full = market_close.pct_change().dropna()
            market_return = market_return_full.loc[market_return_full.index.intersection(returns.index)]
        else:
            market_return = features.groupby('datetime')['ret'].mean()
            
        aligned = pd.concat([returns, market_return, factor_df], axis=1).dropna()
        aligned.columns = ['Portfolio', 'Market'] + list(factor_df.columns)
        
        if len(aligned) < 2:
            return {}
        
        # Calculate annualized returns for each factor FROM THE ALIGNED DATA.
        # Critical: Factor_Annualized must use the same sample as the regression,
        # otherwise the OLS identity E(Y) = β₀ + Σβᵢ*E(Xᵢ) will NOT hold.
        # Note: Zero-investment long-short factor returns cannot be geometrically compounded (CAGR).
        # We must use arithmetic annualized returns (mean daily return * 252) mathematically.
        factor_annualized = {}
        for col in factor_df.columns:
            factor_annualized[col] = float(aligned[col].mean() * self.periods_per_year)
            
        # Deduct risk-free rate from Portfolio and Market for Excess-Return multi-factor regression
        rf_daily = RISK_FREE_RATE_ANNUAL / self.periods_per_year
        market_ret_total = aligned['Market'].copy()
        # Capture the aligned portfolio total return BEFORE subtracting rf
        aligned_portfolio_total = aligned['Portfolio'].copy()
        aligned['Portfolio'] = aligned['Portfolio'] - rf_daily
        aligned['Market'] = aligned['Market'] - rf_daily
            
        X = sm.add_constant(aligned[['Market', 'liquidity', 'momentum', 'volatility']])
        model = sm.OLS(aligned.iloc[:, 0], X).fit()
        
        # OLS inference statistics for multi-factor model
        mf_alpha_tstat  = float(model.tvalues.get('const', float('nan')))
        mf_beta_tstat   = float(model.tvalues.get('Market', float('nan')))
        mf_alpha_pvalue = float(model.pvalues.get('const', float('nan')))
        mf_beta_pvalue  = float(model.pvalues.get('Market', float('nan')))
        
        return {
            'Multi_Factor_Intercept': float(model.params.get('const', 0)) * self.periods_per_year,
            'Multi_Factor_Intercept_t': mf_alpha_tstat,
            'Multi_Factor_Intercept_p': mf_alpha_pvalue,
            'Multi_Factor_Beta': float(model.params.get('Market', 0)),
            'Multi_Factor_Beta_t': mf_beta_tstat,
            'Multi_Factor_Beta_p': mf_beta_pvalue,
            BARRA_LIQD_KEY: float(model.params.get('liquidity', 0)),
            BARRA_MOMT_KEY: float(model.params.get('momentum', 0)),
            BARRA_VOLA_KEY: float(model.params.get('volatility', 0)),
            'Barra_Style_R_Squared': float(model.rsquared),
            'Factor_Annualized': factor_annualized,
            'Market_Total_Return_Annualized': float(market_ret_total.mean() * self.periods_per_year),
            'Market_Excess_Return_Annualized': float(aligned['Market'].mean() * self.periods_per_year),
            'Portfolio_Arithmetic_Annual_Return': float(aligned_portfolio_total.mean() * self.periods_per_year),
            'Aligned_Sample_Size': len(aligned)
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
            
        # Cache full portfolio daily NAVs (including CASH) for accurate concentration denominators
        full_daily_navs = self.holding_log.groupby('成交日期')['收盘价值'].sum()
        
        # Exclude CASH
        df = self.holding_log[self.holding_log['证券代码'] != 'CASH'].copy()
        if df.empty:
            return {}
            
        # Group by date
        daily_groups = df.groupby('成交日期')
        
        # 1. Avg count
        daily_count = daily_groups.size()
        avg_count = float(daily_count.mean())
        
        # 2. Avg top 1 concentration (against total portfolio NAV including CASH)
        def top1_conc(g):
            date = g.name
            total_val = full_daily_navs.get(date, g['收盘价值'].sum())
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
