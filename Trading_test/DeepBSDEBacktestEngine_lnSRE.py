import pandas as pd
import numpy as np
import torch
import scipy.optimize as sco
import matplotlib.pyplot as plt
import warnings
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from io import StringIO
import math
warnings.filterwarnings("ignore")

@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: pd.Timestamp
    history_window: pd.DataFrame
    vm_lagged: float
    total_equity: float
    lagged_residual: np.ndarray

class BenchmarkCalculator:
    @staticmethod
    def compute_ew(assets: List[str]) -> Dict[str, float]:
        n = len(assets)
        return {a: 1.0/n for a in assets}

    @staticmethod
    def compute_iv(history: pd.DataFrame, assets: List[str]) -> Dict[str, float]:
        window = 20
        history = history.iloc[-window:]
        variances = history.var().values * 252.0
        inv_vars = 1.0 / variances
        w = inv_vars / np.sum(inv_vars)
        return dict(zip(assets, w))

    @staticmethod
    def compute_gmv(history: pd.DataFrame, assets: List[str], allow_leverage: bool) -> Dict[str, float]:
        window = 20
        history = history.iloc[-window:]
        n = len(assets)
        cov = history.cov().values
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(n)
        w = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
        if not allow_leverage:
            s = np.sum(np.abs(w))
            if s > 1.0: w /= s
        return dict(zip(assets, w))

class BacktestEngine:
    def __init__(self, assets=['xa', 'xb', 'xc', 'xd']):
        self.assets = assets
        self.results = {}
        self.raw_data = None
        self.trade_logs = []
        self.config = {}
        self.lagged_residual = np.zeros(len(assets), dtype=np.float64)

    def load_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                start = 0
                for i, l in enumerate(lines):
                    if l.strip() == "Data:": start = i + 1; break
            df = pd.read_csv(StringIO("".join(lines[start:])))
            df.columns = [c.strip() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            req = self.assets + ['VM']
            if not all(c in df.columns for c in req):
                raise ValueError(f"Missing columns. Found: {df.columns}")
            
            self.raw_data = df
            print(f"[Engine] Data Loaded. Rows: {len(df)}")
            
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            raise

    def run_backtest(self, strategy, start_date, end_date, 
                     target_return, initial_capital, rebalance_period, strategy_window=20,
                     allow_leverage=False, include_individual=False):
        
        self.config = {
            'start_date': start_date,
            'end_date': end_date,
            'strategy_window': strategy_window,
            'rebalance_period': rebalance_period,
            'allow_leverage': allow_leverage
        }
        
        self.trade_logs = []
        df = self.raw_data.copy()
        
        if end_date: df = df[df.index <= pd.to_datetime(end_date)]
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            valid_indices = np.where(df.index >= start_dt)[0]
            if len(valid_indices) == 0:
                raise ValueError(f"No data available on or after {start_date}")
            start_idx = valid_indices[0]
        else:
            start_idx = strategy_window
            
        if start_idx < strategy_window:
            raise ValueError(f"Not enough history data ({strategy_window} days) before absolute start_date {start_date}.")
        
        core_cols = self.assets + ['VM']
        if df.iloc[start_idx-strategy_window:][core_cols].isnull().values.any():
            nan_rows = df.iloc[start_idx-strategy_window:][df.iloc[start_idx-strategy_window:][core_cols].isnull().any(axis=1)]
            raise ValueError(f"CRITICAL: NaN detected. Bad Dates: {nan_rows.index.tolist()}")
        if len(df) < start_idx + 2:
            raise ValueError(f"Data length insufficient for backtest starting at {start_date}.")
        
        start_date_disp = df.index[start_idx].date()
        print(f"\n[Start]")
        print(f" Data Start Date : {start_date_disp}")
        print(f" Strategy Window : {strategy_window}")
        print(f" Allow Leverage  : {allow_leverage}")
        print("-" * 40)
        
        strat_names = [f'Affine_{target_return:.1%}', 'EW', 'IV', 'GMV']
        if include_individual: strat_names += self.assets
        
        strategies = {}
        for s in strat_names:
            strategies[s] = {
                'cash': initial_capital,
                'holdings': {a: 0.0 for a in self.assets},
                'history': [initial_capital],
                'weights': []
            }
            if s in self.assets:
                strategies[s]['cash'] = 0.0
                strategies[s]['holdings'][s] = initial_capital
        
        n_steps = len(df)
        
        for t in range(start_idx, n_steps - 1):
            
            strat_history_slice = df.iloc[t-strategy_window : t][self.assets]
            bench_history_slice = df.iloc[:t][self.assets]
            
            vm_lagged = df.iloc[t-1]['VM']
            curr_date = df.index[t]
            
            is_rebalance = ((t - start_idx) % rebalance_period == 0)
            time_remain = strategy.T - (t - start_idx) * strategy.dt
            
            next_vals_arr = df.iloc[t+1][self.assets].values.astype(np.float64)
            drift_val = strategy.r_f * strategy.dt
            daily_residual = next_vals_arr - drift_val
            
            if is_rebalance:
                aff_key = f'Affine_{target_return:.1%}'
                aff_eq = strategies[aff_key]['cash'] + sum(strategies[aff_key]['holdings'].values())
                
                snapshot = MarketSnapshot(
                    timestamp=curr_date,
                    history_window=strat_history_slice,
                    vm_lagged=vm_lagged,
                    total_equity=aff_eq,
                    lagged_residual=self.lagged_residual.copy()
                )
                
                w_aff = strategy.calculate_weights(snapshot, target_return, initial_capital,
                                                   allow_leverage, time_remain)
                self._execute_rebalance(curr_date, strategies[aff_key], w_aff, aff_key)
                
                self.lagged_residual = daily_residual.copy()
                
                w_ew = BenchmarkCalculator.compute_ew(self.assets)
                self._execute_rebalance(curr_date, strategies['EW'], w_ew, 'EW')
                
                w_iv = BenchmarkCalculator.compute_iv(bench_history_slice, self.assets)
                self._execute_rebalance(curr_date, strategies['IV'], w_iv, 'IV')
                
                w_gmv = BenchmarkCalculator.compute_gmv(bench_history_slice, self.assets, allow_leverage)
                self._execute_rebalance(curr_date, strategies['GMV'], w_gmv, 'GMV')
                
            else:
                for s in strategies:
                    if s in self.assets: continue
                    last_w = strategies[s]['weights'][-1] if strategies[s]['weights'] else [0.0]*len(self.assets)
                    strategies[s]['weights'].append(last_w)
                
                self.lagged_residual += daily_residual
            
            next_vals = df.iloc[t+1][self.assets]
            rf_mult = 1.0 + strategy.r_f * strategy.dt
            
            for s_name, acct in strategies.items():
                acct['cash'] *= rf_mult
                curr_h_val = 0.0
                for a in self.assets:
                    r = next_vals[a]
                    mult = np.exp(r)
                    acct['holdings'][a] *= mult
                    curr_h_val += acct['holdings'][a]
                
                nav = acct['cash'] + curr_h_val
                if nav <= 0: nav = 1e-9
                acct['history'].append(nav)
                
        self.results = {
            'dates': df.index[start_idx:],
            'w_affine_dict': {}
        }
        
        for s_name, acct in strategies.items():
            self.results[s_name] = np.array(acct['history'])
            if s_name not in self.assets:
                w_arr = np.array(acct['weights'])
                if s_name.startswith('Affine'):
                    self.results['w_affine_dict'][s_name] = w_arr
                else:
                    self.results[f'w_{s_name}'] = w_arr
                     
        
        full_strategy_list = []
        for k in strategies.keys():
            if k.startswith('Affine'): full_strategy_list.append(k)
        full_strategy_list += ['EW', 'IV', 'GMV']
        if include_individual:
            full_strategy_list += self.assets
        self._print_risk_metrics(full_strategy_list)
        return self.results

    def _execute_rebalance(self, date, account, target_weights, name):
        w_list = [target_weights.get(a, 0.0) for a in self.assets]
        account['weights'].append(w_list)
        
        pre_holdings = account['holdings'].copy()
        pre_cash = account['cash']
        pre_total_eq = pre_cash + sum(pre_holdings.values())
        
        new_holdings = {}
        alloc_cash = 0.0
        
        details = []
        for a in self.assets:
            w = target_weights.get(a, 0.0)
            target_val = pre_total_eq * w
            
            old_val = pre_holdings.get(a, 0.0)
            aux = target_val - old_val
            
            new_holdings[a] = target_val
            alloc_cash += target_val
            
            details.append({
                'asset': a,
                'pre_val': old_val,
                'post_val': target_val,
                'aux': aux
            })
            
        new_cash = pre_total_eq - alloc_cash
        
        account['holdings'] = new_holdings
        account['cash'] = new_cash
        
        trade_record = {
            'date': date,
            'strategy': name,
            'total_equity': pre_total_eq,
            'pre_cash': pre_cash,
            'post_cash': new_cash,
            'details': details
        }
        self.trade_logs.append(trade_record)

    def view_transaction_logs(self, strategy_name=None):
        if not self.trade_logs:
            print("[Info] No logs.")
            return
        print(f"\n" + "="*80)
        print(f"Log Detail | Strategy: {strategy_name if strategy_name else 'ALL'}")
        print("="*80)
        count = 0
        for record in self.trade_logs:
            if strategy_name and record['strategy'] != strategy_name:
                continue
            
            has_changes = any(abs(d['aux']) > 0.01 for d in record['details'])
            if not has_changes and not strategy_name: continue
            total_eq = record['total_equity']
            if total_eq < 1e-6: total_eq = 1e-6
            print(f"\n[Date] {record['date'].date()} | [Strategy] {record['strategy']}")
            print(f"{'Item':<8} | {'Pre-Val':<12} | {'Post-Val':<12} | {'Post-%':<7} | {'Change':<10}")
            print("-" * 58)
            
            print(f"{'Equity':<8} | {total_eq:<12.2f} | {total_eq:<12.2f} | {'100%':<7} | {'0.00':<10}")
            
            cash_diff = record['post_cash'] - record['pre_cash']
            cash_pct = (record['post_cash'] / total_eq) * 100.0
            print(f"{'Cash':<8} | {record['pre_cash']:<12.2f} | {record['post_cash']:<12.2f} | {cash_pct:>6.1f}% | {cash_diff:<+10.2f}")
            
            print("-" * 58)
            print(f"{'Asset':<8} | {'Pre-Val':<12} | {'Post-Val':<12} | {'Post-%':<7} | {'Aux':<10}")
            for d in record['details']:
                pct = (d['post_val'] / total_eq) * 100.0
                print(f"{d['asset']:<8} | {d['pre_val']:>12.2f} | {d['post_val']:>12.2f} | {pct:>6.1f}% | {d['aux']:>+10.2f}")
            
            count += 1
            
        if count == 0:
            print(f"[Info] No trades for: {strategy_name}")
        else:
            print(f"\nTotal Records: {count}")

    def _print_risk_metrics(self, strategy_list):
        print(f"\n[Metrics]")
        header = f"{'Strategy':<15} | {'Return':<8} | {'Vol':<8} | {'Sharpe':<8} | {'Sortino':<8} | {'Calmar':<8} | {'MDD':<8} | {'RT':<6} | {'MES(5%)':<8}"
        print(header)
        print("-" * len(header))
        
        dates = self.results['dates']
        
        tail_mask = None
        if 'EW' in self.results:
            market_arr = self.results['EW']
            market_nav = pd.Series(market_arr, index=dates)
            market_log_ret = np.log(market_nav / market_nav.shift(1)).dropna()
            if len(market_log_ret) > 0:
                threshold_5 = np.percentile(market_log_ret, 5)
                tail_mask = market_log_ret <= threshold_5
        
        for name in strategy_list:
            if name not in self.results: continue
            arr = self.results[name]
            if len(arr) < 2: continue
            
            nav_series = pd.Series(arr, index=dates)
            
            start_val, end_val = nav_series.iloc[0], nav_series.iloc[-1]
            years = len(nav_series) / 252.0
            ann_ret = np.log(end_val / start_val) / years if start_val > 0 and end_val > 0 else 0.0
            daily_log_ret = np.log(nav_series / nav_series.shift(1)).dropna()
            ann_vol = daily_log_ret.std() * np.sqrt(252) if len(daily_log_ret) > 1 else 0.0
            sharpe = (ann_ret / ann_vol) if ann_vol > 1e-6 else 0.0
            
            downside_sq = np.minimum(0, daily_log_ret)**2
            downside_dev = np.sqrt(np.mean(downside_sq)) * np.sqrt(252)
            sortino = (ann_ret / downside_dev) if downside_dev > 1e-6 else 0.0
            running_max = nav_series.cummax()
            drawdown = (nav_series - running_max) / running_max
            max_dd = drawdown.min()
            is_underwater = (nav_series < running_max)
            current_dd_duration = max_dd_duration = 0
            for u in is_underwater:
                if u: current_dd_duration += 1
                else:
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                    current_dd_duration = 0
            rt_days = max(max_dd_duration, current_dd_duration)
            calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
            
            mes_val = 0.0
            if tail_mask is not None:
                if len(daily_log_ret) == len(tail_mask):
                    mes_val = daily_log_ret.values[tail_mask.values].mean()
            print(f"{name:<15} | {ann_ret:>8.2%} | {ann_vol:>8.2%} | {sharpe:>8.2f} | {sortino:>8.2f} | {calmar:>8.2f} | {max_dd:>8.2%} | {rt_days:>6d} | {mes_val:>8.2%}")

    def plot_comparison(self):
        if not self.results: return
        res = self.results
        dates = res['dates']
        plt.figure(figsize=(12, 6))
        
        aff_keys = sorted([k for k in res.keys() if k.startswith('Affine_')])
        colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(aff_keys)))
        for i, key in enumerate(aff_keys):
            plt.plot(dates, res[key], label=key, color=colors[i], linewidth=2)
            
        if 'EW' in res: plt.plot(dates, res['EW'], label='Equal Weight', color='gray', linestyle='--', alpha=0.6)
        if 'IV' in res: plt.plot(dates, res['IV'], label='Inverse Variance', color='orange', linestyle='-.', alpha=0.6)
        if 'GMV' in res: plt.plot(dates, res['GMV'], label='Global Min Var', color='green', linestyle=':', linewidth=2)
            
        plt.title('Strategy Comparison')
        plt.ylabel('Wealth'); plt.xlabel('Date'); plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
        plt.show()

    def plot_asset_allocation(self, strategy_name=None):
        if not self.results or 'w_affine_dict' not in self.results: return
        w_dict = self.results['w_affine_dict']
        
        if strategy_name is None:
            keys = list(w_dict.keys())
            if keys: strategy_name = keys[0]
            else: return
        
        weights = w_dict.get(strategy_name)
        if weights is None: return
        
        dates = self.results['dates']
        min_len = min(len(weights), len(dates))
        weights = weights[:min_len]
        dates = dates[:min_len]
        plt.figure(figsize=(12, 6))
        assets, colors = self.assets, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bottom_pos = np.zeros(len(weights))
        bottom_neg = np.zeros(len(weights))
        
        for i, asset in enumerate(assets):
            w = weights[:, i]
            c = colors[i % len(colors)]
            
            pos_w = np.maximum(w, 0)
            plt.bar(dates, pos_w, bottom=bottom_pos, label=f'Asset {asset}', color=c, alpha=0.8, width=1.0)
            bottom_pos += pos_w
            
            neg_w = np.minimum(w, 0)
            plt.bar(dates, neg_w, bottom=bottom_neg, color=c, alpha=0.8, width=1.0)
            bottom_neg += neg_w
        
        total_abs = np.sum(np.abs(weights), axis=1)
        plt.plot(dates, total_abs, label='Total Inv', color='black', linestyle='--', linewidth=2)
        
        plt.title(f'Allocation: {strategy_name}')
        plt.ylabel('Weight')
        plt.xlabel('Date')
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def export_trades_to_csv(self, file_path: str):
        if not self.results or 'dates' not in self.results:
            print("[Error] No results."); return
        
        start_ts = self.results['dates'][0]
        end_ts = self.results['dates'][-1]
        
        print(f"\n[Export]")
        print(f" Interval : {start_ts.date()} -> {end_ts.date()}")
        
        strat_win = self.config.get('strategy_window', 60)
        reb = self.config.get('rebalance_period', 5)
        allow_l = self.config.get('allow_leverage', False)
        cfg_start = self.config.get('start_date', str(start_ts.date()))
        cfg_end = self.config.get('end_date', str(end_ts.date()))
        
        flat = []
        benchmarks = ['EW', 'IV', 'GMV']
        
        if self.trade_logs:
            for rec in self.trade_logs:
                type_label = "Benchmark" if rec['strategy'] in benchmarks else "Control"
                for d in rec['details']:
                    flat.append({
                        'Date': rec['date'], 'Strategy': rec['strategy'], 'Type': type_label,
                        'Total_Equity': rec['total_equity'], 'Post_Cash': rec['post_cash'],
                        'Asset': d['asset'], 'Post_Val': d['post_val'], 'Aux': d['aux'],
                        'Is_End_Marker': False, 'Strategy_Window': strat_win, 'Rebalance_Period': reb,
                        'Allow_Leverage': allow_l,
                        'Start_Date': cfg_start, 'End_Date': cfg_end
                    })
        
        all_keys = [k for k in self.results.keys() if k not in ['dates', 'w_affine_dict']]
        for s_name in all_keys:
            data_array = np.array(self.results[s_name])
            if data_array.ndim == 1:
                last_val = data_array[-1]
                val_scalar = last_val.item() if hasattr(last_val, 'item') else float(last_val)
                type_label = "Benchmark" if s_name in benchmarks else "Control"
                
                flat.append({
                    'Date': end_ts, 'Strategy': s_name, 'Type': type_label,
                    'Total_Equity': val_scalar, 'Post_Cash': 0.0, 'Asset': 'END_MARKER',
                    'Post_Val': 0.0, 'Aux': 0.0, 'Is_End_Marker': True,
                    'Strategy_Window': strat_win, 'Rebalance_Period': reb,
                    'Allow_Leverage': allow_l,
                    'Start_Date': cfg_start, 'End_Date': cfg_end
                })
        pd.DataFrame(flat).to_csv(file_path, index=False)
        print(f"[Saved] {file_path}")
