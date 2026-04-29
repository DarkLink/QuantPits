"""
Coordinator for the MAS Deep Analysis System.

Orchestrates data discovery, window generation, agent dispatch, and result collection.
Scans both active workspace and archive directories for dated files.
"""

import os
import re
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple

from .base_agent import AnalysisContext, AgentFindings, BaseAgent


# ---------------------------------------------------------------------------
# Date extraction patterns (mirrors archive_dated_files.py)
# ---------------------------------------------------------------------------
DATE_PATTERN_SUFFIX = re.compile(r'_(\d{4}-\d{2}-\d{2})(?:_(\d{6}))?')
DATE_PATTERN_PREFIX = re.compile(r'^(\d{4}-\d{2}-\d{2})-')


def _extract_date(filename: str) -> Optional[str]:
    """Extract YYYY-MM-DD date string from a filename."""
    m = DATE_PATTERN_PREFIX.match(filename)
    if m:
        return m.group(1)
    m = DATE_PATTERN_SUFFIX.search(filename)
    if m:
        return m.group(1)
    return None


def _scan_dated_files(directories: List[str], pattern_prefix: str,
                      extension: str = '.csv') -> List[Tuple[str, str]]:
    """
    Scan multiple directories for files matching a prefix pattern with dates.
    
    Returns list of (file_path, date_str), sorted by date.
    """
    results = {}  # date -> path (later dates override for dedup)

    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            if not entry.endswith(extension):
                continue
            if not entry.startswith(pattern_prefix):
                # Also check for ensemble-style names: buy_suggestion_ensemble_2026-04-20.csv
                if pattern_prefix not in entry:
                    continue
            date_str = _extract_date(entry)
            if date_str:
                full_path = os.path.join(directory, entry)
                # Prefer active workspace over archive (later in list overrides)
                results[date_str] = full_path

    return sorted(results.items(), key=lambda x: x[0])


def _filter_files_by_window(files: List[Tuple[str, str]],
                            start_date: str, end_date: str) -> List[str]:
    """Filter (date, path) tuples to those within the date range."""
    return [path for date_str, path in files
            if start_date <= date_str <= end_date]


class Coordinator:
    """
    Main orchestrator for the MAS Deep Analysis System.
    
    Responsibilities:
    1. Discover data files across workspace and archive
    2. Generate analysis time windows
    3. Load shared data (CSVs loaded once, sliced per window)
    4. Dispatch agents and collect findings
    """

    def __init__(self, workspace_root: str,
                 freq_change_date: Optional[str] = None,
                 external_notes: str = "",
                 windows: Optional[List[str]] = None):
        self.workspace_root = workspace_root
        self.freq_change_date = freq_change_date
        self.external_notes = external_notes
        self.requested_windows = windows or ['full', 'weekly_era', '1y', '6m', '3m', '1m']

        # Core directories
        self.output_dir = os.path.join(workspace_root, 'output')
        self.data_dir = os.path.join(workspace_root, 'data')
        self.archive_dir = os.path.join(workspace_root, 'archive')
        self.config_dir = os.path.join(workspace_root, 'config')
        self.order_history_dir = os.path.join(self.data_dir, 'order_history')

        # Discovered data
        self._daily_amount_df = None
        self._trade_log_df = None
        self._holding_log_df = None
        self._trade_classification_df = None
        self._discovered_files = {}
        self._data_end_date = None
        self._data_start_date = None

    # ------------------------------------------------------------------
    # Data Discovery
    # ------------------------------------------------------------------

    def discover(self):
        """Scan workspace for all available data and determine date range."""
        print("📂 Discovering data files...")
        self._load_shared_dataframes()
        self._discover_dated_files()
        print(f"   Data range: {self._data_start_date} → {self._data_end_date}")
        print(f"   Freq change date: {self.freq_change_date or 'Not set'}")

    def _load_shared_dataframes(self):
        """Load the core CSV files that all agents share."""
        # Daily amount log
        daily_path = os.path.join(self.data_dir, 'daily_amount_log_full.csv')
        if os.path.exists(daily_path):
            self._daily_amount_df = pd.read_csv(daily_path)
            if '成交日期' in self._daily_amount_df.columns:
                self._daily_amount_df['成交日期'] = pd.to_datetime(self._daily_amount_df['成交日期'])
                dates = self._daily_amount_df['成交日期'].dropna()
                self._data_start_date = dates.min().strftime('%Y-%m-%d')
                self._data_end_date = dates.max().strftime('%Y-%m-%d')
        else:
            self._daily_amount_df = pd.DataFrame()

        # Trade log
        trade_path = os.path.join(self.data_dir, 'trade_log_full.csv')
        if os.path.exists(trade_path):
            self._trade_log_df = pd.read_csv(trade_path)
            if '成交日期' in self._trade_log_df.columns:
                self._trade_log_df['成交日期'] = pd.to_datetime(self._trade_log_df['成交日期'])
        else:
            self._trade_log_df = pd.DataFrame()

        # Holding log
        holding_path = os.path.join(self.data_dir, 'holding_log_full.csv')
        if os.path.exists(holding_path):
            self._holding_log_df = pd.read_csv(holding_path)
            if '成交日期' in self._holding_log_df.columns:
                self._holding_log_df['成交日期'] = pd.to_datetime(self._holding_log_df['成交日期'])
        else:
            self._holding_log_df = pd.DataFrame()

        # Trade classification
        class_path = os.path.join(self.data_dir, 'trade_classification.csv')
        if os.path.exists(class_path):
            self._trade_classification_df = pd.read_csv(class_path)
            if 'trade_date' in self._trade_classification_df.columns:
                self._trade_classification_df['trade_date'] = pd.to_datetime(
                    self._trade_classification_df['trade_date'])
        else:
            self._trade_classification_df = pd.DataFrame()

    def _discover_dated_files(self):
        """Discover all dated files across output/, archive/, and order_history/."""
        # Directories to scan for each file type
        output_dirs = [self.output_dir]
        ensemble_dirs = [
            os.path.join(self.output_dir, 'ensemble'),
            os.path.join(self.archive_dir, 'output', 'ensemble'),
        ]
        archive_output_dirs = [
            self.output_dir,
            os.path.join(self.archive_dir, 'output'),
        ]
        suggestion_dirs = [
            self.output_dir,
            self.order_history_dir,
            os.path.join(self.archive_dir, 'output'),
        ]

        self._discovered_files = {
            'model_performance': _scan_dated_files(
                archive_output_dirs, 'model_performance', '.json'),
            'combo_comparison': _scan_dated_files(
                ensemble_dirs, 'combo_comparison', '.csv'),
            'correlation_matrix': _scan_dated_files(
                ensemble_dirs, 'correlation_matrix', '.csv'),
            'model_opinions': _scan_dated_files(
                suggestion_dirs, 'model_opinions', '.json'),
            'buy_suggestion': _scan_dated_files(
                suggestion_dirs, 'buy_suggestion', '.csv'),
            'sell_suggestion': _scan_dated_files(
                suggestion_dirs, 'sell_suggestion', '.csv'),
            'model_contribution': _scan_dated_files(
                ensemble_dirs, 'model_contribution', '.json'),
        }

        # Leaderboard files: leaderboard_{combo}_{date}.csv
        self._discovered_files['leaderboard'] = _scan_dated_files(
            ensemble_dirs, 'leaderboard', '.csv')

        # Ensemble fusion config files: ensemble_fusion_config_{combo}_{date}.json
        self._discovered_files['ensemble_fusion_config'] = _scan_dated_files(
            ensemble_dirs, 'ensemble_fusion_config', '.json')

        for key, files in self._discovered_files.items():
            if files:
                print(f"   {key}: {len(files)} files ({files[0][0]} → {files[-1][0]})")

    # ------------------------------------------------------------------
    # Window Generation
    # ------------------------------------------------------------------

    def generate_windows(self) -> List[dict]:
        """
        Generate analysis time windows based on available data.
        
        Returns list of dicts: {label, start_date, end_date, is_pre_cutoff}
        """
        if not self._data_end_date:
            print("⚠️  No data found. Cannot generate windows.")
            return []

        end = pd.Timestamp(self._data_end_date)
        start_full = pd.Timestamp(self._data_start_date)

        windows = []
        window_specs = {
            'full': None,  # Uses data_start_date
            'weekly_era': self.freq_change_date,
            '1y': (end - relativedelta(years=1)).strftime('%Y-%m-%d'),
            '6m': (end - relativedelta(months=6)).strftime('%Y-%m-%d'),
            '3m': (end - relativedelta(months=3)).strftime('%Y-%m-%d'),
            '1m': (end - relativedelta(months=1)).strftime('%Y-%m-%d'),
        }

        for label in self.requested_windows:
            if label == 'full':
                windows.append({
                    'label': 'full',
                    'start_date': self._data_start_date,
                    'end_date': self._data_end_date,
                    'is_pre_cutoff': False,
                })
            elif label == 'weekly_era':
                if self.freq_change_date:
                    windows.append({
                        'label': 'weekly_era',
                        'start_date': self.freq_change_date,
                        'end_date': self._data_end_date,
                        'is_pre_cutoff': False,
                    })
                # Skip silently if no freq_change_date
            elif label in window_specs:
                w_start = window_specs[label]
                if w_start:
                    # Clamp to data start
                    if w_start < self._data_start_date:
                        w_start = self._data_start_date
                    is_pre = False
                    if self.freq_change_date and w_start < self.freq_change_date:
                        is_pre = True
                    windows.append({
                        'label': label,
                        'start_date': w_start,
                        'end_date': self._data_end_date,
                        'is_pre_cutoff': is_pre,
                    })

        print(f"\n📊 Generated {len(windows)} analysis windows:")
        for w in windows:
            flag = " ⚠️ (contains pre-cutoff data)" if w['is_pre_cutoff'] else ""
            print(f"   [{w['label']}] {w['start_date']} → {w['end_date']}{flag}")

        return windows

    # ------------------------------------------------------------------
    # Context Building
    # ------------------------------------------------------------------

    def build_context(self, window: dict) -> AnalysisContext:
        """Build an AnalysisContext for a specific time window."""
        start = pd.Timestamp(window['start_date'])
        end = pd.Timestamp(window['end_date'])

        # Slice shared DataFrames
        def _slice(df, date_col='成交日期'):
            if df is None or df.empty:
                return pd.DataFrame()
            if date_col in df.columns:
                return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
            return df.copy()

        def _slice_class(df):
            if df is None or df.empty:
                return pd.DataFrame()
            if 'trade_date' in df.columns:
                return df[(df['trade_date'] >= start) & (df['trade_date'] <= end)].copy()
            return df.copy()

        # Filter discovered files by window date range
        def _filter(key):
            files = self._discovered_files.get(key, [])
            return _filter_files_by_window(files, window['start_date'], window['end_date'])

        return AnalysisContext(
            start_date=window['start_date'],
            end_date=window['end_date'],
            window_label=window['label'],
            workspace_root=self.workspace_root,
            external_notes=self.external_notes,
            daily_amount_df=_slice(self._daily_amount_df),
            trade_log_df=_slice(self._trade_log_df),
            holding_log_df=_slice(self._holding_log_df),
            trade_classification_df=_slice_class(self._trade_classification_df),
            model_performance_files=_filter('model_performance'),
            ensemble_fusion_config_files=_filter('ensemble_fusion_config'),
            combo_comparison_files=_filter('combo_comparison'),
            leaderboard_files=_filter('leaderboard'),
            correlation_matrix_files=_filter('correlation_matrix'),
            model_opinions_files=_filter('model_opinions'),
            buy_suggestion_files=_filter('buy_suggestion'),
            sell_suggestion_files=_filter('sell_suggestion'),
            model_contribution_files=_filter('model_contribution'),
            freq_change_date=self.freq_change_date,
            is_pre_cutoff_window=window.get('is_pre_cutoff', False),
        )

    # ------------------------------------------------------------------
    # Agent Dispatch
    # ------------------------------------------------------------------

    def run(self, agents: List[BaseAgent]) -> List[AgentFindings]:
        """
        Execute the full analysis pipeline.
        
        1. Discover data
        2. Generate windows
        3. For each window, build context and run all agents
        4. Return all findings
        """
        self.discover()
        windows = self.generate_windows()
        if not windows:
            return []

        all_findings = []

        for window in windows:
            ctx = self.build_context(window)
            print(f"\n{'='*60}")
            print(f"  Window: [{window['label']}] {window['start_date']} → {window['end_date']}")
            print(f"{'='*60}")

            for agent in agents:
                try:
                    print(f"  🔍 Running {agent.name}...")
                    findings = agent.analyze(ctx)
                    all_findings.append(findings)
                    n_findings = len(findings.findings)
                    n_critical = sum(1 for f in findings.findings if f.severity == 'critical')
                    n_warning = sum(1 for f in findings.findings if f.severity == 'warning')
                    print(f"     → {n_findings} findings "
                          f"({n_critical} critical, {n_warning} warning)")
                except Exception as e:
                    print(f"  ❌ {agent.name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create empty findings for failed agent
                    all_findings.append(AgentFindings(
                        agent_name=agent.name,
                        window_label=window['label'],
                        findings=[],
                        recommendations=[],
                        raw_metrics={'error': str(e)},
                    ))

        return all_findings
