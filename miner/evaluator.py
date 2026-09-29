"""
evaluator.py - High-Performance Real-Data GPU Tensor Evaluation Engine for Futures.

Maintains a persistent, cached GlobalTensorContext in GPU VRAM spanning 1-Year,
runs batch factors concurrently on PyTorch DAG graph scheduler, and cleans dynamic VRAM
at the end of each iteration cycle to prevent leaks and OOMs.
"""

from __future__ import annotations

import os
import sys
import torch
import gc
import logging
import pandas as pd
import numpy as np

# ==============================================================================
# 🐒 [Monkey-Patch] Intercept pd.read_parquet to clean non-numeric columns
# ==============================================================================
_original_read_parquet = pd.read_parquet

def _monkey_patched_read_parquet(*args, **kwargs):
    df = _original_read_parquet(*args, **kwargs)
    if isinstance(df, pd.DataFrame):
        # Core metadata columns we MUST keep for future_system processing
        keep_metadata = {'contract', 'trading_date', 'main_contract'}

        # Select columns to keep:
        # - Core metadata columns
        # - Any numeric column (floats, ints, booleans)
        # - Any datetime/timestamp column (e.g., index_time)
        cols_to_keep = [
            col for col in df.columns
            if col in keep_metadata
            or pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_datetime64_any_dtype(df[col])
        ]

        if len(cols_to_keep) < len(df.columns):
            df = df[cols_to_keep]

    return df

# Apply the global monkey patch
pd.read_parquet = _monkey_patched_read_parquet
# ==============================================================================
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Add external projects to system path to ensure absolute imports resolve
project_root = Path(__file__).resolve().parent.parent

# Imports from future_sim package (installed in Pegasus future_research)
from tensor_context import GlobalTensorContext
from dag_scheduler import DAGManager
from batch_executor import BatchExecutor


class CPUAlphaProfiler:
    """
    100% Vectorized, High-Performance CPU/Pandas Feature Evaluation Engine.
    Replaces GPU AlphaProfiler completely during the metrics calculation stage to avoid CUDA OOM errors.
    """
    def __init__(self, df, factor_col, label_col, device='cpu', use_rank_ic=False, holding_horizon=15):
        logger.info(f"📊 [CPU Profiler] Initializing Vectorized CPU Evaluation Engine (Period IC, 2000-min Window) | Factor: {factor_col} | Label: {label_col}")
        self.df = df.copy()
        self.factor_col = factor_col
        self.label_col = label_col
        self.use_rank_ic = use_rank_ic
        self.holding_horizon = holding_horizon

        # 1. 完整保留：处理交易日与夜盘对齐 (修复跨年/跨日边界)
        self.df['index_time'] = pd.to_datetime(self.df['index_time'])
        self.df['trading_date'] = self.df['index_time'].dt.date
        is_night = self.df['index_time'].dt.hour >= 18
        self.df.loc[is_night, 'trading_date'] = (self.df.loc[is_night, 'index_time'] + pd.Timedelta(days=1)).dt.date

        logger.info("🔧 [Hotfix] 正在强行对齐夜盘的主力合约标签 (Trading Date Alignment)...")
        self.df['contract'] = self.df.groupby(['instrument', 'trading_date'])['contract'].transform('last')
        
        # 基于正确的 trading_date 提取年份
        self.df['trading_year'] = pd.to_datetime(self.df['trading_date']).dt.year
        
        # 💥 新增：生成 2000 分钟无重叠窗口 ID (用于替代 Daily 计算 Mean Period IC)
        time_factor, _ = pd.factorize(self.df['index_time'], sort=True)
        self.df['window_id'] = time_factor // 2000

        # 2. Tag universe pools (脱敏处理：使用白名单替代私有 CSV 依赖)
        self._tag_universe_pools()

        # 3. [Critical Update] Apply robust 1% - 99% Winsorization to factor and label values per Pool
        self._winsorize_data_by_pool()
        
        # 提取有效测试年份
        self.valid_years = sorted(self.df['trading_year'].unique().tolist())

    def _winsorize_data_by_pool(self):
        logger.info("🛡️  [Winsorization] Performing robust 1% - 99% outlier clipping on Factor and Label series per Pool subgroup...")
        for pool_name, group in self.df.groupby('pool'):
            idx = group.index
            # Winsorize Factor
            f_col = self.df.loc[idx, self.factor_col]
            f_v = f_col.dropna().values
            if len(f_v) > 0:
                f_q01, f_q99 = np.percentile(f_v, [1, 99])
                self.df.loc[idx, self.factor_col] = f_col.clip(lower=f_q01, upper=f_q99)
            # Winsorize Label
            l_col = self.df.loc[idx, self.label_col]
            l_v = l_col.dropna().values
            if len(l_v) > 0:
                l_q01, l_q99 = np.percentile(l_v, [1, 99])
                self.df.loc[idx, self.label_col] = l_col.clip(lower=l_q01, upper=l_q99)

    def _tag_universe_pools(self):
        """直接用标准白名单划分宇宙，彻底摆脱私有外部文件依赖"""
        index_insts = ['IC', 'IF', 'IH', 'IM']
        self.df['pool'] = np.where(self.df['instrument'].isin(index_insts), 'Index', 'Commodity')

    def _filter_complete_years(self):
        year_month_counts = self.df.groupby('trading_year')['trading_month'].nunique()
        self.valid_years = year_month_counts[year_month_counts == 12].index.tolist()
        if not self.valid_years:
            self.valid_years = sorted(self.df['trading_year'].unique().tolist())
        else:
            self.df = self.df[self.df['trading_year'].isin(self.valid_years)]

    @staticmethod
    def _fast_mi_cpu(x_v, y_v, bins=10):
        if len(x_v) < 50:
            return np.nan
        x_min, x_max = x_v.min(), x_v.max()
        y_min, y_max = y_v.min(), y_v.max()

        # Prevent division by zero
        x_range = x_max - x_min if x_max > x_min else 1e-8
        y_range = y_max - y_min if y_max > y_min else 1e-8

        x_bins = np.clip(np.floor((x_v - x_min) / x_range * bins), 0, bins - 1).astype(int)
        y_bins = np.clip(np.floor((y_v - y_min) / y_range * bins), 0, bins - 1).astype(int)

        # Build 2D histogram
        hist2d, _, _ = np.histogram2d(x_bins, y_bins, bins=(bins, bins), range=[[0, bins], [0, bins]])
        pxy = hist2d / hist2d.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            mi_matrix = pxy * np.log2((pxy + 1e-8) / (px * py + 1e-8))
        return float(np.nansum(mi_matrix))

    @staticmethod
    def _compute_anova_r2(f_vals, l_vals, bins=10):
        """
        Computes ANOVA R^2 (Eta-squared) by dividing the factor values into bins (deciles)
        and evaluating how much of the label's variance is explained by bin membership.
        """
        if len(f_vals) < 50:
            return np.nan
        try:
            edges = np.percentile(f_vals, np.linspace(0, 100, bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2:
                return 0.0
            bucket_idx = np.digitize(f_vals, edges)

            overall_mean = np.mean(l_vals)
            ss_total = np.sum((l_vals - overall_mean) ** 2)
            if ss_total < 1e-12:
                return 0.0

            ss_between = 0.0
            for b in np.unique(bucket_idx):
                mask = (bucket_idx == b)
                n_k = np.sum(mask)
                if n_k > 0:
                    mean_k = np.mean(l_vals[mask])
                    ss_between += n_k * (mean_k - overall_mean) ** 2

            return float(ss_between / ss_total)
        except Exception:
            return np.nan

    def _compute_vectorized_daily_ic(self, df_clean: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
        """
        Calculates daily Pearson IC coefficient fully vectorized across groups
        without any python loops!
        """
        df_clean = df_clean.copy()

        if getattr(self, "use_rank_ic", False):
            # 100% Vectorized Daily Cross-Sectional Rank-Transformation
            df_clean[self.factor_col] = df_clean.groupby(groupby_cols)[self.factor_col].rank()
            df_clean[self.label_col] = df_clean.groupby(groupby_cols)[self.label_col].rank()

        df_clean['X2'] = df_clean[self.factor_col] ** 2
        df_clean['Y2'] = df_clean[self.label_col] ** 2
        df_clean['XY'] = df_clean[self.factor_col] * df_clean[self.label_col]

        # Aggregate across all target groups in one shot!
        grouped = df_clean.groupby(groupby_cols).agg(
            N=(self.factor_col, 'count'),
            sum_X=(self.factor_col, 'sum'),
            sum_Y=(self.label_col, 'sum'),
            sum_X2=('X2', 'sum'),
            sum_Y2=('Y2', 'sum'),
            sum_XY=('XY', 'sum')
        ).reset_index()

        N = grouped['N']
        sum_X = grouped['sum_X']
        sum_Y = grouped['sum_Y']
        sum_X2 = grouped['sum_X2']
        sum_Y2 = grouped['sum_Y2']
        sum_XY = grouped['sum_XY']

        numerator = N * sum_XY - sum_X * sum_Y
        denominator = np.sqrt(np.clip(N * sum_X2 - sum_X**2, 0, None) * np.clip(N * sum_Y2 - sum_Y**2, 0, None))

        # Pearson IC definition
        valid_mask = (N >= 6) & (denominator > 1e-8)
        grouped['IC'] = np.where(valid_mask, numerator / (denominator + 1e-12), np.nan)
        return grouped

    def profile_pooled_and_quantile(self):
        logger.info("-> [CPU Profiler] Computing Table B/C: Vectorized Pooled & Quantile Table (Based on 2000-min Windows)...")

        df_clean = self.df[["trading_year", "pool", "window_id", "index_time", self.factor_col, self.label_col]].dropna(subset=[self.factor_col, self.label_col]).copy()
        if df_clean.empty:
            return pd.DataFrame()

        # 1. Compute all period ICs for Table B/C in one single vectorized step!
        grouped_ic = self._compute_vectorized_daily_ic(df_clean, ['trading_year', 'pool', 'window_id'])

        results = []
        for target_year in self.valid_years:
            year_grouped = grouped_ic[grouped_ic['trading_year'] == target_year]
            year_df = self.df[self.df['trading_year'] == target_year]

            for pool_name, pool_group in year_df.groupby('pool'):
                pool_ics = year_grouped[(year_grouped['pool'] == pool_name)]['IC'].dropna().values

                if len(pool_ics) == 0:
                    continue

                pooled_mean_ic = float(np.mean(pool_ics))
                # 💥 关键修改：2000分钟为一窗口，一年有效窗口约 30.24 个
                pooled_ir = float(pooled_mean_ic / (np.std(pool_ics, ddof=1) + 1e-8) * np.sqrt(30.24))
                pooled_wr = float(np.mean(pool_ics > 0))

                f_raw = pool_group[self.factor_col].values
                l_raw = pool_group[self.label_col].values
                valid = (~np.isnan(f_raw)) & (~np.isnan(l_raw))
                f_v = f_raw[valid]
                l_v = l_raw[valid]

                if getattr(self, "use_rank_ic", False):
                    f_v_ranked = pd.Series(f_v).rank().values
                    l_v_ranked = pd.Series(l_v).rank().values
                    full_pooled_ic = float(np.corrcoef(f_v_ranked, l_v_ranked)[0, 1]) if len(f_v) > 10 else np.nan
                else:
                    full_pooled_ic = float(np.corrcoef(f_v, l_v)[0, 1]) if len(f_v) > 10 else np.nan

                full_pooled_mi = self._fast_mi_cpu(f_v, l_v)
                full_pooled_anova_r2 = self._compute_anova_r2(f_v, l_v)

                # Window-based Mean MI and ANOVA
                window_mis = []
                window_anovas = []
                for _, window_group in pool_group.groupby('window_id'):
                    g_clean = window_group[[self.factor_col, self.label_col]].dropna()
                    if len(g_clean) >= 50:
                        f_day = g_clean[self.factor_col].values
                        l_day = g_clean[self.label_col].values
                        window_mis.append(self._fast_mi_cpu(f_day, l_day))
                        window_anovas.append(self._compute_anova_r2(f_day, l_day))

                pooled_mean_mi = float(np.mean(window_mis)) if window_mis else 0.0
                pooled_mean_anova_r2 = float(np.mean(window_anovas)) if window_anovas else 0.0

                # Quantile Stratification (Deciles) & Extreme Tail Open-Zone Risk-Adjusted Returns (5% vs 95%)
                if len(f_v) > 100:
                    q_edges = np.percentile(f_v, np.linspace(0, 100, 11))
                    q_edges = np.unique(q_edges)
                    bucket_idx = np.digitize(f_v, q_edges)

                    q1_mask = (bucket_idx == 1)
                    q10_mask = (bucket_idx == bucket_idx.max())

                    ret_q1 = float(np.mean(l_v[q1_mask])) if q1_mask.any() else np.nan
                    ret_q10 = float(np.mean(l_v[q10_mask])) if q10_mask.any() else np.nan
                    spread = ret_q10 - ret_q1

                    # 5% and 95% Quantiles for open-zone risk-adjusted returns (Sharpe Equivalent of open zones)
                    q05_val = np.percentile(f_v, 5)
                    q95_val = np.percentile(f_v, 95)
                    q05_mask = (f_v <= q05_val)
                    q95_mask = (f_v >= q95_val)

                    l_v_q05 = l_v[q05_mask]
                    l_v_q95 = l_v[q95_mask]

                    # Standard deviation calculation
                    std_global = np.std(l_v, ddof=1) if len(l_v) > 1 else np.nan
                    std_q05 = np.std(l_v_q05, ddof=1) if len(l_v_q05) > 1 else np.nan
                    std_q95 = np.std(l_v_q95, ddof=1) if len(l_v_q95) > 1 else np.nan

                    # Risk-Adjusted Returns (Sharpe Equivalent on extreme tail open zones)
                    tail_low_adj_ret = float(np.mean(l_v_q05) / (std_q05 + 1e-8)) if std_q05 is not None and not np.isnan(std_q05) else np.nan
                    tail_high_adj_ret = float(np.mean(l_v_q95) / (std_q95 + 1e-8)) if std_q95 is not None and not np.isnan(std_q95) else np.nan

                    # Global Standard deviation-adjusted returns
                    tail_low_glob_adj_ret = float(np.mean(l_v_q05) / (std_global + 1e-8)) if std_global is not None and not np.isnan(std_global) else np.nan
                    tail_high_glob_adj_ret = float(np.mean(l_v_q95) / (std_global + 1e-8)) if std_global is not None and not np.isnan(std_global) else np.nan
                else:
                    ret_q1, ret_q10, spread = np.nan, np.nan, np.nan
                    tail_low_adj_ret, tail_high_adj_ret = np.nan, np.nan
                    tail_low_glob_adj_ret, tail_high_glob_adj_ret = np.nan, np.nan

                # 20 equal-sized percentiles and Monotonicity Spearman IC
                # 10 equal-sized percentiles and Monotonicity Spearman IC
                group_returns_10 = [0.0] * 10
                group_spearman_ic = 0.0
                if len(f_v) >= 100:
                    try:
                        ranks = pd.Series(f_v).rank(pct=True, method='first').values
                        group_idx = np.minimum(9, np.floor(ranks * 10).astype(int))

                        grouped_means = []
                        for g in range(10):
                            mask = (group_idx == g)
                            if np.any(mask):
                                grouped_means.append(float(np.mean(l_v[mask])))
                            else:
                                grouped_means.append(0.0)
                        group_returns_10 = grouped_means

                        group_indices = np.arange(1, 11)
                        group_spearman_ic = float(np.corrcoef(pd.Series(group_indices).rank(), pd.Series(group_returns_10).rank())[0, 1])
                        if np.isnan(group_spearman_ic):
                            group_spearman_ic = 0.0
                    except Exception as ge:
                        logger.error(f"Failed to compute 10 group returns: {ge}")

                results.append({
                    'Year': target_year,
                    'Pool': pool_name.upper(),
                    'Mean_IC': pooled_mean_ic,
                    'IR_Ann': pooled_ir,
                    'WinRate': pooled_wr,
                    'Full_IC': full_pooled_ic,
                    'Full_MI': full_pooled_mi,
                    'Mean_MI': pooled_mean_mi,
                    'Full_ANOVA_R2': full_pooled_anova_r2,
                    'Mean_ANOVA_R2': pooled_mean_anova_r2,
                    'Group_Spearman_IC': group_spearman_ic,
                    'Group_Returns_10': group_returns_10,
                    f'Samples_{self.holding_horizon}m': len(f_v),
                    'Q1_Ret (Short)': ret_q1,
                    'Q10_Ret (Long)': ret_q10,
                    'Q_Spread': spread,
                    'Tail_Low_Adj_Ret': tail_low_adj_ret,
                    'Tail_High_Adj_Ret': tail_high_adj_ret,
                    'Tail_Low_Global_Adj_Ret': tail_low_glob_adj_ret,
                    'Tail_High_Global_Adj_Ret': tail_high_glob_adj_ret
                })

        return pd.DataFrame(results).sort_values(by=['Pool', 'Year'])

    def profile_per_symbol(self):
        logger.info("-> [CPU Profiler] Computing Table A: Vectorized Per-Symbol Table...")

        empty_df = pd.DataFrame(columns=['Year', 'Pool', 'Instrument', 'Mean_IC', 'IR_Ann', 'Win_Rate', 'Valid_Periods', 'Full_IC', 'Full_MI'])

        df_clean = self.df[["trading_year", "pool", "instrument", "window_id", self.factor_col, self.label_col]].dropna(subset=[self.factor_col, self.label_col]).copy()
        if df_clean.empty:
            return empty_df

        grouped_ic = self._compute_vectorized_daily_ic(df_clean, ['trading_year', 'pool', 'instrument', 'window_id'])
        grouped_ic_clean = grouped_ic.dropna(subset=['IC']).copy()
        if grouped_ic_clean.empty:
            return empty_df

        grouped_ic_clean['is_positive'] = (grouped_ic_clean['IC'] > 0).astype(int)
        sym_metrics = grouped_ic_clean.groupby(['trading_year', 'pool', 'instrument']).agg(
            Mean_IC=('IC', 'mean'),
            std_ic=('IC', 'std'),
            win_sum=('is_positive', 'sum'),
            Valid_Periods=('IC', 'count')
        ).reset_index()

        sym_metrics['IR_Ann'] = sym_metrics['Mean_IC'] / (sym_metrics['std_ic'] + 1e-8) * np.sqrt(30.24)
        sym_metrics['Win_Rate'] = sym_metrics['win_sum'] / sym_metrics['Valid_Periods']

        sym_metrics = sym_metrics.rename(columns={'trading_year': 'Year', 'pool': 'Pool', 'instrument': 'Instrument'})
        sym_metrics['Pool'] = sym_metrics['Pool'].str.upper()

        # 3. Compute Full_IC and Full_MI per instrument and year
        full_metrics = []
        for (year, pool, inst), group in self.df.groupby(['trading_year', 'pool', 'instrument']):
            f_down = group[self.factor_col].values
            l_down = group[self.label_col].values
            valid_all = (~np.isnan(f_down)) & (~np.isnan(l_down))
            f_all = f_down[valid_all]
            l_all = l_down[valid_all]

            full_ic = float(np.corrcoef(f_all, l_all)[0, 1]) if len(f_all) > 10 else np.nan
            mi = self._fast_mi_cpu(f_all, l_all)

            full_metrics.append({
                'Year': year,
                'Instrument': inst,
                'Pool': pool.upper(),
                'Full_IC': full_ic,
                'Full_MI': mi
            })

        df_full = pd.DataFrame(full_metrics)

        # Merge
        merged = pd.merge(
            sym_metrics[['Year', 'Pool', 'Instrument', 'Mean_IC', 'IR_Ann', 'Win_Rate', 'Valid_Periods']],
            df_full,
            on=['Year', 'Pool', 'Instrument'],
            how='inner'
        )

        return merged.sort_values(by=['Year', 'Pool', 'Mean_IC'], ascending=[True, True, False])


# Re-alias to AlphaProfiler to preserve full transparent usage
AlphaProfiler = CPUAlphaProfiler

logger = logging.getLogger("alpha_evaluator")

# Persistent, cached Global Context to avoid repeating Parquet reads
_PERSISTENT_GLOBAL_CONTEXT: GlobalTensorContext | None = None


def load_valid_instruments(focus_universe: str = "All") -> list[str]:
    """
    获取合法品种代码池，拒绝僵尸与残缺品种污染。
    """
    CFFEX_INSTRUMENTS = ['IC', 'IF', 'IH', 'IM']
    COMMODITY_INSTRUMENTS = [
        'AG', 'AL', 'AO', 'AU', 'BR', 'BU', 'CU', 'FU', 'HC', 'NI', 'PB', 'RB', 'RU', 'SN', 'SP', 'SS', 'WR', 'ZN',
        'A', 'B', 'BB', 'C', 'CS', 'EB', 'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V', 'Y',
        'AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'PF', 'PK', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR',
        'BC', 'EC', 'LU', 'NR', 'SC', 'LC'
    ]
    
    EXCLUDED_INSTRUMENTS = {'LC', 'EC', 'AO', 'BB', 'FB', 'WR', 'RS', 'BC', 'LU'}
    
    if focus_universe == "Index":
        all_targets = set(CFFEX_INSTRUMENTS)
    elif focus_universe == "Commodity":
        all_targets = set(COMMODITY_INSTRUMENTS)
    else:
        all_targets = set(CFFEX_INSTRUMENTS + COMMODITY_INSTRUMENTS)
        
    return [inst for inst in all_targets if inst not in EXCLUDED_INSTRUMENTS]


_PERSISTENT_GLOBAL_CONTEXT: GlobalTensorContext | None = None
_CURRENT_CACHED_UNIVERSE: str = "All"


def get_or_create_global_context(focus_universe: str = "All") -> GlobalTensorContext:
    """
    Instantiates or retrieves the cached GlobalTensorContext spanning 2025 (1-Year).
    Caches the loaded base tensors securely inside the GPU VRAM.
    """
    global _PERSISTENT_GLOBAL_CONTEXT, _CURRENT_CACHED_UNIVERSE
    if _PERSISTENT_GLOBAL_CONTEXT is not None and _CURRENT_CACHED_UNIVERSE == focus_universe:
        logger.info(f"⚡ [VRAM Optimization] Re-using persistent GlobalTensorContext currently resident in GPU VRAM (Universe: {focus_universe}).")
        return _PERSISTENT_GLOBAL_CONTEXT

    if _PERSISTENT_GLOBAL_CONTEXT is not None:
        logger.info(f"♻️ [VRAM Re-allocation] Universe changed from {_CURRENT_CACHED_UNIVERSE} to {focus_universe}. Discarding previous context...")
        _PERSISTENT_GLOBAL_CONTEXT = None
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"💾 [VRAM Cache Miss] Initializing persistent Futures High-Frequency GlobalTensorContext (1-Year Mode: 2025 | Universe: {focus_universe})...")
    base_panel_dir = project_root / "data" / "intraday_parquet" / "base_panel"

    # Load validated instruments based on universe focus
    valid_instruments = set(load_valid_instruments(focus_universe))
    logger.info(f"💾 Loading all {len(valid_instruments)} valid {focus_universe} instruments into Global Context...")

    panel_paths = []
    for p in base_panel_dir.rglob("base_panel_*.parquet"):
        try:
            ticker = p.stem.split("_")[2]
            if ticker in valid_instruments:
                panel_paths.append(p)
        except Exception:
            continue

    if not panel_paths:
        raise FileNotFoundError(f"No valid commodity parquet panels found under {base_panel_dir}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1-Month Time range setting for quick test: January 2025
    target_start = "2025-01-08"
    target_end = "2025-12-31"
    warm_up_days = 3

    logger.info(f"Loaded {len(panel_paths)} commodity instruments panels. Instantiating concurrent loaders on {device}...")

    _PERSISTENT_GLOBAL_CONTEXT = GlobalTensorContext(
        panel_paths,
        target_start=target_start,
        target_end=target_end,
        warm_up_days=warm_up_days,
        device=device,
        max_workers=8 # Standard headnode CPU thread-pool loading
    )

    _CURRENT_CACHED_UNIVERSE = focus_universe
    logger.info(f"🚀 [VRAM Initialized] 1-Year high-frequency panels ({focus_universe}) permanently resident in GPU VRAM.")
    return _PERSISTENT_GLOBAL_CONTEXT


def clean_dynamic_vram_tensors(global_ctx: GlobalTensorContext, keep_features: list[str]):
    """
    Physically deletes calculated dynamic candidate factor tensors from each instrument's context dict.
    Preserves ONLY the immutable base features (Layer 0), ensuring the GPU memory footprint remains constant.
    """
    # Base feature lists (修复了拼写错误 ask1_qty, bid1_qty，并加入了 open_interest)
    base_features = {
        "close", "close_adj", "close_trd", "open", "high", "low", "vwap", "vwap_adj", "volume",
        "bid1", "ask1", "bid1_qty", "ask1_qty", "tick_count", "tick_count_trd",
        "open_trd", "high_trd", "low_trd", "open_trd_adj", "high_trd_adj", "low_trd_adj", "close_trd_adj",
        "high_adj", "open_adj", "low_adj", "open_interest",
        "bid2", "ask2", "bid2_qty", "ask2_qty",
        "bid3", "ask3", "bid3_qty", "ask3_qty",
        "bid4", "ask4", "bid4_qty", "ask4_qty",
        "bid5", "ask5", "bid5_qty", "ask5_qty"
    }

    # 动态加载所有的额外物理基础字段，将它们加入白名单保护，绝不误删！
    try:
        from libraries import load_extra_base_fields
        extra_feats = load_extra_base_fields()
        if extra_feats:
            base_features.update(extra_feats.keys())
    except Exception as e:
        logger.warning(f"VRAM GC: Failed to load extra base fields for whitelist: {e}")

    logger.info("♻️  [VRAM Cleaning] Purging dynamic candidate factor tensors from GPU VRAM...")
    purged_count = 0
    for inst_id, inst_ctx in global_ctx.inst_contexts.items():
        keys_to_delete = []
        for key in list(inst_ctx.context_dict.keys()):
            # Delete if not a base feature and not explicitly flagged to keep
            if key not in base_features and key not in keep_features:
                keys_to_delete.append(key)

        for k in keys_to_delete:
            inst_ctx.delete(k)
            purged_count += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"♻️  [VRAM Cleared] Safely deleted {purged_count} dynamic tensor variables. GPU memory recycled.")


def _evaluate_single_factor(name: str, df_final: pd.DataFrame, label_col: str, device: str, focus_universe: str = "All", holding_horizon: int = 15) -> tuple[str, dict[str, Any]]:
    """Evaluates a single factor using AlphaProfiler in a two-stage cascade manner."""
    logger.info(f"📊 [Profiler] Starting metrics profiling for candidate factor: '{name}' (Holding Horizon: {holding_horizon}m)...")
    # Create a filtered copy for evaluation
    df_factor = df_final[["index_time", "instrument", name, label_col]].dropna(subset=[name])

    # 防御性安全守卫：如果非 NaN 样本数过少，直接跳过评测，防止底层空 DataFrame 抛出 Pool 划分错误
    valid_samples = df_factor[name].notna().sum() if not df_factor.empty else 0
    if df_factor.empty or valid_samples < 50:
        logger.warning(f"⚠️ [Profiler] Factor {name} has insufficient non-NaN samples ({valid_samples}). Skipping profiler.")
        return name, {
            "passed": False,
            "error_msg": "Factor has insufficient non-NaN samples.",
            "ic": 0.0,
            "ir": 0.0,
            "win_rate": 0.0,
            "feedback_log": "⚠️ 异常：该因子计算出的非 NaN 有效值样本数过少或全为空。请检查所用原始变量（如 close_trd）在该测试集时间段内是否有充足数据，或滑动窗口设置是否过大。"
        }

    try:
        profiler = AlphaProfiler(df_factor, factor_col=name, label_col=label_col, device=device, holding_horizon=holding_horizon)

        # Stage 1: Fast pooled metrics profiling on GPU
        df_pooled = profiler.profile_pooled_and_quantile()

        # Extract target pool metrics (Index -> INDEX, Commodity -> COMMODITY)
        target_pool_name = "COMMODITY"
        if focus_universe == "Index":
            target_pool_name = "INDEX"

        comm_row = df_pooled[df_pooled["Pool"] == target_pool_name]
        if comm_row.empty:
            comm_row = df_pooled.iloc[[0]]

        mean_ic = float(comm_row["Mean_IC"].iloc[0])
        full_ic = float(comm_row["Full_IC"].iloc[0]) if "Full_IC" in comm_row else mean_ic
        ir_ann = float(comm_row["IR_Ann"].iloc[0])
        win_rate = float(comm_row["WinRate"].iloc[0])
        q_spread = float(comm_row["Q_Spread"].iloc[0])
        samples = int(comm_row[f"Samples_{holding_horizon}m"].iloc[0])

        # Load new non-linear & monotonicity metrics
        full_mi = float(comm_row["Full_MI"].iloc[0]) if "Full_MI" in comm_row else 0.0
        mean_mi = float(comm_row["Mean_MI"].iloc[0]) if "Mean_MI" in comm_row else 0.0
        full_anova = float(comm_row["Full_ANOVA_R2"].iloc[0]) if "Full_ANOVA_R2" in comm_row else 0.0
        mean_anova = float(comm_row["Mean_ANOVA_R2"].iloc[0]) if "Mean_ANOVA_R2" in comm_row else 0.0
        group_spearman = float(comm_row["Group_Spearman_IC"].iloc[0]) if "Group_Spearman_IC" in comm_row else 0.0

        # Load tail open-zone risk-adjusted returns (5% vs 95% quantiles)
        tail_low_adj_ret = float(comm_row["Tail_Low_Adj_Ret"].iloc[0]) if "Tail_Low_Adj_Ret" in comm_row else 0.0
        tail_high_adj_ret = float(comm_row["Tail_High_Adj_Ret"].iloc[0]) if "Tail_High_Adj_Ret" in comm_row else 0.0
        tail_low_glob_adj_ret = float(comm_row["Tail_Low_Global_Adj_Ret"].iloc[0]) if "Tail_Low_Global_Adj_Ret" in comm_row else 0.0
        tail_high_glob_adj_ret = float(comm_row["Tail_High_Global_Adj_Ret"].iloc[0]) if "Tail_High_Global_Adj_Ret" in comm_row else 0.0

        # Core Acceptance Gates: either absolute value of pooled mean IC > 0.02 or whole year IC (Full_IC) > 0.008
        passed_cascade_gate = abs(mean_ic) > 0.02 or abs(full_ic) > 0.008
        passed = passed_cascade_gate
        pool_cn_name = "股指池" if focus_universe == "Index" else "商品池"

        if not passed_cascade_gate:
            # Bypass heavy per-symbol profiling completely to save VRAM and prevent OOM
            logger.info(f"  ❌ Factor '{name}' rejected by Cascade Gate (Mean IC={mean_ic:.5f}, Whole Year IC={full_ic:.5f}). Bypassing per-symbol profiling.")

            df_pooled_clean = df_pooled.replace([np.nan, np.inf, -np.inf], None)
            pooled_metrics = df_pooled_clean.to_dict(orient="records")

            feedback_log = (
                f"📉 未通过：因子表现未达标。{pool_cn_name} Pooled Mean IC = {mean_ic:.5f} (门槛绝对值 > 0.02)，"
                f"或 Pooled Whole Year IC = {full_ic:.5f} (门槛绝对值 > 0.008，均未满足)。\n"
                f"  - 非线性关联度：Full MI = {full_mi:.5f}，Mean Daily MI = {mean_mi:.5f}，Full ANOVA R2 = {full_anova:.5f}，Mean Daily ANOVA R2 = {mean_anova:.5f}。\n"
                f"  - 分组单调性：10组等分分组收益率 Spearman IC = {group_spearman:.5f}。\n"
                f"  - 尾部极端开仓交易区风险调整收益率 (Trade Zone Risk-Adjusted Return / Sharpe Equivalent):\n"
                f"    * 5%低分位数区间(空头开仓区)自适应调整收益率 = {tail_low_adj_ret:.5f} (全样本调整: {tail_low_glob_adj_ret:.5f})\n"
                f"    * 95%高分位数区间(多头开仓区)自适应调整收益率 = {tail_high_adj_ret:.5f} (全样本调整: {tail_high_glob_adj_ret:.5f})\n"
                f"    * 注：这直接反映了极端极值分位数（对应具体交易开仓比例）的理想信息比率/Sharpe表现，至少应有一端具备显著绝对值。"
            )

            return name, {
                "passed": False,
                "ic": mean_ic, "full_ic": full_ic, "ir": ir_ann, "win_rate": win_rate, "q_spread": q_spread, "samples": samples,
                "feedback_log": feedback_log,
                "top_performers": [], "worst_performers": [],
                "pooled_metrics": pooled_metrics, "per_symbol_metrics": [],
                "tail_low_adj_ret": tail_low_adj_ret, "tail_high_adj_ret": tail_high_adj_ret,
                "tail_low_glob_adj_ret": tail_low_glob_adj_ret, "tail_high_glob_adj_ret": tail_high_glob_adj_ret
            }

        # ====================================================================
        # 🛡️ 护城河机制启动：因子已证明其实力，后续代码哪怕全部崩溃，也要强行保活！
        # ====================================================================
        logger.info(f"  🔥 Factor '{name}' PASSED Cascade Gate! Launching heavy per-symbol profiling...")
        
        # 提前准备兜底的数据和反馈文案
        df_pooled_clean = df_pooled.replace([np.nan, np.inf, -np.inf], None)
        pooled_metrics = df_pooled_clean.to_dict(orient="records")
        
        top_performers = []
        worst_performers = []
        per_symbol_metrics = []
        
        feedback_lines = [
            f"🎉 成功：候选因子通过评估门槛！{pool_cn_name} Pooled Whole Year IC = {full_ic:.5f} (目标绝对值 > 0.0080)。",
            f"  - 核心分布统计：Mean Pearson IC = {mean_ic:.5f}，Annualized IR = {ir_ann:.3f}，多空分层 Q_Spread = {q_spread:.6f}，Win Rate = {win_rate * 100:.1f}%，总样本 = {samples}。",
            f"  - 非线性与单调性：Full MI = {full_mi:.5f}，Mean Period MI = {mean_mi:.5f}，Full ANOVA R2 = {full_anova:.5f}，Mean Period ANOVA R2 = {mean_anova:.5f}，10组单调性 Spearman IC = {group_spearman:.5f}。",
            f"  - 尾部极端开仓交易区风险调整收益率 (Trade Zone Risk-Adjusted Return):\n"
            f"    * 5%低分位数区间(空头开仓区)自适应调整收益率 = {tail_low_adj_ret:.5f}\n"
            f"    * 95%高分位数区间(多头开仓区)自适应调整收益率 = {tail_high_adj_ret:.5f}"
        ]

        try:
            # Stage 2: 尝试极其容易因极端数据报错的单品种详细分析
            df_per_sym = profiler.profile_per_symbol()
            
            df_per_sym_sorted = df_per_sym.sort_values(by="IR_Ann", ascending=False)
            top_performers = df_per_sym_sorted.head(3)[["Instrument", "Mean_IC", "IR_Ann"]].to_dict(orient="records")
            worst_performers = df_per_sym_sorted.tail(3)[["Instrument", "Mean_IC", "IR_Ann"]].to_dict(orient="records")

            top_str = ", ".join(f"{r['Instrument']}(IR={r['IR_Ann']:.2f})" for r in top_performers)
            worst_str = ", ".join(f"{r['Instrument']}(IR={r['IR_Ann']:.2f})" for r in worst_performers)
            feedback_lines.append(f"  - 表现最好品种：{top_str}")
            feedback_lines.append(f"  - 表现最差品种：{worst_str}")
            
            df_per_sym_clean = df_per_sym.replace([np.nan, np.inf, -np.inf], None)
            per_symbol_metrics = df_per_sym_clean.to_dict(orient="records")
            
        except Exception as inner_e:
            # 即使这里崩了，只会少掉单品种分析，整体因子依旧是 Passed: True！
            logger.warning(f"⚠️ [护城河保活] 因子 '{name}' 全局表现已达标，但后续单品种剖析崩溃 ({inner_e})。已强制保活该因子！")
            feedback_lines.append(f"  - ⚠️ 警告：该因子的单品种拆解分析在计算中途崩溃 ({inner_e})，但由于其全局 IC 已达标，系统予以强制保活。")

        if ir_ann < -1.0:
            feedback_lines.append("  - 提示：该因子表现出显著的负向超额收益。建议在最终公式前添加 '-' (负号) 从而将其校正为正向强 Alpha！")
        if abs(ir_ann) < 0.5:
            feedback_lines.append("  - 提示：因子信息比率极低，属于高斯噪声。建议更换其他原始变量或使用不同的时序算子。")

        feedback_log = "\n".join(feedback_lines)

        return name, {
            "passed": True,  # 只要来到这里，永远是 True！
            "ic": mean_ic,
            "full_ic": full_ic,
            "ir": ir_ann,
            "win_rate": win_rate,
            "q_spread": q_spread,
            "full_mi": full_mi,
            "mean_mi": mean_mi,
            "full_anova": full_anova,
            "mean_anova": mean_anova,
            "group_spearman": group_spearman,
            "samples": samples,
            "feedback_log": feedback_log,
            "top_performers": top_performers,
            "worst_performers": worst_performers,
            "pooled_metrics": pooled_metrics,
            "per_symbol_metrics": per_symbol_metrics,
            "tail_low_adj_ret": tail_low_adj_ret,
            "tail_high_adj_ret": tail_high_adj_ret,
            "tail_low_glob_adj_ret": tail_low_glob_adj_ret,
            "tail_high_glob_adj_ret": tail_high_glob_adj_ret
        }

    except Exception as e:
        logger.error(f"Failed to profile metrics for {name}: {e}")
        return name, {
            "passed": False,
            "error_msg": f"Profiler crashed: {e}",
            "ic": 0.0,
            "ir": 0.0,
            "win_rate": 0.0,
            "feedback_log": f"评估该因子绩效时崩溃: {e}。请检查因子输出值分布是否正确。",
            "tail_low_adj_ret": 0.0,
            "tail_high_adj_ret": 0.0,
            "tail_low_glob_adj_ret": 0.0,
            "tail_high_glob_adj_ret": 0.0
        }

def compute_lightweight_metrics_on_cpu(df_final: pd.DataFrame, factor_col: str, label_col: str, use_rank_ic: bool = False) -> tuple[float, float, int]:
    """
    Computes pooled mean IC and pooled whole-year IC on CPU for the target instruments pool,
    perfectly matching the mathematical logic of AlphaProfiler.
    Supports Pearson IC on raw values, with no downsampling.
    Returns (pooled_mean_ic, full_pooled_ic, samples).
    """
    # Create copy of relevant columns to save memory and avoid side-effects
    df = df_final[["index_time", "instrument", factor_col, label_col]].dropna(subset=[factor_col]).copy()
    if df.empty or len(df) < 50:
        return 0.0, 0.0, len(df)

    df['index_time'] = pd.to_datetime(df['index_time'])
    time_factor, _ = pd.factorize(df['index_time'], sort=True)
    df['window_id'] = time_factor // 2000

    index_insts = ['IC', 'IF', 'IH', 'IM']
    df['pool'] = np.where(df['instrument'].isin(index_insts), 'Index', 'Commodity')

    df_comm = df[df['pool'] == 'Commodity'].copy()
    if df_comm.empty or len(df_comm) < 50:
        return 0.0, 0.0, len(df_comm)

    for col in [factor_col, label_col]:
        col_series = df_comm[col]
        if not col_series.dropna().empty:
            q01 = col_series.quantile(0.01)
            q99 = col_series.quantile(0.99)
            df_comm[col] = col_series.clip(lower=q01, upper=q99)

    if use_rank_ic:
        df_comm[factor_col] = df_comm.groupby('window_id')[factor_col].rank()
        df_comm[label_col] = df_comm.groupby('window_id')[label_col].rank()

    window_ics = []
    for _, group in df_comm.groupby('window_id'):
        g_clean = group[[factor_col, label_col]].dropna()
        if len(g_clean) >= 6:
            cov = np.cov(g_clean[factor_col], g_clean[label_col])[0, 1]
            std_f = np.std(g_clean[factor_col], ddof=1)
            std_l = np.std(g_clean[label_col], ddof=1)
            if std_f > 1e-8 and std_l > 1e-8:
                window_ics.append(cov / (std_f * std_l))

    pooled_mean_ic = np.mean(window_ics) if window_ics else 0.0

    f_raw = df_comm[factor_col].values
    l_raw = df_comm[label_col].values
    valid_mask = (~np.isnan(f_raw)) & (~np.isnan(l_raw))
    f_v = f_raw[valid_mask]
    l_v = l_raw[valid_mask]

    samples = len(f_v)
    if len(f_v) > 10:
        if use_rank_ic:
            f_v_ranked = pd.Series(f_v).rank().values
            l_v_ranked = pd.Series(l_v).rank().values
            full_pooled_ic = np.corrcoef(f_v_ranked, l_v_ranked)[0, 1]
        else:
            full_pooled_ic = np.corrcoef(f_v, l_v)[0, 1]
        if np.isnan(full_pooled_ic):
            full_pooled_ic = 0.0
    else:
        full_pooled_ic = 0.0

    return float(pooled_mean_ic), float(full_pooled_ic), int(samples)


def evaluate_factor_batch(factor_formulas: dict[str, str], focus_universe: str = "All", holding_horizon: int = 15) -> dict[str, dict[str, Any]]:
    """
    Compiles and executes a batch of formulas in a SINGLE GPU run, extracts signals,
    profiles metrics, and dynamically recycles VRAM.
    """
    logger.info(f"======================================================================")
    logger.info(f"🚀  批量压测启动：同时并发编译计算 {len(factor_formulas)} 个候选特征 (Universe: {focus_universe} | Horizon: {holding_horizon}m) 🚀")
    logger.info(f"======================================================================")

    # 1. Compile all factors into a single DAGManager
    dag = DAGManager()
    target_nodes = {}
    compile_failures = {}

    for name, formula in factor_formulas.items():
        try:
            node_id = dag.add_formula(formula)
            target_nodes[name] = node_id
            logger.info(f"  √ {name:15} -> DAG node: {node_id} : {formula}")
        except Exception as e:
            logger.error(f"Failed to compile formula for {name} ({formula}): {e}")
            compile_failures[name] = {
                "passed": False,
                "error_msg": f"DAG compilation failed: {e}",
                "ic": 0.0,
                "ir": 0.0,
                "win_rate": 0.0,
                "feedback_log": f"语法或表达式编译失败: {e}。请检查使用的变量或算子拼写是否合法。"
            }

    if not target_nodes:
        # All failed compilation
        return compile_failures

    # Add label node (Ret_Xm_Fwd)
    label_col = f"Ret_{holding_horizon}m_Fwd"
    label_formula = f"TsForwardReturn(close, {holding_horizon})"
    label_node_id = dag.add_formula(label_formula)
    target_nodes[label_col] = label_node_id

    # 2. Retrieve or Load persistent Global Context
    try:
        global_ctx = get_or_create_global_context(focus_universe)
    except Exception as e:
        logger.error(f"Failed to load Global Tensor Context: {e}")
        return {
            name: {
                "passed": False,
                "error_msg": f"VRAM Context Load failed: {e}",
                "ic": 0.0,
                "ir": 0.0,
                "win_rate": 0.0,
                "feedback_log": "系统无法装载 GPU VRAM 缓存，暂停评估。"
            } for name in factor_formulas
        }

    # 3. Execute concurrently on GPU via BatchExecutor (Parallelizing all compiled nodes)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 1-Month Time range setting for quick test: January 2025
    target_start = "2025-01-08"
    target_end = "2025-12-31"

    try:
        logger.info("⚡ [GPU Graph Execution] Running graph scheduling in parallel...")
        executor = BatchExecutor(dag, global_ctx, max_concurrent_insts=4)

        # Artificially inflate the out-degrees of all target factor nodes to prevent BatchExecutor's
        # aggressive VRAM garbage collector from deleting them before we can extract their signals.
        for node_id in target_nodes.values():
            if node_id in executor.out_degrees:
                executor.out_degrees[node_id] += 999999

        executor.run()

        # Extract unified signal panel
        df_final = executor.extract_multiple_signals(
            target_nodes_dict=target_nodes,
            mode="main_only",
            target_start=target_start,
            target_end=target_end
        )
    except Exception as e:
        logger.error(f"Computation error during GPU graph execution: {e}")
        # Clean context to be safe
        clean_dynamic_vram_tensors(global_ctx, [])
        return {
            name: {
                "passed": False,
                "error_msg": f"GPU Engine crashed: {e}",
                "ic": 0.0,
                "ir": 0.0,
                "win_rate": 0.0,
                "feedback_log": f"GPU 计算张量表达式时报错崩溃: {e}。可能是滑动窗口越界或计算数值发散。"
            } for name in target_nodes if name != label_col
        }

    # 4. Truncate and clean day-ends (Prevent overnight label leak)
    if df_final.empty:
        clean_dynamic_vram_tensors(global_ctx, [])
        return {
            name: {
                "passed": False,
                "error_msg": "DataFrame returned from BatchExecutor is empty.",
                "ic": 0.0,
                "ir": 0.0,
                "win_rate": 0.0,
                "feedback_log": "执行引擎返回的结果为空表，请检查时间窗口设置。"
            } for name in target_nodes if name != label_col
        }

    logger.info(f"✂️ [EOD Truncation] 执行下午 15:00 收盘前的固定尾部时间段修剪 (Cutoff: {holding_horizon + 1}m)...")
    
    import datetime
    
    # 1. 动态计算应该被剪掉的起止时间段
    # 下午收盘时间固定为 15:00:00
    dummy_dt = datetime.datetime(2000, 1, 1, 15, 0, 0)
    # 往前倒推 holding_horizon + 1 分钟
    cutoff_dt = dummy_dt - datetime.timedelta(minutes=(holding_horizon + 1))
    
    cutoff_time = cutoff_dt.time()
    end_time = datetime.time(15, 0, 0)
    
    # 2. 提取当前所有数据的时间部分
    bar_times = df_final['index_time'].dt.time
    
    # 3. 找出所有落在每天尾部修剪时间段内的 K 线
    # 例如 horizon=30, 则时间段为 (14:29:00, 15:00:00]
    tail_mask = (bar_times > cutoff_time) & (bar_times <= end_time)
    
    # 4. 将这部分不再开仓的尾部时段的收益率标签全部抹除
    df_final.loc[tail_mask, label_col] = np.nan

    # 5. Evaluate each compiled factor in PARALLEL using a ThreadPoolExecutor
    batch_results = {}

    # Store compilation failures
    batch_results.update(compile_failures)

    factors_to_eval = [name for name in target_nodes if name != label_col]

    logger.info(f"⚡ [Profiler-Parallel] Spawning ThreadPoolExecutor with {min(len(factors_to_eval), 8)} workers for concurrent metric evaluation...")
    with ThreadPoolExecutor(max_workers=min(len(factors_to_eval), 4)) as executor:
        # Submit all evaluation tasks
        futures = {
            executor.submit(_evaluate_single_factor, name, df_final, label_col, device, focus_universe, holding_horizon): name
            for name in factors_to_eval
        }

        # Collect results as they complete
        for future in futures:
            name = futures[future]
            try:
                fname, res = future.result()
                batch_results[fname] = res
            except Exception as e:
                logger.error(f"Thread execution failed for factor {name}: {e}")
                batch_results[name] = {
                    "passed": False,
                    "error_msg": f"Thread crashed: {e}",
                    "ic": 0.0,
                    "ir": 0.0,
                    "win_rate": 0.0,
                    "feedback_log": f"多线程并发评估该因子绩效时崩溃: {e}。"
                }
            finally:
                # Proactively clean VRAM and run garbage collection as tasks complete
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # 6. Recycle VRAM (Delete all temporary candidate tensors from InstrumentContexts)
    clean_dynamic_vram_tensors(global_ctx, [])

    logger.info("======================================================================")
    logger.info(f"🏆  批量压测结束！共完成 {len(batch_results)} 个候选因子的并行评测 🏆")
    logger.info("======================================================================")

    return batch_results


if __name__ == "__main__":
    # Test batch execution
    formulas_to_test = {
        "MOM_20": "TsPct(close_adj, 20)",
        "Dev_MA_120": "close_adj / TsMean(close_adj, 120) - 1.0"
    }
    batch_res = evaluate_factor_batch(formulas_to_test)
    for name, res in batch_res.items():
        print(f"\n[{name}] Evaluation Output:")
        print(res["feedback_log"])
