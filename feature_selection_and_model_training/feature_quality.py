#!/usr/bin/env python3
"""
feature_quality.py - High-Resolution Feature Quality Evaluation & Dashboard Generator.

Parses features from the central dataset, evaluates Multi-Quantile (Bin) performance, 
extracts Period IC / Full IC, and exports granular JSON metrics and Parquet trade positions.
Re-architected for Local Multi-Processing (replaces proprietary Spark cluster).
"""

import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# =========================================================
# 🎛️ 1. 全局配置与动态路径定义 (彻底脱敏)
# =========================================================
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"

PARQUET_PATH = str(data_dir / "factor_dataset_AllMarket_main_only_2025.parquet")
OUTPUT_JSON = str(data_dir / "feature_quality_dashboard_2025.json")
MARKET_STATES_PATH = str(data_dir / "best_market_states_list.json")

# 💥 用于存储每个特征分钟级仓位的分布式目录
POSITIONS_DIR = str(data_dir / "feature_positions")
os.makedirs(POSITIONS_DIR, exist_ok=True)

# 纯净版全市场标准主力合约前缀
EXCHANGE_PREFIXES = {
    'cffex': ['IC', 'IF', 'IH', 'IM'],
    'shfe': ['AG', 'AL', 'AO', 'AU', 'BR', 'BU', 'CU', 'FU', 'HC', 'NI', 'PB', 'RB', 'RU', 'SN', 'SP', 'SS', 'WR', 'ZN'],
    'dce':  ['A', 'B', 'BB', 'C', 'CS', 'EB', 'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V', 'Y'],
    'czce': ['AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'PF', 'PK', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR'],
    'ine':  ['BC', 'EC', 'LU', 'NR', 'SC'],
    'gfex': ['LC']
}

CFFEX_RAW = EXCHANGE_PREFIXES['cffex']
COMM_RAW = EXCHANGE_PREFIXES['shfe'] + EXCHANGE_PREFIXES['dce'] + EXCHANGE_PREFIXES['czce'] + EXCHANGE_PREFIXES['ine'] + EXCHANGE_PREFIXES['gfex']

# 剔除缺乏流动性或不规则的非主流品种
EXCLUDED_INSTRUMENTS = {'LC', 'EC', 'AO', 'BB', 'FB', 'WR', 'RS', 'BC', 'LU'}

VALID_INSTRUMENTS = [inst for inst in set(CFFEX_RAW + COMM_RAW) if inst not in EXCLUDED_INSTRUMENTS]
CFFEX_POOL = set([x for x in VALID_INSTRUMENTS if x in CFFEX_RAW])
COMM_POOL = set([x for x in VALID_INSTRUMENTS if x in COMM_RAW])
TARGETS = ['Ret_15m_Fwd', 'Ret_30m_Fwd']

# 💡 定义不同宇宙的固定手续费率
FEE_COMM = 0.00015   # 商品 万分之1.5
FEE_CFFEX = 0.00025  # 股指 万分之2.5

# 💡 回测超参数：多空头的截面开仓比例 (0.2 代表做多前20%，做空后20%)
LS_RATIO = 0.05

# =========================================================
# ⏱️ 2. 全市场高精度时间清洗 (含剔除开盘第一根 Bar)
# =========================================================
def filter_global_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    time_str = df['index_time'].dt.strftime('%H:%M:%S')
    
    opening_bars = {'09:00:00', '09:30:00', '21:00:00'}
    
    # 股指规则
    is_cffex = df['instrument'].isin(CFFEX_RAW)
    mask_cffex = (
        ((time_str >= '09:30:00') & (time_str <= '11:30:00')) |
        ((time_str >= '13:00:00') & (time_str <= '15:00:00'))
    )
    
    # 商品规则
    is_commodity = ~is_cffex
    mask_comm_day = (
        ((time_str >= '09:00:00') & (time_str <= '10:15:00')) |
        ((time_str >= '10:30:00') & (time_str <= '11:30:00')) |
        ((time_str >= '13:30:00') & (time_str <= '15:00:00'))
    )
    
    # 使用标准品种代码重构夜盘收盘时间分组
    group_0230 = {'AU', 'AG', 'SC'}
    group_0100 = {'CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'SS', 'BC'}
    group_no_night = {'WR', 'BB', 'FB', 'EG', 'LH', 'PK', 'AP', 'CJ', 'RS', 'SM', 'UR', 'LC', 'EC', 'PF'}
    
    mask_night_2300 = (time_str >= '21:00:00') & (time_str <= '23:00:00')
    mask_night_0100 = (time_str >= '21:00:00') | (time_str <= '01:00:00')
    mask_night_0230 = (time_str >= '21:00:00') | (time_str <= '02:30:00')

    is_0230 = df['instrument'].isin(group_0230)
    is_0100 = df['instrument'].isin(group_0100)
    is_no_night = df['instrument'].isin(group_no_night)
    is_2300 = ~(is_0230 | is_0100 | is_no_night | is_cffex)

    mask_comm = mask_comm_day | (
        (is_0230 & mask_night_0230) |
        (is_0100 & mask_night_0100) |
        (is_2300 & mask_night_2300)
    )
    
    valid_mask = ((is_cffex & mask_cffex) | (is_commodity & mask_comm)) & (~time_str.isin(opening_bars))
    return df[valid_mask].copy()

# =========================================================
# ⚡ 3. 核心计算算子 (双重 Winsorize 与 Period IC 对齐)
# =========================================================
def calc_metrics_and_charts(df_sub, feature, tgt, num_bins=10, hist_bins=50):
    valid_df = df_sub.dropna(subset=[feature, tgt]).copy()
    if len(valid_df) < 100:
        return None

    # 💡 精准划定尾盘禁区，不伤及夜盘
    time_strs = valid_df['index_time'].dt.strftime('%H:%M:%S')
    if '15m' in tgt:
        # 标记出 14:45 到 15:00 这个时间段
        mask_drop = (time_strs >= '14:45:00') & (time_strs <= '17:00:00')
        valid_df = valid_df[~mask_drop].copy()
    elif '30m' in tgt:
        # 标记出 14:30 到 15:00 这个时间段
        mask_drop = (time_strs >= '14:30:00') & (time_strs <= '17:00:00')
        valid_df = valid_df[~mask_drop].copy()
        
    if len(valid_df) < 100:
        return None

    res = {}
    
    # 1. 基础极值截断 (Winsorize)
    f_lower, f_upper = valid_df[feature].quantile(0.01), valid_df[feature].quantile(0.99)
    if f_lower == f_upper: f_upper = f_lower + 1e-8
    valid_df[feature] = valid_df[feature].clip(lower=f_lower, upper=f_upper)
    
    t_lower, t_upper = valid_df[tgt].quantile(0.01), valid_df[tgt].quantile(0.99)
    if t_lower == t_upper: t_upper = t_lower + 1e-8
    valid_df[tgt] = valid_df[tgt].clip(lower=t_lower, upper=t_upper)
    
    # 🚨 回归 Pearson 相关系数
    res['full_ic'] = float(valid_df[feature].corr(valid_df[tgt], method='pearson'))
    
    # 💥 替换为 2000 分钟无重叠窗口 Period IC，完美匹配全局评测基准
    time_factor, _ = pd.factorize(valid_df['index_time'], sort=True)
    valid_df['window_id'] = time_factor // 2000
    
    period_ic = valid_df.groupby('window_id').apply(
        lambda x: x[feature].corr(x[tgt], method='pearson') if len(x) > 10 else np.nan
    )
    res['mean_period_ic'] = float(period_ic.mean())

    if 'Quarter' in valid_df.columns:
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_mask = valid_df['Quarter'] == q
            if q_mask.sum() > 100:
                res[f'{q}_ic'] = float(valid_df.loc[q_mask, feature].corr(valid_df.loc[q_mask, tgt], method='pearson'))
            else:
                res[f'{q}_ic'] = None

    # 3. 分布直方图
    h1_mask = valid_df['Half_Year'] == 'H1'
    counts_h1, edges = np.histogram(valid_df.loc[h1_mask, feature], bins=hist_bins, range=(f_lower, f_upper))
    counts_h2, _ = np.histogram(valid_df.loc[~h1_mask, feature], bins=edges)
    res['hist_x'] = ((edges[:-1] + edges[1:]) / 2).tolist()
    res['hist_h1'] = counts_h1.tolist()
    res['hist_h2'] = counts_h2.tolist()

    # 4. 全局数值分位数收益
    try:
        valid_df['Bin'] = pd.qcut(valid_df[feature], q=num_bins, labels=False, duplicates='drop')
        m_all = valid_df.groupby('Bin')[tgt].mean()
        s_all = valid_df.groupby('Bin')[tgt].std()
        # 💥 强转 float，在源头消灭 np.float32
        res['decile_all'] = [float(m_all.get(i, 0.0)) for i in range(num_bins)]
        res['decile_h1'] = [float(valid_df[h1_mask].groupby('Bin')[tgt].mean().get(i, 0.0)) for i in range(num_bins)]
        res['decile_h2'] = [float(valid_df[~h1_mask].groupby('Bin')[tgt].mean().get(i, 0.0)) for i in range(num_bins)]
        res['q_first_ir'] = float(m_all.iloc[0] / s_all.iloc[0]) if s_all.iloc[0] != 0 else 0.0
        res['q_last_ir'] = float(m_all.iloc[-1] / s_all.iloc[-1]) if s_all.iloc[-1] != 0 else 0.0
    except Exception:
        pass

    del valid_df
    return res

# =========================================================
# 🚀 4. 纯时序资金曲线模拟回测
# =========================================================
def run_multi_mode_ts_backtest(df_in: pd.DataFrame, feature_name: str, fee_rate: float, inst_directions: dict, windows=[2000, 4000, 6000]):
    cols = ['index_time', 'instrument', 'contract', feature_name, 'Ret_1m_Fwd', 'Ret_15m_Fwd', 'Ret_30m_Fwd', 'Quarter']
    df_bt = df_in[cols].dropna(subset=[feature_name, 'Ret_1m_Fwd']).copy()
    df_bt = df_bt.sort_values(['instrument', 'index_time']).reset_index(drop=True)
    if len(df_bt) < 100:
        return {}

    # 1. 全自动方向映射 (强制 float32 省内存)
    df_bt['direction'] = df_bt['instrument'].map(inst_directions).fillna(1.0)
    
    # 提取纯 Numpy 数组进行高速运算，剥离 Pandas 开销
    ret_1m_arr = (np.exp(df_bt['Ret_1m_Fwd'].values.astype(np.float32)) - 1.0).astype(np.float32)
    dir_arr = df_bt['direction'].values.astype(np.float32)
    feat_arr = df_bt[feature_name].values.astype(np.float32)
    
    # 品种切换掩码，用于极速计算换手率
    inst_change_mask = (df_bt['instrument'] != df_bt['instrument'].shift(1)).values

    df_export = df_bt[['index_time', 'instrument', 'Ret_1m_Fwd', 'Ret_15m_Fwd', 'Ret_30m_Fwd']].copy()

    # 2. 物理防御级时空掩码生成
    import datetime
    bar_times = df_bt['index_time'].dt.time
    
    df_bt['is_eod_1m'] = (bar_times >= datetime.time(14, 59, 0)) & (bar_times <= datetime.time(15, 0, 0))
    def rev_roll_max(x):
        return x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1].astype(bool)
    
    is_eod_30m = df_bt.groupby('instrument', observed=True)['is_eod_1m'].transform(rev_roll_max).values
    is_eod_1m = df_bt['is_eod_1m'].values
    
    same_inst = df_bt['instrument'] == df_bt['instrument'].shift(-1)
    df_bt['contract_change'] = same_inst & (df_bt['contract'] != df_bt['contract'].shift(-1))
    
    is_rollover_1m = df_bt.groupby('instrument', observed=True)['contract_change'].transform(lambda x: x.shift(-1).fillna(False) | x).values
    is_rollover_30m = df_bt.groupby('instrument', observed=True)['contract_change'].transform(rev_roll_max).values
    
    df_bt['bar_idx'] = df_bt.groupby('instrument', observed=True).cumcount()
    rebalance_mask_30m = (df_bt['bar_idx'] % 30) == 0

    trading_days = 252
    mins_per_day = 345 if fee_rate == FEE_COMM else 240
    
    inst_results = {inst: {} for inst in df_bt['instrument'].unique()}

    # --- 极速 Numpy 闭包计算引擎 ---
    def calc_metrics(pos_arr, use_fee, prefix):
        # 纯 Numpy 计算换手率，避免修改 df_bt
        pos_shift = np.roll(pos_arr, 1)
        turnover = np.abs(pos_arr - pos_shift)
        turnover[inst_change_mask] = 0.0 
        
        cost = turnover * fee_rate if use_fee else 0.0
        pnl = (pos_arr * ret_1m_arr) - cost
        
        pnl_series = pd.Series(pnl, dtype=np.float32)
        nav_arr = pnl_series.groupby(df_bt['instrument'], observed=True).transform(lambda x: (1.0 + x).cumprod()).values
        
        # 临时聚合表，用完即焚
        temp_df = pd.DataFrame({
            'inst': df_bt['instrument'],
            'quarter': df_bt['Quarter'],
            'pnl': pnl_series,
            'turnover': turnover,
            'nav': nav_arr
        })
        
        def assign_metrics(group_df, suffix_key):
            if group_df.empty: return
            grps = group_df.groupby('inst', observed=True)
            
            # 高精度剥离片段收益：片段末尾净值 / (片段初净值回退一格) - 1.0
            rets = (grps['nav'].last() / (grps['nav'].first() / (1.0 + grps['pnl'].first())) - 1.0) * 10000
            sharpes = grps['pnl'].mean() / (grps['pnl'].std() + 1e-8) * np.sqrt(trading_days * mins_per_day)
            avg_turnovers = grps['turnover'].mean() * mins_per_day
            
            for inst in inst_results:
                if inst in rets.index:
                    v_ret, v_shp, v_to = float(rets[inst]), float(sharpes.get(inst, 0.0)), float(avg_turnovers.get(inst, 0.0))
                    
                    inst_results[inst][f'{prefix}_{suffix_key}ret'] = round(v_ret, 4) if not (np.isnan(v_ret) or np.isinf(v_ret)) else None
                    inst_results[inst][f'{prefix}_{suffix_key}shp'] = round(v_shp, 4) if not (np.isnan(v_shp) or np.isinf(v_shp)) else None
                    inst_results[inst][f'{prefix}_{suffix_key}to']  = round(v_to, 4)  if not (np.isnan(v_to)  or np.isinf(v_to))  else None

        # 1. 算全年总指标 (没有 suffix)
        assign_metrics(temp_df, "")
        
        # 2. 分割 4 个季度分别算指标 (加上 Q1_, Q2_... 前缀)
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_df = temp_df[temp_df['quarter'] == q]
            assign_metrics(q_df, f"{q}_")
            
        del pos_shift, turnover, cost, pnl, pnl_series, nav_arr, temp_df

    # 3. 多窗口向量化运算
    grouped_feat = df_bt.groupby('instrument', observed=True)[feature_name]
    
    for w in windows:
        # -------------------------------------------------------------
        # 映射 A：Z-Score 连续线性映射
        # -------------------------------------------------------------
        r_mean = grouped_feat.transform(lambda x: x.rolling(w, min_periods=w//2).mean()).values.astype(np.float32)
        r_std  = grouped_feat.transform(lambda x: x.rolling(w, min_periods=w//2).std()).values.astype(np.float32)
        
        pos_z_arr = np.clip((feat_arr - r_mean) / (r_std + 1e-8) / 3.0, -1.0, 1.0) * dir_arr
        pos_z_arr = np.nan_to_num(pos_z_arr, nan=0.0).astype(np.float32)
        del r_mean, r_std

        # 1m 连续调仓
        pos_1m_cont_z = np.where(is_rollover_1m, 0.0, pos_z_arr).astype(np.float32)
        calc_metrics(pos_1m_cont_z, False, f'w{w}_z_1m_cont_nofee')
        calc_metrics(pos_1m_cont_z, True,  f'w{w}_z_1m_cont_fee')
        df_export[f'w{w}_pos_1m_cont_z'] = pos_1m_cont_z
        del pos_1m_cont_z

        # 1m 间隔调仓
        pos_1m_intra_z = np.where(is_eod_1m | is_rollover_1m, 0.0, pos_z_arr).astype(np.float32)
        calc_metrics(pos_1m_intra_z, False, f'w{w}_z_1m_intra_nofee')
        del pos_1m_intra_z

        # 30m 间隔调仓_ZScore
        raw_30m_z = np.where(rebalance_mask_30m, pos_z_arr, np.nan)
        raw_30m_z = np.where(is_eod_30m | is_rollover_30m, 0.0, raw_30m_z) 
        pos_30m_intra_z = pd.Series(raw_30m_z).groupby(df_bt['instrument'], observed=True).ffill().fillna(0.0).values.astype(np.float32)
        
        calc_metrics(pos_30m_intra_z, False, f'w{w}_z_30m_intra_nofee')
        calc_metrics(pos_30m_intra_z, True,  f'w{w}_z_30m_intra_fee')
        df_export[f'w{w}_pos_30m_intra_z'] = pos_30m_intra_z

        del raw_30m_z, pos_30m_intra_z, pos_z_arr
        gc.collect()

        # -------------------------------------------------------------
        # 映射 C：Robust Z-Score 稳健映射
        # -------------------------------------------------------------
        r_median = grouped_feat.transform(lambda x: x.rolling(w, min_periods=w//2).median()).values.astype(np.float32)
        r_q75 = grouped_feat.transform(lambda x: x.rolling(w, min_periods=w//2).quantile(0.75)).values.astype(np.float32)
        r_q25 = grouped_feat.transform(lambda x: x.rolling(w, min_periods=w//2).quantile(0.25)).values.astype(np.float32)
        r_robust_std = ((r_q75 - r_q25) / 1.349).astype(np.float32)
        del r_q75, r_q25
        
        pos_rob_arr = np.clip((feat_arr - r_median) / (r_robust_std + 1e-8) / 3.0, -1.0, 1.0) * dir_arr
        pos_rob_arr = np.nan_to_num(pos_rob_arr, nan=0.0).astype(np.float32)
        del r_median, r_robust_std

        # 1m 连续调仓 (稳健)
        pos_1m_cont_rob = np.where(is_rollover_1m, 0.0, pos_rob_arr).astype(np.float32)
        calc_metrics(pos_1m_cont_rob, False, f'w{w}_rob_1m_cont_nofee')
        calc_metrics(pos_1m_cont_rob, True,  f'w{w}_rob_1m_cont_fee')
        df_export[f'w{w}_pos_1m_cont_rob'] = pos_1m_cont_rob
        del pos_1m_cont_rob

        # 1m 间隔调仓 (稳健)
        pos_1m_intra_rob = np.where(is_eod_1m | is_rollover_1m, 0.0, pos_rob_arr).astype(np.float32)
        calc_metrics(pos_1m_intra_rob, False, f'w{w}_rob_1m_intra_nofee')
        del pos_1m_intra_rob

        # 30m 间隔调仓 (稳健)
        raw_30m_rob = np.where(rebalance_mask_30m, pos_rob_arr, np.nan)
        raw_30m_rob = np.where(is_eod_30m | is_rollover_30m, 0.0, raw_30m_rob)
        pos_30m_intra_rob = pd.Series(raw_30m_rob).groupby(df_bt['instrument'], observed=True).ffill().fillna(0.0).values.astype(np.float32)
        
        calc_metrics(pos_30m_intra_rob, False, f'w{w}_rob_30m_intra_nofee')
        calc_metrics(pos_30m_intra_rob, True,  f'w{w}_rob_30m_intra_fee')
        df_export[f'w{w}_pos_30m_intra_rob'] = pos_30m_intra_rob
        
        del raw_30m_rob, pos_30m_intra_rob, pos_rob_arr
        gc.collect()

        # -------------------------------------------------------------
        # 映射 B： Extreme 10% 离散极值映射
        # -------------------------------------------------------------
        r_q95 = grouped_feat.transform(lambda x: x.rolling(w, min_periods=w//2).quantile(0.9)).values.astype(np.float32)
        r_q05 = grouped_feat.transform(lambda x: x.rolling(w, min_periods=w//2).quantile(0.1)).values.astype(np.float32)
        
        pos_q_arr = np.zeros(len(df_bt), dtype=np.float32)
        pos_q_arr = np.where(feat_arr > r_q95, 1.0, pos_q_arr).astype(np.float32)
        pos_q_arr = np.where(feat_arr < r_q05, -1.0, pos_q_arr).astype(np.float32)
        pos_q_arr = (pos_q_arr * dir_arr).astype(np.float32)
        del r_q95, r_q05
        
        # 30m 间隔调仓_极值映射
        raw_30m_q = np.where(rebalance_mask_30m, pos_q_arr, np.nan)
        raw_30m_q = np.where(is_eod_30m | is_rollover_30m, 0.0, raw_30m_q)
        pos_30m_intra_q = pd.Series(raw_30m_q).groupby(df_bt['instrument'], observed=True).ffill().fillna(0.0).values.astype(np.float32)
        
        calc_metrics(pos_30m_intra_q, False, f'w{w}_q_30m_intra_nofee')
        calc_metrics(pos_30m_intra_q, True,  f'w{w}_q_30m_intra_fee')
        df_export[f'w{w}_pos_30m_intra_q'] = pos_30m_intra_q
        
        del raw_30m_q, pos_30m_intra_q, pos_q_arr
        gc.collect()

    del df_bt
    return inst_results, df_export

# =========================================================
# ⚙️ 5. Multiprocessing Worker 入口函数 (局部无重叠版)
# =========================================================
def process_single_feature(feature_name):
    try:
        # 只读所需列防 OOM
        cols_to_load = ['index_time', 'instrument', 'contract', 'Ret_1m_Fwd', 'Ret_15m_Fwd', 'Ret_30m_Fwd', feature_name]
        df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)

        df['instrument'] = df['instrument'].astype('category')
        df['contract'] = df['contract'].astype('category')
        for c in ['Ret_1m_Fwd', 'Ret_15m_Fwd', 'Ret_30m_Fwd', feature_name]:
            if c in df.columns:
                df[c] = df[c].astype('float32')
        
        df = df[df['instrument'].isin(VALID_INSTRUMENTS)].copy()
        df['index_time'] = pd.to_datetime(df['index_time'])
        
        is_night = df['index_time'].dt.hour >= 18
        df['Date'] = np.where(
            is_night, 
            (df['index_time'] + pd.Timedelta(days=1)).dt.date, 
            df['index_time'].dt.date
        )
        df['Half_Year'] = np.where(pd.to_datetime(df['Date']).dt.month <= 6, 'H1', 'H2')
        df['Quarter'] = 'Q' + pd.to_datetime(df['Date']).dt.quarter.astype(str)

        df = filter_global_trading_hours(df)

        unique_insts = df['instrument'].unique()
        df_cffex = df[df['instrument'].isin(CFFEX_POOL)].copy()
        df_comm = df[df['instrument'].isin(COMM_POOL)].copy()
        
        del df
        gc.collect()

        feature_result = {"Pools": {"CFFEX": {}, "COMMODITY": {}}, "Instruments": {}}
        
        # 1. 遍历单品种计算指标与方向
        inst_directions = {}
        for inst in unique_insts:
            feature_result["Instruments"][inst] = {}
            if inst in CFFEX_POOL:
                df_inst = df_cffex[df_cffex['instrument'] == inst]
            else:
                df_inst = df_comm[df_comm['instrument'] == inst]
                
            for tgt in TARGETS:
                res = calc_metrics_and_charts(df_inst, feature_name, tgt)
                feature_result["Instruments"][inst][tgt] = res
                
                if res and tgt == 'Ret_30m_Fwd':
                    ic = res.get('full_ic', 0.0)
                    inst_directions[inst] = 1.0 if ic >= 0 else -1.0
            
            if inst not in inst_directions:
                inst_directions[inst] = 1.0

        # 2. 回测计算
        if not df_cffex.empty:
            cffex_backtests, df_export_cffex = run_multi_mode_ts_backtest(df_cffex, feature_name, FEE_CFFEX, inst_directions)
        else:
            cffex_backtests, df_export_cffex = {}, pd.DataFrame()
            
        if not df_comm.empty:
            comm_backtests, df_export_comm = run_multi_mode_ts_backtest(df_comm, feature_name, FEE_COMM, inst_directions)
        else:
            comm_backtests, df_export_comm = {}, pd.DataFrame()

        # 3. 分发回测结果
        for inst in unique_insts:
            if inst in CFFEX_POOL:
                feature_result["Instruments"][inst]["Backtest"] = cffex_backtests.get(inst, {})
            else:
                feature_result["Instruments"][inst]["Backtest"] = comm_backtests.get(inst, {})

        # 4. 全局聚合指标
        for tgt in TARGETS:
            if not df_cffex.empty:
                feature_result["Pools"]["CFFEX"][tgt] = calc_metrics_and_charts(df_cffex, feature_name, tgt)
            if not df_comm.empty:
                feature_result["Pools"]["COMMODITY"][tgt] = calc_metrics_and_charts(df_comm, feature_name, tgt)

        df_export_final = pd.concat([df_export_cffex, df_export_comm], ignore_index=True)
        if not df_export_final.empty:
            df_export_final = df_export_final.sort_values(['instrument', 'index_time']).reset_index(drop=True)
            out_file = os.path.join(POSITIONS_DIR, f"{feature_name}.parquet")
            df_export_final.to_parquet(out_file, engine='pyarrow', compression='snappy')

        del df_cffex, df_comm, df_export_cffex, df_export_comm, df_export_final
        gc.collect()

        return feature_name, feature_result

    except Exception as e:
        return feature_name, {"Error": str(e)}

# =========================================================
# 🚀 6. 发射 Local Process Pool
# =========================================================
if __name__ == "__main__":
    import pyarrow.parquet as pq

    print("🚀 Extracting Feature List from Parquet Metadata via PyArrow...")
    
    schema = pq.read_schema(PARQUET_PATH)
    all_columns = schema.names
    
    exclude_cols = ['index_time', 'instrument', 'contract', 'time', 'Date', 'Half_Year', 'Quarter', 'Bin', 'Ret_1m_Fwd'] + TARGETS
    TRADABILITY_COLS = ['L5_Depth_Absorbing_Ratio_60', 'Taker_Impact_Sweep_L5', 'Exec_Granularity_Ratio_60', 'True_Matching_Heat_Rank_120']
    
    state_features = []
    if os.path.exists(MARKET_STATES_PATH):
        with open(MARKET_STATES_PATH, 'r') as f:
            state_features = json.load(f)
            print(f"🛡 成功加载并排除了 {len(state_features)} 个状态变量")
    
    features = [col for col in all_columns if col not in exclude_cols + TRADABILITY_COLS + state_features]
    
    print(f"📦 Total Features to Compute: {len(features)}")
    
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"🔥 Submitting map tasks to Local Multi-Process Pool (Workers: {max_workers})...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(process_single_feature, features))
    
    final_output = {}
    for feat_name, data_payload in results:
        if "Error" not in data_payload:
            final_output[feat_name] = data_payload
        else:
            print(f"❌ Feature Failed: {feat_name} | Error: {data_payload['Error']}")

    import simplejson
    
    def np_encoder(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    print("💾 正在极速且安全地写入 JSON 文件...")
    
    try:
        with open(OUTPUT_JSON, "w") as f:
            simplejson.dump(final_output, f, ignore_nan=True, default=np_encoder)
    except Exception as e:
        import json
        with open(OUTPUT_JSON, "w") as f:
            json.dump(final_output, f, default=np_encoder)
            
    print(f"🎉 All Done! Safe & Clean Dashboard Data saved to: {OUTPUT_JSON}")