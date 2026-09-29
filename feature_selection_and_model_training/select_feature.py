#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import datetime
import networkx as nx
from networkx.algorithms import community
import pandas as pd
import numpy as np
from collections import Counter
import pyarrow.parquet as pq
from scipy.stats import pearsonr
import gc

# ============================================================
# 🎛️ 0. 路径与核心超参数配置
# ============================================================
from pathlib import Path

# 动态获取项目根目录
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = project_root / "data"

DASHBOARD_2024_PATH = str(DATA_DIR / "feature_quality_dashboard_2024.json") 
CORR_MATRIX_PATH = str(DATA_DIR / "feature_corr_matrix.csv")

OUT_DIR = DATA_DIR / "pooled_inst_features"
os.makedirs(OUT_DIR, exist_ok=True)

# Parquet 底层数据路径 (用于最终决战的混池 Full Year IC 计算)
PARQUET_2024 = str(DATA_DIR / "factor_dataset_AllMarket_main_only_2024.parquet")
TARGET = "Ret_30m_Fwd"
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

# 动态组合出合法的候选池 (商品为例)
TARGET_POOL = list(set(COMM_RAW) - EXCLUDED_INSTRUMENTS)
print(f"🎯 Target Pool Selected: {TARGET_POOL}")

# 💡 2. 社区覆盖率超参数
MIN_COMMUNITY_COVERAGE_RATIO = 0.2  

# 💡 单品种初筛超参数
MIN_CONSISTENT_QUARTERS = 3      
SELECT_METRIC_GROUP = "Ret_30m_Fwd"
SELECT_METRIC = "full_ic"  

CORR_THRESHOLD = 0.7            
FINAL_TOP_N = 200               
FINAL_MIN_METRIC = 0.01         

USE_ABS_FOR_SELECT = True if "ic" in SELECT_METRIC.lower() else False

# ============================================================
# 🛠️ 1. 核心 JSON 抓取与过滤函数 (保留单品种初筛逻辑)
# ============================================================
def get_val(feat_name, inst, dash, m_group, m_name):
    feat_data = dash.get(feat_name) or {}
    inst_data = feat_data.get("Instruments") or {}
    spec_data = inst_data.get(inst) or {}
    group_data = spec_data.get(m_group) or {}
    return group_data.get(m_name, None)

def get_avg_val(feat_name, inst, dash_24, m_group, m_name):
    v24 = get_val(feat_name, inst, dash_24, m_group, m_name)
    valid_vals = [v24] if v24 is not None and not np.isnan(v24) else []
    return np.mean(valid_vals) if valid_vals else None

def get_quarterly_values(dash, feat_name, inst, m_group, base_metric):
    feat_data = dash.get(feat_name) or {}
    inst_data = feat_data.get("Instruments") or {}
    spec_data = inst_data.get(inst) or {}
    group_data = spec_data.get(m_group) or {}
    
    if m_group == "Backtest":
        if "_" in base_metric:
            prefix, m_type = base_metric.rsplit('_', 1) 
            q_keys = [f"{prefix}_{q}_{m_type}" for q in ['Q1', 'Q2', 'Q3', 'Q4']]
        else: q_keys = []
    else:
        q_keys = [f"{q}_ic" for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    return [group_data.get(qk) for qk in q_keys]

def check_4q_consistency(dash_24, feat_name, inst, m_group, base_metric, min_consistent):
    vals_24 = get_quarterly_values(dash_24, feat_name, inst, m_group, base_metric)
    
    all_q_vals = vals_24
    pos_count, neg_count = 0, 0
    for val in all_q_vals:
        if val is not None and not np.isnan(val) and val != 0:
            if val > 0: pos_count += 1
            elif val < 0: neg_count += 1
                
    return max(pos_count, neg_count) >= min_consistent

# =========================================================
# ⏱️ 2. 全市场高精度时间清洗 (精准对齐回测与训练引擎)
# =========================================================
def filter_global_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    time_str = df['index_time'].dt.strftime('%H:%M:%S')
    opening_bars = {'09:00:00', '09:30:00', '21:00:00'}
    
    is_cffex = df['instrument'].isin(CFFEX_RAW)
    mask_cffex = (((time_str >= '09:30:00') & (time_str <= '11:30:00')) | ((time_str >= '13:00:00') & (time_str <= '15:00:00')))
    
    is_commodity = ~is_cffex
    mask_comm_day = (((time_str >= '09:00:00') & (time_str <= '10:15:00')) | ((time_str >= '10:30:00') & (time_str <= '11:30:00')) | ((time_str >= '13:30:00') & (time_str <= '15:00:00')))
    
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

    mask_comm = mask_comm_day | ((is_0230 & mask_night_0230) | (is_0100 & mask_night_0100) | (is_2300 & mask_night_2300))
    valid_mask = ((is_cffex & mask_cffex) | (is_commodity & mask_comm)) & (~time_str.isin(opening_bars))
    return df[valid_mask].copy()

# ============================================================
# 🕸️ 3. 加载数据 & 社区发现 (全局只算一次)
# ============================================================
print(f"📦 正在联合加载 2024 年的特征质量 Dashboard JSON...")
with open(DASHBOARD_2024_PATH, 'r') as f:
    dash_2024 = json.load(f)

print("🕸️ 构建特征相关图与全局社区发现...")
corr_df = pd.read_csv(CORR_MATRIX_PATH, index_col=0)
corr_matrix = corr_df.fillna(0.0).abs()
features = corr_df.columns.tolist()

G = nx.Graph()
G.add_nodes_from(features)
edges = [(features[i], features[j], corr_matrix.iloc[i, j]) 
         for i in range(len(features)) for j in range(i + 1, len(features)) 
         if corr_matrix.iloc[i, j] > CORR_THRESHOLD]
G.add_weighted_edges_from(edges)
communities = list(community.greedy_modularity_communities(G, weight='weight'))
print(f"✅ 在 {len(features)} 个特征中，全局提取出 {len(communities)} 个互斥社区。")

# ============================================================
# 🏆 4. 多品种初筛与投票统计 (保存候选特征)
# ============================================================
print(f"\n🔍 开始针对品种池 {TARGET_POOL} 进行社区海选与投票...")
print(f"   - 社区有效覆盖率门槛: {MIN_COMMUNITY_COVERAGE_RATIO*100}%")

community_tied_info = []
unique_candidates = set()

for comm_idx, comm in enumerate(communities):
    inst_best_feats = {}
    
    for inst in TARGET_POOL:
        valid_feats = [feat for feat in comm if check_4q_consistency(dash_2024, feat, inst, SELECT_METRIC_GROUP, SELECT_METRIC, MIN_CONSISTENT_QUARTERS)]
        if not valid_feats: continue
            
        best_feat, best_score = None, -np.inf
        for feat in valid_feats:
            s_val = get_avg_val(feat, inst, dash_2024, SELECT_METRIC_GROUP, SELECT_METRIC)
            if s_val is not None and not np.isnan(s_val):
                score = abs(s_val) if USE_ABS_FOR_SELECT else s_val
                if score > best_score:
                    best_score = score
                    best_feat = feat

        if best_feat is not None and best_score >= FINAL_MIN_METRIC:
            inst_best_feats[inst] = best_feat

    coverage_ratio = len(inst_best_feats) / len(TARGET_POOL)
    if coverage_ratio < MIN_COMMUNITY_COVERAGE_RATIO:
        continue  

    vote_counts = Counter(inst_best_feats.values())
    max_votes = max(vote_counts.values())
    tied_feats = [f for f, v in vote_counts.items() if v == max_votes]
    
    community_tied_info.append({
        'comm_idx': comm_idx,
        'tied_feats': tied_feats,
        'coverage': coverage_ratio,
        'votes': max_votes
    })
    unique_candidates.update(tied_feats)

unique_candidates = list(unique_candidates)

# ============================================================
# 💥 5. 终极对决：截取严格可交易样本，计算混池 Pooled IC
# ============================================================
print(f"\n⚙️ 正在提取底层 Parquet，准备对 {len(unique_candidates)} 个候选特征进行深度清洗及 Pooled IC 计算...")
pooled_ic_dict = {}

if unique_candidates:
    schema = pq.read_schema(PARQUET_2024)
    valid_cols = [c for c in unique_candidates if c in schema.names]
    load_cols = ['index_time', 'instrument', 'contract', TARGET] + valid_cols
    
    # 加载 TARGET_POOL 在两年内的原始数据
    df_24 = pq.read_table(PARQUET_2024, columns=load_cols, filters=[('instrument', 'in', TARGET_POOL)]).to_pandas()
    df_pool = df_24.copy()
    df_pool['index_time'] = pd.to_datetime(df_pool['index_time'])
    del df_24; gc.collect()

    print("   ➤ 正在执行基础交易时段过滤...")
    df_pool = filter_global_trading_hours(df_pool)

    print("   ➤ 正在执行 30-min 严格屏蔽 (尾盘/长假/换月)...")
    # 按照品种和时间排序，确保 rolling 逻辑正确
    df_pool = df_pool.sort_values(['instrument', 'index_time']).reset_index(drop=True)
    
    bt = df_pool['index_time'].dt.time
    df_pool['is_eod_1m'] = (bt >= datetime.time(14, 59, 0)) & (bt <= datetime.time(15, 0, 0))
    eod30 = df_pool.groupby('instrument')['is_eod_1m'].transform(
        lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]
    ).fillna(0).astype(bool)

    df_pool['contract_tmp'] = df_pool['contract'].fillna('Unknown')
    df_pool['contract_change'] = (df_pool['instrument'] == df_pool['instrument'].shift(-1)) & \
                                 (df_pool['contract_tmp'] != df_pool['contract_tmp'].shift(-1))
    roll30 = df_pool.groupby('instrument')['contract_change'].transform(
        lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]
    ).fillna(0).astype(bool)

    t_next = df_pool.groupby('instrument')['index_time'].shift(-1)
    gap_hours = (t_next - df_pool['index_time']).dt.total_seconds() / 3600.0
    day_gap = (t_next.dt.normalize() - df_pool['index_time'].dt.normalize()).dt.days
    is_night = ((bt >= datetime.time(20, 0)) | (bt <= datetime.time(3, 0)))
    night_end = ((gap_hours > 8.0) & is_night & (day_gap >= 2)).fillna(False)
    eod_night30 = df_pool.assign(ne=night_end).groupby('instrument')['ne'].transform(
        lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]
    ).fillna(0).astype(bool)

    # 剔除无效交易时间
    df_pool['is_traded'] = ~(eod30 | roll30 | eod_night30)
    df_clean = df_pool[df_pool['is_traded']].reset_index(drop=True)
    del df_pool; gc.collect()

    print(f"   ➤ 深度清洗完成，获得 {len(df_clean):,} 行纯净样本。正在计算真实横截面 Pooled IC...")
    
    # 拆分年份来算平均 IC，保证和原来按季度或年度评价的尺度一致
    mask_24 = df_clean['index_time'].dt.year == 2024
    mask_25 = df_clean['index_time'].dt.year == 2025

    for feat in unique_candidates:
        if feat not in valid_cols:
            pooled_ic_dict[feat] = 0.0
            continue
            
        ic24 = np.nan
        # 2024 Pooled IC
        m24 = df_clean[feat].notna() & df_clean[TARGET].notna()
        if m24.sum() > 1000:
            fv, tv = df_clean.loc[m24, feat].values, df_clean.loc[m24, TARGET].values
            if np.std(fv) > 1e-8 and np.std(tv) > 1e-8:
                ic24 = pearsonr(fv, tv)[0]
                
        pooled_ic_dict[feat] = ic24 if not np.isnan(ic24) else 0.0
        
    del df_clean; gc.collect()

# ============================================================
# 🏁 6. 基于混池 Pooled IC 打破平局并全局排序
# ============================================================
pool_candidates = []

for info in community_tied_info:
    winner_feat = None
    winner_score = -np.inf
    winner_raw = 0.0
    
    for feat in info['tied_feats']:
        raw_ic = pooled_ic_dict.get(feat, 0.0)
        # 因为我们使用的是 IC，所以无论如何都需要取绝对值进行强弱评判
        score = abs(raw_ic) if USE_ABS_FOR_SELECT else raw_ic
        
        if score > winner_score:
            winner_score = score
            winner_raw = raw_ic
            winner_feat = feat
            
    if winner_feat is not None:
        pool_candidates.append({
            "community_idx": info['comm_idx'], 
            "feature": winner_feat,
            "coverage_ratio": info['coverage'],
            "votes": info['votes'],
            f"pool_score_{SELECT_METRIC}": winner_score, 
            f"pool_raw_mean_{SELECT_METRIC}": winner_raw
        })

# 按真实的跨品种混池分数全局排序
pool_candidates.sort(key=lambda x: x[f"pool_score_{SELECT_METRIC}"], reverse=True)
final_reps = pool_candidates[:FINAL_TOP_N]

if len(final_reps) > 0:
    print(f"\n" + "="*125)
    print(f"🏆 混池子最终入选通用特征 (展示 Top 20 / 共选出 {len(final_reps)} 个满足严苛普适性的特征)")
    print("-" * 125)
    print(f"{'Rank':<4} | {'CommID':<6} | {'Feature Name':<45} | {'Coverage':<9} | {'Votes':<5} | {'Pooled IC Score':<15} | {'Pooled Raw IC':<15}")
    print("-" * 125)
    
    for rank, rep in enumerate(final_reps[:20]):
        feat = rep['feature']
        if len(feat) > 42: feat = feat[:39] + "..."
            
        cov_str = f"{rep['coverage_ratio']*100:.0f}%"
        votes = rep['votes']
        score_val = rep[f"pool_score_{SELECT_METRIC}"]
        raw_mean_val = rep[f"pool_raw_mean_{SELECT_METRIC}"]
        
        print(f"{rank+1:<4} | {rep['community_idx']:<6} | {feat:<45} | {cov_str:<9} | {votes:<5} | {score_val:<15.4f} | {raw_mean_val:>+15.4f}")
    print("="*125)

# 保存至文件
feature_names = [r["feature"] for r in final_reps]
pool_name = "Pooled_" + "_".join(TARGET_POOL)
if len(pool_name) > 50: pool_name = "Pooled_Custom_Set"

out_path = os.path.join(OUT_DIR, f"{pool_name}_best_features.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"Pool": TARGET_POOL, "Feature": feature_names}, f, indent=2)
    
detail_path = os.path.join(OUT_DIR, f"{pool_name}_best_features_detail.json")
with open(detail_path, "w", encoding="utf-8") as f:
    json.dump(final_reps, f, indent=2)
    
print(f"\n🎉 混池通用因子严苛筛选完成 (依据严格清洗后的 Pooled Full Year IC 排序)！文件已保存至: {OUT_DIR}")