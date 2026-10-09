# -*- coding: utf-8 -*-
"""
多品种混池 LightGBM 零训练基线 (Pooled Multi-Instrument Framework)
- 核心特性: 支持全局混合样本的 Z-Score / Quantile 滚动映射。
- 核心特性: 支持基于混合样本内 (IS) 的全局静态绝对分位数锁仓映射。
- 支持自动组合级别的回测指标 (Portfolio Level Metrics) 与单品种切割。
"""
import warnings
warnings.filterwarnings('ignore')
import gc
import json
import datetime
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

# =======================================================
# 🎛️ 1. 全局配置与参数 (Multi-Instrument Config)
# =======================================================
# 💡 [修改这里] 填入你要混池训练的品种列表

TARGET_POOL = ["GLZ","SLZ"]

TRADING_MINUTE = 1
POS_NEG_WEIGHT = False
USE_CV = True
CV_FOLDS = 10         
DYNAMIC_SIGN = False
USE_SCALER = False
ZSCORE_SCALE = 1.5

# 💡 新增：全局分位数映射阈值 (Upper 和 Lower)
Q_UPPER_LOCK = 0.90  # 即 90%
Q_LOWER_LOCK = 0.10  # 即 10%

# 2. Robust 连续仓位映射策略 (IS_Rob) 的极值锚点
Q_UPPER_ROB = 0.95   # 95% (映射为满仓 +1.0)
Q_LOWER_ROB = 0.05   #  5% (映射为满仓 -1.0)


LGB_PARAMS = dict(
    n_estimators=2000,
    objective='regression',
    learning_rate=0.01,
    max_depth=10,
    num_leaves=63,
    min_data_in_leaf=500,  # 混品种数据量巨大，叶子节点可适当调大
    feature_fraction=0.6,
    bagging_fraction=0.7,
    bagging_freq=5,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

TARGET = "Ret_30m_Fwd"
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = project_root / "data"

PARQUET_2024 = str(DATA_DIR / "factor_dataset_AllMarket_main_only_2024.parquet")
PARQUET_2025 = str(DATA_DIR / "factor_dataset_AllMarket_main_only_2025.parquet")
MARKET_STATES_PATH = str(DATA_DIR / "best_market_states_list.json")

# 💡 [修改这里] 填入你要混池训练的标准品种列表
TARGET_POOL = ["AU", "AG"]  # 原 GLZ, SLZ

# 标准化股指代码
CFFEX_RAW = ['IM', 'IC', 'IF', 'IH']
CFFEX_INDEX = ['IF', 'IH', 'IC', 'IM']
CFFEX_BOND  = ['TS', 'TF', 'T', 'TL']
BLACK_SERIES = ['RB', 'HC', 'I', 'J', 'JM']  # 原 DBZ 等

# =======================================================
# 🎛️ 终极领域知识字典：全市场独立开平费率与MPD (已含隐性滑点)
# =======================================================
# 💡 隐性成本
SLIPPAGE = 0.00005  

# 1. 基础开仓费率映射表 (万分之)
# 1. 基础开仓费率映射表 (万分之)
base_open_fee = {
    # --- 股指 ---
    'IM': 0.000023, 'IC': 0.000023, 'IF': 0.000023, 'IH': 0.000023,
    # --- 黑色系 ---
    'RB': 0.0001, 'HC': 0.0001, 'I': 0.0001, 'J': 0.0001, 'JM': 0.0001,
    'SM': 0.00001, 'SF': 0.00001, 
    # --- 贵金属 ---
    'AG': 0.00001,  
    'AU': 0.000016, 
    # --- 有色金属 ---
    'CU': 0.00005, 'AL': 0.00005, 'ZN': 0.00005, 'PB': 0.00004,  
    'NI': 0.0003, 'SN': 0.0003, 'SS': 0.0001, 'LC': 0.00008, 'AO': 0.0001
}

# 2. 基础平今日费率映射表 (万分之)
base_close_today_fee = {
    # --- 股指 (天价平今惩罚) ---
    'IF': 0.000345, 'IH': 0.000345, 'IM': 0.00023, 'IC': 0.00023,   
    # --- 黑色系 (照收不误) ---
    'RB': 0.0001, 'HC': 0.0001, 'I': 0.0001, 'J': 0.0001, 'JM': 0.0001, 
    # --- 贵金属/合金 (享受平今免) ---
    'AG': 0.0, 'AU': 0.0, 'SM': 0.0, 'SF': 0.0,                             
    # --- 有色金属 (基本照收不误) ---
    'CU': 0.00005, 'AL': 0.00005, 'ZN': 0.00005, 'PB': 0.00004, 
    'NI': 0.0003, 'SN': 0.0003, 'SS': 0.0001, 'LC': 0.00008, 'AO': 0.0001
}

# 3. 融合滑点生成最终引擎字典
open_fee_dict = {inst: (base_open_fee.get(inst, 0.0001) + SLIPPAGE) for inst in TARGET_POOL}
close_fee_dict = {inst: (base_close_today_fee.get(inst, 0.0001) + SLIPPAGE) for inst in TARGET_POOL}

# 4. 每日交易分钟数 (精确划分长短夜盘)
# 缺省值为 345 (普通化工/农产品，23:00收盘)
mpd_bars_dict = {}
for inst in TARGET_POOL:
    if inst in ['IM', 'IC', 'IF', 'IH']:
        mpd_bars_dict[inst] = 240
    elif inst in ['AG', 'AU', 'SC']:  
        mpd_bars_dict[inst] = 555
    elif inst in ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'SS', 'BC', 'LC', 'AO']: 
        mpd_bars_dict[inst] = 465
    else:
        mpd_bars_dict[inst] = 345  # 黑色系及其他 (23:00收盘)

STOP_ROUND = 10

# =========================================================
# ⏱️ 2. 全市场高精度时间清洗
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

# =======================================================
# 3. 加载共用混池特征
# =======================================================
pool_name = "Pooled_" + "_".join(TARGET_POOL)
if len(pool_name) > 50: pool_name = "Pooled_Custom_Set"
SELECTED_FEATURES_PATH = f"{DATA_DIR}/pooled_inst_features/{pool_name}_best_features.json"

print(f"⏳ 正在加载混池通用特征: {SELECTED_FEATURES_PATH} ...")
try:
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        feature_data = json.load(f)
    pool_features = feature_data.get("Feature", [])
except FileNotFoundError:
    raise FileNotFoundError("找不到池化特征 JSON，请确认前一步已执行完毕。")

state_features = []
if os.path.exists(MARKET_STATES_PATH):
    with open(MARKET_STATES_PATH, 'r') as f:
        state_features = json.load(f)
#+ state_features
selected_features = list(dict.fromkeys(pool_features ))
print(f"🎯 选定训练池: {TARGET_POOL} | 最终使用特征数: {len(selected_features)}")

all_cols = pq.read_schema(PARQUET_2024).names
cols = ['index_time', 'instrument', 'contract', TARGET, 'Ret_1m_Fwd', 'Ret_15m_Fwd'] + selected_features
cols = [c for c in cols if c in all_cols]
features = [f for f in selected_features if f in cols]

print(f"⚙️ 正在加载 2024 和 2025 年多品种数据...")
df_24 = pq.read_table(PARQUET_2024, columns=cols, filters=[('instrument', 'in', TARGET_POOL)]).to_pandas()
df_25 = pq.read_table(PARQUET_2025, columns=cols, filters=[('instrument', 'in', TARGET_POOL)]).to_pandas()
df = pd.concat([df_24, df_25], ignore_index=True)
df['index_time'] = pd.to_datetime(df['index_time'])

df = filter_global_trading_hours(df)
df = df.dropna(subset=[TARGET, 'Ret_1m_Fwd']).reset_index(drop=True)

VALID_START_DATE = "2024-01-08"
df = df[df['index_time'] >= pd.to_datetime(VALID_START_DATE)]

# =======================================================
# 4. 品种隔离全局掩码对齐
# =======================================================
df['Weight'] = 1.0 
# 注意：构建交易掩码时，必须先按 品种 再按 时间 排序！
df = df.sort_values(['instrument', 'index_time']).reset_index(drop=True)

bt = df['index_time'].dt.time
df['is_eod_1m'] = (bt >= datetime.time(14, 59, 0)) & (bt <= datetime.time(15, 0, 0))

def rev_roll_max(x): return x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1].astype(bool)

df['is_eod_30m'] = df.groupby('instrument')['is_eod_1m'].transform(rev_roll_max)
df['contract_tmp'] = df['contract'].fillna('Unknown')
df['contract_change'] = (df['instrument'] == df['instrument'].shift(-1)) & (df['contract_tmp'] != df['contract_tmp'].shift(-1))
df.drop(columns=['contract_tmp'], inplace=True)

df['is_rollover_1m'] = df.groupby('instrument')['contract_change'].transform(lambda x: x.shift(-1).fillna(False) | x)
df['is_rollover_30m'] = df.groupby('instrument')['contract_change'].transform(rev_roll_max)

df['bi'] = df.groupby('instrument').cumcount()
df['is_traded'] = ~(df['is_eod_30m'] | df['is_rollover_30m'])

for col in ['Ret_1m_Fwd', 'Ret_15m_Fwd', 'Ret_30m_Fwd']:
    if col in df.columns: df[col] = df[col].clip(lower=-1, upper=1)

# =======================================================
# 5. 🚀 滚动训练引擎 (多品种全局分布融合)
# =======================================================
if POS_NEG_WEIGHT:
    pos_mask, neg_mask = (df[TARGET] > 0), (df[TARGET] < 0)
    n_total = pos_mask.sum() + neg_mask.sum()
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        df.loc[pos_mask, 'Weight'] = (n_total / 2.0) / pos_mask.sum()
        df.loc[neg_mask, 'Weight'] = (n_total / 2.0) / neg_mask.sum()

roll_periods = [
    ('2025_Q1', '2025-01-05', '2025-03-31'), ('2025_Q2', '2025-04-01', '2025-06-30'),
    ('2025_Q3', '2025-07-01', '2025-09-30'), ('2025_Q4', '2025-10-01', '2025-12-31')
]
DROP_MONTHS = 3
train_start = pd.to_datetime(VALID_START_DATE)

df['Pred_OOS'], df['Pred_IS'] = np.nan, np.nan
df['Fold_IS_Mean'] = np.nan
df['Fold_IS_Std']  = np.nan
df['Fold_IS_Q90']  = np.nan
df['Fold_IS_Q10']  = np.nan

print(f"\n🚀 开始多品种混池滚动训练...")
for i, (q_name, start_dt, end_dt) in enumerate(roll_periods):
    train_mask = (df['index_time'] >= train_start) & (df['index_time'] < pd.to_datetime(start_dt))
    test_mask = (df['index_time'] >= pd.to_datetime(start_dt)) & (df['index_time'] <= pd.to_datetime(end_dt + ' 23:59:59'))
    if not test_mask.any(): continue

    print(f"\n   ➤ 正在训练组合模型预测 [{q_name}] ...")
    
    scaler = StandardScaler() if USE_SCALER else None
    if USE_SCALER: scaler.fit(df.loc[train_mask, features].values)

    train_dates = np.sort(df.loc[train_mask, 'index_time'].dt.date.unique())
    models = []
    
    date_blocks = np.array_split(train_dates, CV_FOLDS)
    for fold, val_dates in enumerate(date_blocks):
        if (not USE_CV) and (fold != (CV_FOLDS-1)): continue
        
        val_dates_set = set(val_dates)
        fold_val_mask = train_mask & df['index_time'].dt.date.isin(val_dates_set)
        fold_fit_mask = train_mask & ~fold_val_mask
        
        X_fit, y_fit, w_fit = df.loc[fold_fit_mask, features].values, df.loc[fold_fit_mask, TARGET].values, df.loc[fold_fit_mask, 'Weight'].values
        X_val, y_val = df.loc[fold_val_mask, features].values, df.loc[fold_val_mask, TARGET].values
        
        if USE_SCALER:
            X_tr_np, X_val_np = scaler.transform(X_fit).astype(np.float32), scaler.transform(X_val).astype(np.float32)
        else:
            X_tr_np, X_val_np = np.asarray(X_fit, dtype=np.float32), np.asarray(X_val, dtype=np.float32)
        y_tr_np, w_tr_np, y_val_np = np.asarray(y_fit), np.asarray(w_fit), np.asarray(y_val)

        model = lgb.LGBMRegressor(**LGB_PARAMS).set_params(metric='rmse')  
        model.fit(X_tr_np, y_tr_np, sample_weight=w_tr_np, eval_set=[(X_val_np, y_val_np)], callbacks=[lgb.early_stopping(STOP_ROUND, verbose=False)])
        n_trees = model.best_iteration_ if model.best_iteration_ else model.n_estimators_
        model = model.set_params(n_estimators=n_trees) 
        models.append(model)
        print(f"         📌 [Fold {fold+1}] Best Trees: {n_trees:<4} | Val RMSE: {model.best_score_['valid_0']['rmse']:.6f}")

    is_preds_acc = np.zeros(train_mask.sum())
    oos_preds_acc = np.zeros(test_mask.sum())

    X_train_infer = scaler.transform(df.loc[train_mask, features].values).astype(np.float32) if USE_SCALER else df.loc[train_mask, features].values
    X_test_infer = scaler.transform(df.loc[test_mask, features].values).astype(np.float32) if USE_SCALER else df.loc[test_mask, features].values
    
    for m in models:
        best_n = m.best_iteration_ if m.best_iteration_ else m.n_estimators_
        is_preds_acc += m.predict(X_train_infer, num_iteration=best_n)
        oos_preds_acc += m.predict(X_test_infer, num_iteration=best_n)
    
    is_preds = is_preds_acc / len(models)
    oos_preds = oos_preds_acc / len(models)

    df.loc[test_mask, 'Pred_OOS'] = oos_preds
    if i == 0: df.loc[train_mask, 'Pred_IS'] = is_preds

    # === 🌟 提取当前确切的训练集分布，赋给对应的 OOS 和 IS ===
    fold_mean = np.mean(is_preds)
    fold_std = max(np.std(is_preds), 1e-8)  
    
    # 动态使用全局超参数计算分位数
    fold_q_up_lock = np.percentile(is_preds, Q_UPPER_LOCK * 100)
    fold_q_dn_lock = np.percentile(is_preds, Q_LOWER_LOCK * 100)
    fold_q_up_rob = np.percentile(is_preds, Q_UPPER_ROB * 100)
    fold_q_dn_rob = np.percentile(is_preds, Q_LOWER_ROB * 100)
    fold_med = np.median(is_preds)
    
    df.loc[test_mask, 'Fold_IS_Mean'] = fold_mean
    df.loc[test_mask, 'Fold_IS_Std']  = fold_std
    df.loc[test_mask, 'Fold_IS_Q_Up_Lock'] = fold_q_up_lock
    df.loc[test_mask, 'Fold_IS_Q_Dn_Lock'] = fold_q_dn_lock
    df.loc[test_mask, 'Fold_IS_Q_Up_Rob']  = fold_q_up_rob
    df.loc[test_mask, 'Fold_IS_Q_Dn_Rob']  = fold_q_dn_rob
    df.loc[test_mask, 'Fold_IS_Med']  = fold_med
    
    if i == 0:
        df.loc[train_mask, 'Fold_IS_Mean'] = fold_mean
        df.loc[train_mask, 'Fold_IS_Std']  = fold_std
        df.loc[train_mask, 'Fold_IS_Q_Up_Lock'] = fold_q_up_lock
        df.loc[train_mask, 'Fold_IS_Q_Dn_Lock'] = fold_q_dn_lock
        df.loc[train_mask, 'Fold_IS_Q_Up_Rob']  = fold_q_up_rob
        df.loc[train_mask, 'Fold_IS_Q_Dn_Rob']  = fold_q_dn_rob
        df.loc[train_mask, 'Fold_IS_Med']  = fold_med
        
    # 📈 [IC] & 📊 [Plots]
    target_is, target_oos = df.loc[train_mask, TARGET].values, df.loc[test_mask, TARGET].values
    ic_is = pearsonr(is_preds, target_is)[0] if len(is_preds) > 30 and np.std(is_preds) > 1e-8 else np.nan
    ic_oos = pearsonr(oos_preds, target_oos)[0] if len(oos_preds) > 30 and np.std(oos_preds) > 1e-8 else np.nan
    print(f"         📈 [IC] IS IC: {ic_is:7.4f} | OOS IC: {ic_oos:7.4f} (Pooled across instruments)")
    
    if len(oos_preds) > 30:
        df_plot = pd.DataFrame({'Pred': oos_preds, 'Target': target_oos}).dropna()
        if not df_plot.empty and df_plot['Pred'].nunique() > 5:
            try:
                # =========================================================
                # 🌟 全新升级的分布图: 对齐回测引擎的物理仓位映射
                # =========================================================
                # 1A. 减去均值 (Demean) 版本
                raw_z = (df_plot['Pred'] - fold_mean) / (fold_std + 1e-8)
                df_plot['Mapped_Pos'] = np.clip(raw_z / ZSCORE_SCALE, -1.0, 1.0)
                
                # 1B. [新增] 不减均值 (No-Demean) 版本：直接除以标准差
                raw_z_nodemean = df_plot['Pred'] / (fold_std + 1e-8)
                df_plot['Mapped_Pos_NoDemean'] = np.clip(raw_z_nodemean / ZSCORE_SCALE, -1.0, 1.0)
                
                # 2. 依然用原始 Pred 划分为 10 组
                df_plot['Group'] = pd.qcut(df_plot['Pred'], q=10, labels=False, duplicates='drop')
                
                # 3. 聚合各组指标
                grouped = df_plot.groupby('Group')
                group_mean = grouped['Target'].mean() * 10000 
                
                group_pos_min = grouped['Mapped_Pos'].min()
                group_pos_max = grouped['Mapped_Pos'].max()
                
                group_pos_mean = grouped['Mapped_Pos'].mean()
                group_pos_mean_nodemean = grouped['Mapped_Pos_NoDemean'].mean() # [新增] 不减均值的平均仓位
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # 📊 [左轴] 画收益率柱状图
                bars = ax1.bar(group_mean.index + 1, group_mean.values, color='cornflowerblue', edgecolor='black', alpha=0.8, label='Return (bps)')
                for bar in bars:
                    yval = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (0.5 if yval > 0 else -0.5), f'{yval:.1f}', va='bottom' if yval > 0 else 'top', ha='center', fontsize=10)
                
                ax1.set_title(f"[PORTFOLIO] {q_name} OOS Pred Deciles vs Fwd Return (bps)", fontsize=14, fontweight='bold')
                ax1.set_ylabel("Mean Fwd Return (bps)", fontsize=12)
                ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
                
                x_labels = [f"G{int(i)+1}\n[{group_pos_min.loc[i]:.2f}, {group_pos_max.loc[i]:.2f}]" for i in group_mean.index]
                ax1.set_xticks(group_mean.index + 1)
                ax1.set_xticklabels(x_labels, fontsize=9)
                ax1.set_xlabel("OOS Decile Bins & Actual Position Range (Demeaned)", fontsize=12)
                
                # 📈 [右轴] 折线画真实的平均映射仓位
                ax2 = ax1.twinx()
                # 橘色实线：减去均值的映射
                ax2.plot(group_mean.index + 1, group_pos_mean.values, color='darkorange', marker='o', linewidth=2, linestyle='-', label='OOS Mean Pos (Demean)')
                # 棕色虚线：不减均值的映射
                ax2.plot(group_mean.index + 1, group_pos_mean_nodemean.values, color='saddlebrown', marker='s', linewidth=2, linestyle='--', label='OOS Mean Pos (Raw/No-Demean)')
                
                ax2.set_ylabel("Mapped Target Position", fontsize=12)
                ax2.set_ylim(-1.1, 1.1)
                
                # =========================================================
                # 🌟 IS 阈值定位
                # =========================================================
                x_q_upper = (df_plot['Pred'] <= fold_q_up_lock).mean() * 10.0 + 0.5
                x_q_lower = (df_plot['Pred'] <= fold_q_dn_lock).mean() * 10.0 + 0.5
                
                lbl_upper = f'IS Q{int(Q_UPPER_LOCK*100)}'
                if 0.5 <= x_q_upper <= 10.5:
                    ax1.axvline(x_q_upper, color='green', linestyle='-.', linewidth=2.5, label=f'{lbl_upper} Trigger')
                else:
                    edge_x = 10.5 if x_q_upper > 10.5 else 0.5
                    ax1.axvline(edge_x, color='green', linestyle='-.', linewidth=2.5, alpha=0.5, label=f'{lbl_upper} (Out of Range)')
                    
                lbl_lower = f'IS Q{int(Q_LOWER_LOCK*100)}'
                if 0.5 <= x_q_lower <= 10.5:
                    ax1.axvline(x_q_lower, color='purple', linestyle='-.', linewidth=2.5, label=f'{lbl_lower} Trigger')
                else:
                    edge_x = 0.5 if x_q_lower < 0.5 else 10.5
                    ax1.axvline(edge_x, color='purple', linestyle='-.', linewidth=2.5, alpha=0.5, label=f'{lbl_lower} (Out of Range)')
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                ax1.grid(axis='y', linestyle=':', alpha=0.7)
                plt.tight_layout()
                plt.show()
                
            except Exception as e: 
                print(f"         ⚠️ Failed to plot deciles: {e}")

    train_start = train_start + pd.DateOffset(months=DROP_MONTHS)


# =======================================================
# 6. 🚀 多品种精细化回测引擎
# =======================================================
print(f"\n🔮 开始 Portfolio 级别全局预热回测计算...")

def run_backtest(df_in, feature_name, eval_start, eval_end, windows=[2000]):
    cols = ['index_time', 'instrument', 'Ret_1m_Fwd', 'Ret_30m_Fwd', feature_name, 
            'is_eod_1m', 'contract_change', 'bi', 
            'Fold_IS_Mean', 'Fold_IS_Std', 
            'Fold_IS_Q_Up_Lock', 'Fold_IS_Q_Dn_Lock', 
            'Fold_IS_Q_Up_Rob', 'Fold_IS_Q_Dn_Rob', 'Fold_IS_Med']
    df_bt = df_in[cols].dropna(subset=[feature_name, 'Ret_1m_Fwd']).copy()
    if len(df_bt) < 100: return {}

    df_bt['Ret_1m_Fwd_Exp'] = np.exp(df_bt['Ret_1m_Fwd'].values.astype(np.float32)) - 1.0

    eval_mask = ((df_bt['index_time'] >= pd.to_datetime(eval_start)) & (df_bt['index_time'] <= pd.to_datetime(eval_end + ' 23:59:59'))).values
    
    results = {'PORTFOLIO': {}}
    for inst in TARGET_POOL: results[inst] = {}

    def calc(pos_arr, use_fee, prefix):
        df_bt['pos'] = pos_arr
        df_bt['pos_shift'] = df_bt.groupby('instrument')['pos'].shift(1).fillna(0.0)
        
        # 🌟 向量化判断开平仓量
        P1 = df_bt['pos_shift'].values
        P2 = df_bt['pos'].values
        same_dir = (P1 * P2 >= 0)
        abs_P1, abs_P2 = np.abs(P1), np.abs(P2)
        
        open_amt = np.where(same_dir, np.maximum(abs_P2 - abs_P1, 0.0), abs_P2)
        close_amt = np.where(same_dir, np.maximum(abs_P1 - abs_P2, 0.0), abs_P1)
        df_bt['to'] = open_amt + close_amt
        
        df_bt['open_fee_rate'] = df_bt['instrument'].map(open_fee_dict)
        df_bt['close_fee_rate'] = df_bt['instrument'].map(close_fee_dict)
        
        fee_cost = (open_amt * df_bt['open_fee_rate'] + close_amt * df_bt['close_fee_rate']) if use_fee else 0.0
        df_bt['pnl'] = df_bt['pos'] * df_bt['Ret_1m_Fwd_Exp'] - fee_cost
        eval_df = df_bt[eval_mask]
        if eval_df.empty: return
        
        for inst_name, df_group in eval_df.groupby('instrument', observed=True):
            ps = df_group.set_index('index_time')['pnl']
            nav_s = (1.0 + ps).cumprod()
            mpd = mpd_bars_dict[inst_name]
            bpy = 252 * mpd
            n_years = len(nav_s) / max(bpy, 1)
            ann_ret = (nav_s.iloc[-1] - 1.0) / max(n_years, 1e-8) if n_years < 0.1 else (nav_s.iloc[-1]) ** (1.0 / max(n_years, 1e-8)) - 1.0
            mdd = (1.0 - nav_s / nav_s.cummax()).max()
            calmar = ann_ret / mdd if mdd > 0 else 0.0
            
            results[inst_name][f'{prefix}_ret'] = round(float((nav_s.iloc[-1] - 1.0) * 10000), 1)
            results[inst_name][f'{prefix}_shp'] = round(float(ps.mean() / (ps.std() + 1e-8) * np.sqrt(bpy)), 3)
            results[inst_name][f'{prefix}_mdd'] = round(float(mdd), 4)
            results[inst_name][f'{prefix}_calmar'] = round(float(calmar), 2)
            results[inst_name][f'{prefix}_to'] = round(float(df_group['to'].mean() * mpd), 3)
            results[inst_name][f'{prefix}_nav'] = nav_s
            
        # PORTFOLIO 集成指标计算
        port_pnl = eval_df.groupby('index_time')['pnl'].mean()
        port_nav = (1.0 + port_pnl).cumprod()
        avg_bpy = np.mean(list(mpd_bars_dict.values())) * 252
        n_years = len(port_nav) / max(avg_bpy, 1)
        ann_ret = (port_nav.iloc[-1] - 1.0) / max(n_years, 1e-8) if n_years < 0.1 else (port_nav.iloc[-1]) ** (1.0 / max(n_years, 1e-8)) - 1.0
        mdd = (1.0 - port_nav / port_nav.cummax()).max()
        calmar = ann_ret / mdd if mdd > 0 else 0.0
        
        # 换手率: 先算每天每个品种平均，再算整个组合的平均换手率
        port_to = eval_df.groupby('index_time')['to'].mean().mean() * np.mean(list(mpd_bars_dict.values()))
        
        results['PORTFOLIO'][f'{prefix}_ret'] = round(float((port_nav.iloc[-1] - 1.0) * 10000), 1)
        results['PORTFOLIO'][f'{prefix}_shp'] = round(float(port_pnl.mean() / (port_pnl.std() + 1e-8) * np.sqrt(avg_bpy)), 3)
        results['PORTFOLIO'][f'{prefix}_mdd'] = round(float(mdd), 4)
        results['PORTFOLIO'][f'{prefix}_calmar'] = round(float(calmar), 2)
        results['PORTFOLIO'][f'{prefix}_to'] = round(float(port_to), 3)
        results['PORTFOLIO'][f'{prefix}_nav'] = port_nav

    for w in windows:
        # ==============================================================
        # 🌟 1. 计算全局跨品种的 Rolling 属性 (按时间排序)
        # ==============================================================
        pooled_w = w * len(TARGET_POOL)
        df_bt = df_bt.sort_values('index_time')
        f_series = df_bt[feature_name]

        df_bt['pooled_mean'] = f_series.rolling(pooled_w, min_periods=pooled_w//2).mean().values
        df_bt['pooled_std'] = f_series.rolling(pooled_w, min_periods=pooled_w//2).std().values
        df_bt['pooled_qhigh'] = f_series.rolling(pooled_w, min_periods=pooled_w//2).quantile(Q_UPPER_LOCK).values
        df_bt['pooled_qlow'] = f_series.rolling(pooled_w, min_periods=pooled_w//2).quantile(Q_LOWER_LOCK).values
        df_bt['pooled_q95'] = f_series.rolling(pooled_w, min_periods=pooled_w//2).quantile(0.95).values
        df_bt['pooled_q05'] = f_series.rolling(pooled_w, min_periods=pooled_w//2).quantile(0.05).values

        # ==============================================================
        # 🌟 2. 算完全局特征后，切回各自品种隔离状态，准备遍历
        # ==============================================================
        df_bt = df_bt.sort_values(['instrument', 'index_time'])
        
        f_arr = df_bt[feature_name].values.astype(np.float32)
        
        # 👇 [修复点] 统一增加 .fillna(0).astype(bool) 确保绝对的布尔类型
        eod1 = df_bt['is_eod_1m'].fillna(0).astype(bool).values
        eod30 = df_bt.groupby('instrument')['is_eod_1m'].transform(lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]).fillna(0).astype(bool).values
        roll1 = (df_bt.groupby('instrument')['contract_change'].shift(-1).fillna(False) | df_bt['contract_change']).fillna(0).astype(bool).values
        roll30 = df_bt.groupby('instrument')['contract_change'].transform(lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]).fillna(0).astype(bool).values
        reb30 = (df_bt['bi'] % 30) == TRADING_MINUTE
        
        t_next = df_bt.groupby('instrument')['index_time'].shift(-1)
        gap_hours = (t_next - df_bt['index_time']).dt.total_seconds() / 3600.0
        day_gap = (t_next.dt.normalize() - df_bt['index_time'].dt.normalize()).dt.days
        is_night = ((df_bt['index_time'].dt.time >= datetime.time(20, 0)) | (df_bt['index_time'].dt.time <= datetime.time(3, 0)))
        night_end = ((gap_hours > 8.0) & is_night & (day_gap >= 2)).fillna(False)
        
        # 👇 [修复点] 强制转布尔
        eod_night = df_bt.assign(ne=night_end).groupby('instrument')['ne'].transform(lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]).fillna(0).astype(bool).values

        inst_change = (df_bt['instrument'] != df_bt['instrument'].shift(1)).fillna(0).astype(bool).values
        
        # 现在所有输入都是严格的 bool 数组，位运算 | 畅通无阻
        force_flat_30m = eod1 | eod_night
        force_flat_strict = eod1 | eod_night | roll1 | inst_change
        
        # 🟢 方法 A: Pooled Rolling Z-Score
        global_std = np.nanstd(f_arr)
        min_std_threshold = global_std * 0.1
        s = np.where(df_bt['pooled_std'].values < min_std_threshold, min_std_threshold, df_bt['pooled_std'].values)
        # [A1. 原版减均值]
        pz = np.nan_to_num(np.clip((f_arr - df_bt['pooled_mean'].values) / (s + 1e-8) / ZSCORE_SCALE, -1.0, 1.0), nan=0.0).astype(np.float32)
        df_bt['raw_z'] = np.where(force_flat_30m | roll30, 0.0, np.where(reb30, pz, np.nan))
        p3 = df_bt.groupby('instrument')['raw_z'].ffill().fillna(0.0).values.astype(np.float32)
        calc(p3, True,  f'w{w}_z_30m_intra_fee')
        
        # [A2. 新增：不减均值]
        pz_nodemean = np.nan_to_num(np.clip(f_arr / (s + 1e-8) / ZSCORE_SCALE, -1.0, 1.0), nan=0.0).astype(np.float32)
        df_bt['raw_z_nodemean'] = np.where(force_flat_30m | roll30, 0.0, np.where(reb30, pz_nodemean, np.nan))
        p3_nodemean = df_bt.groupby('instrument')['raw_z_nodemean'].ffill().fillna(0.0).values.astype(np.float32)
        calc(p3_nodemean, True,  f'w{w}_z_nodemean_30m_intra_fee')
        
        # 🟡 方法 B: Pooled Rolling Quantile Lock 
        raw_signal = np.zeros_like(f_arr)
        raw_signal[f_arr > df_bt['pooled_qhigh'].values] = 1.0
        raw_signal[f_arr < df_bt['pooled_qlow'].values] = -1.0
        
        p_q_fixed = np.zeros(len(f_arr), dtype=np.float32)
        curr_p, bars_held = 0.0, 0
        for i in range(len(f_arr)):
            if force_flat_strict[i]:
                curr_p = 0.0; bars_held = 0; p_q_fixed[i] = 0.0; continue
            if bars_held > 0:
                p_q_fixed[i] = curr_p; bars_held += 1
                if bars_held >= 30: bars_held = 0; curr_p = 0.0
            else:
                if raw_signal[i] != 0:
                    curr_p = raw_signal[i]; p_q_fixed[i] = curr_p; bars_held = 1
                else:
                    p_q_fixed[i] = 0.0
        calc(p_q_fixed, True,  f'w{w}_q10_fix30_fee')

        # 🔵 方法 D: IS 全局统计量映射 
        is_mean_arr, is_std_arr = df_bt['Fold_IS_Mean'].values, df_bt['Fold_IS_Std'].values
        
        # [D1. 原版减均值]
        pz_is = np.nan_to_num(np.clip((f_arr - is_mean_arr) / (is_std_arr + 1e-8) / ZSCORE_SCALE, -1.0, 1.0), nan=0.0).astype(np.float32)
        df_bt['raw_is'] = np.where(force_flat_30m | roll30, 0.0, np.where(reb30, pz_is, np.nan))
        p3_is = df_bt.groupby('instrument')['raw_is'].ffill().fillna(0.0).values.astype(np.float32)
        calc(p3_is, True, f'w{w}_is_stat_30m_intra_fee')
        
        # [D2. 新增：不减均值]
        pz_is_nodemean = np.nan_to_num(np.clip(f_arr / (is_std_arr + 1e-8) / ZSCORE_SCALE, -1.0, 1.0), nan=0.0).astype(np.float32)
        df_bt['raw_is_nodemean'] = np.where(force_flat_30m | roll30, 0.0, np.where(reb30, pz_is_nodemean, np.nan))
        p3_is_nodemean = df_bt.groupby('instrument')['raw_is_nodemean'].ffill().fillna(0.0).values.astype(np.float32)
        calc(p3_is_nodemean, True, f'w{w}_is_stat_nodemean_30m_intra_fee')
        
        # ==============================================================
        # 🟠 方法 E: 基于超参数的 IS 静态分位数触发锁定 (Q_Fix30)
        # ==============================================================
        is_q_up_arr = df_bt['Fold_IS_Q_Up_Lock'].values
        is_q_dn_arr = df_bt['Fold_IS_Q_Dn_Lock'].values
        
        raw_signal_is_q = np.zeros_like(f_arr)
        raw_signal_is_q[f_arr > is_q_up_arr] = 1.0
        raw_signal_is_q[f_arr < is_q_dn_arr] = -1.0
        
        p_is_q_fixed = np.zeros(len(f_arr), dtype=np.float32)
        curr_p, bars_held = 0.0, 0
        for i in range(len(f_arr)):
            if force_flat_strict[i]:
                curr_p = 0.0; bars_held = 0; p_is_q_fixed[i] = 0.0; continue
            if bars_held > 0:
                p_is_q_fixed[i] = curr_p; bars_held += 1
                if bars_held >= 30: bars_held = 0; curr_p = 0.0
            else:
                if raw_signal_is_q[i] != 0:
                    curr_p = raw_signal_is_q[i]; p_is_q_fixed[i] = curr_p; bars_held = 1
                else:
                    p_is_q_fixed[i] = 0.0
        calc(p_is_q_fixed, True,  f'w{w}_is_q10_fix30_fee')

        # 🟣 方法 F: IS Robust 极值锚点映射
        is_q_up_rob_arr = df_bt['Fold_IS_Q_Up_Rob'].values
        is_q_dn_rob_arr = df_bt['Fold_IS_Q_Dn_Rob'].values
        is_med_arr = df_bt['Fold_IS_Med'].values
        
        half_spread = (is_q_up_rob_arr - is_q_dn_rob_arr) / 2.0
        half_spread = np.where(half_spread < 1e-6, 1e-6, half_spread)
        
        # [F1. 原版减中位数]
        pz_is_rob = np.nan_to_num(np.clip((f_arr - is_med_arr) / half_spread, -1.0, 1.0), nan=0.0).astype(np.float32)
        df_bt['raw_is_rob'] = np.where(force_flat_30m | roll30, 0.0, np.where(reb30, pz_is_rob, np.nan))
        p3_is_rob = df_bt.groupby('instrument')['raw_is_rob'].ffill().fillna(0.0).values.astype(np.float32)
        calc(p3_is_rob, True, f'w{w}_is_rob_30m_intra_fee')

        # [F2. 新增：不减中位数 (保留绝对漂移)]
        pz_is_rob_nodemean = np.nan_to_num(np.clip(f_arr / half_spread, -1.0, 1.0), nan=0.0).astype(np.float32)
        df_bt['raw_is_rob_nodemean'] = np.where(force_flat_30m | roll30, 0.0, np.where(reb30, pz_is_rob_nodemean, np.nan))
        p3_is_rob_nodemean = df_bt.groupby('instrument')['raw_is_rob_nodemean'].ffill().fillna(0.0).values.astype(np.float32)
        calc(p3_is_rob_nodemean, True, f'w{w}_is_rob_nodemean_30m_intra_fee')

    del df_bt; gc.collect()
    return results

# =======================================================
# 7. 打印报表与画图工具
# =======================================================
def print_report(title, start_dt, end_dt):
    results = run_backtest(df, 'Pred', eval_start=start_dt, eval_end=end_dt)
    print("\n" + "=" * 110)
    print(f"📊 {title} | Range: {start_dt} → {end_dt}")
    print("=" * 110)
    
    modes = [
        'w2000_z_30m_intra_fee',  
        'w2000_z_nodemean_30m_intra_fee',  
        'w2000_is_stat_30m_intra_fee',  
        'w2000_is_stat_nodemean_30m_intra_fee',  
        'w2000_is_rob_30m_intra_fee',
        'w2000_is_rob_nodemean_30m_intra_fee',
        'w2000_is_q10_fix30_fee'
    ]
    
    # === 1. 打印全局 Portfolio 回测 ===
    r_port = results.get('PORTFOLIO', {})
    if r_port:
        print(f"\n{'─' * 85}\n🌟 PORTFOLIO (Mixed Pool)")
        print(f"{'Mode':<30} {'Ret(bps)':>10} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'TO':>8}\n{'─' * 85}")
        for mode in modes:
            if f'{mode}_ret' in r_port:
                rs, ss, ms, cs, ts = f"{r_port[f'{mode}_ret']:.1f}", f"{r_port[f'{mode}_shp']:.3f}", f"{r_port[f'{mode}_mdd']*100:.2f}%", f"{r_port[f'{mode}_calmar']:.2f}", f"{r_port[f'{mode}_to']:.3f}"
                highlight = "🟠 " if "is_q10" in mode else ("🔵 " if "is_stat" in mode else "   ")
                print(f"{highlight}{mode:<27} {rs:>10} {ss:>8} {ms:>8} {cs:>8} {ts:>8}")
    
    # === 2. 简要打印各个单品种的最佳表现，检查稳定性 ===
    best_mode = 'w2000_is_stat_30m_intra_fee'
    print(f"\n{'─' * 85}\n🔹 INDIVIDUAL INSTRUMENTS BREAKDOWN (Using '{best_mode}')")
    print(f"{'Instrument':<10} {'Ret(bps)':>10} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'TO':>8}\n{'─' * 85}")
    for inst in TARGET_POOL:
        r = results.get(inst, {})
        if r and f'{best_mode}_ret' in r:
            rs, ss, ms, cs, ts = f"{r[f'{best_mode}_ret']:.1f}", f"{r[f'{best_mode}_shp']:.3f}", f"{r[f'{best_mode}_mdd']*100:.2f}%", f"{r[f'{best_mode}_calmar']:.2f}", f"{r[f'{best_mode}_to']:.3f}"
            print(f"{inst:<10} {rs:>10} {ss:>8} {ms:>8} {cs:>8} {ts:>8}")
            
    return results

def plot_equity_curve(results, inst, mode, title="Equity Curve"):
    if inst not in results or f'{mode}_nav' not in results[inst]: return
    
    nav = results[inst][f'{mode}_nav']
    
    # 🌟 降频为日线 (取每天最后时刻的累计净值)
    daily_nav = nav.resample('D').last().dropna()
    
    # 🌟 [修复] 强制补齐起点 1.0
    # 获取第一天交易的前一天作为起始锚点
    start_date = daily_nav.index[0] - pd.Timedelta(days=1)
    
    # 将初始本金 1.0 插入到序列的最前方
    daily_nav.loc[start_date] = 1.0
    daily_nav = daily_nav.sort_index()
    
    # 防止由于切片引起的基准偏移，强制将全序列除以第一个点归一化到 1.0
    daily_nav = daily_nav / daily_nav.iloc[0]
    
    # 重新计算回撤
    dd = 1.0 - daily_nav / daily_nav.cummax()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 🌟 [新增 marker='o'] 在折线上每天的位置画一个小圆点
    ax1.plot(daily_nav.index, daily_nav.values, label=f'NAV ({mode})', 
             color='royalblue', linewidth=1.5, marker='o', markersize=4, markerfacecolor='white')
             
    ax1.set_title(title, fontsize=15, fontweight='bold')
    ax1.set_ylabel('Net Asset Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper left")
    
    final_ret = (daily_nav.iloc[-1] - 1.0) * 10000
    ax1.text(daily_nav.index[-1], daily_nav.values[-1], f" +{final_ret:.0f} bps", 
             color='green' if final_ret > 0 else 'red', fontsize=12, va='center')
    
    ax2.fill_between(dd.index, -dd.values * 100, 0, color='crimson', alpha=0.3)
    ax2.plot(dd.index, -dd.values * 100, color='crimson', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# =======================================================
# 8. 执行与输出
# =======================================================
df['Pred'] = df['Pred_IS']
is_results = print_report("In-Sample (IS)", "2024-01-08", "2024-12-31")

df['Pred'] = df['Pred_OOS'].fillna(df['Pred_IS'])
_h2_periods = [
    ('2025_Q1', '2025-01-01', '2025-03-31'), ('2025_Q2', '2025-04-01', '2025-06-30'),
    ('2025_Q3', '2025-07-01', '2025-09-30'), ('2025_Q4', '2025-10-01', '2025-12-31')
]
for (q_name, start_dt, end_dt) in _h2_periods:
    print_report(f"Out-Of-Sample (OOS) {q_name}", start_dt, end_dt)

oos_results = print_report("Out-Of-Sample (OOS) Full Year 2025", "2025-01-01", "2025-12-31")

modes = [
        'w2000_z_30m_intra_fee',  
        'w2000_z_nodemean_30m_intra_fee',  
        'w2000_is_stat_30m_intra_fee',  
        'w2000_is_stat_nodemean_30m_intra_fee',  
        'w2000_is_rob_30m_intra_fee',
        'w2000_is_rob_nodemean_30m_intra_fee',
        'w2000_is_q10_fix30_fee'
    ]

for mode in modes:
    plot_equity_curve(is_results, 'PORTFOLIO', mode, title=f"[PORTFOLIO] IS Curve (2024)")
    plot_equity_curve(oos_results, 'PORTFOLIO', mode, title=f"[PORTFOLIO] OOS Curve (2025)")


print("\n✅ 回测流程全部执行完毕！")
