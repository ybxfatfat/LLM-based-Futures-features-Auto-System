#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import json
import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from itertools import product
import matplotlib.pyplot as plt

# 使用 Python 标准库在本机并行搜索
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# =======================================================
# 1. 🎛️ 全局开关与配置 (Pooled Multi-Instrument)
# =======================================================
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = project_root / "data"

PARQUET_2024 = str(DATA_DIR / "factor_dataset_AllMarket_main_only_2024.parquet")
MARKET_STATES_PATH = str(DATA_DIR / "best_market_states_list.json")

# 通用商品宇宙
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
TARGET_POOL = list(set(COMM_RAW) - {'LC', 'EC', 'AO', 'BB', 'FB', 'WR', 'RS', 'BC', 'LU'})
TARGET = "Ret_30m_Fwd"

# 💡 超参搜索时，严格仅使用 2024 年数据，保留 2025 年为最终 OOS 盲测集

# =========================================================
# ⏱️ 2. 全市场高精度时间清洗函数
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
# 3. 动态加载混池共用特征
# =======================================================
pool_name = "Pooled_" + "_".join(TARGET_POOL)
if len(pool_name) > 50: pool_name = "Pooled_Custom_Set"
# 确保这里转为字符串
SELECTED_FEATURES_PATH = str(DATA_DIR / f"pooled_inst_features/{pool_name}_best_features.json")

print(f"⏳ 正在加载混池通用特征: {SELECTED_FEATURES_PATH} ...")
try:
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        feature_data = json.load(f)
    pool_features = feature_data.get("Feature", [])
except FileNotFoundError:
    raise FileNotFoundError("找不到池化特征 JSON，请确认筛选步骤已完成。")

selected_features = list(dict.fromkeys(pool_features))
print(f"🎯 选定训练池: {TARGET_POOL}")
print(f"✅ 加载了 {len(selected_features)} 个混池专属特征。")

# =======================================================
# 4. 内存友好型数据加载
# =======================================================
all_parquet_cols = pq.read_schema(PARQUET_2024).names
cols_to_load = ['index_time', 'instrument', 'contract', TARGET] + selected_features
valid_cols = [c for c in cols_to_load if c in all_parquet_cols]
features = [f for f in selected_features if f in valid_cols]

print(f"\n📦 正在读取 2024 年 {len(TARGET_POOL)} 个品种的底层数据...")
df = pq.read_table(PARQUET_2024, columns=valid_cols, filters=[('instrument', 'in', TARGET_POOL)]).to_pandas()
df['index_time'] = pd.to_datetime(df['index_time'])
df = df.dropna(subset=[TARGET] + features).reset_index(drop=True)
print(f"📦 原始加载数据行数: {len(df):,}")

# =======================================================
# 5. 深度清洗：剔除所有危险尾盘与节假日边界
# =======================================================
print("⚙️ 正在执行基础交易时段过滤...")
df = filter_global_trading_hours(df)

print("⚙️ 正在应用 30-min 逆向屏蔽算法 (剔除尾盘/长假/换月)...")
df = df.sort_values(['instrument', 'index_time']).reset_index(drop=True)

bt = df['index_time'].dt.time
df['is_eod_1m'] = (bt >= datetime.time(14, 59, 0)) & (bt <= datetime.time(15, 0, 0))
eod30 = df.groupby('instrument')['is_eod_1m'].transform(
    lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]
).fillna(0).astype(bool)

df['contract_tmp'] = df['contract'].fillna('Unknown')
df['contract_change'] = (df['instrument'] == df['instrument'].shift(-1)) & \
                             (df['contract_tmp'] != df['contract_tmp'].shift(-1))
roll30 = df.groupby('instrument')['contract_change'].transform(
    lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]
).fillna(0).astype(bool)

t_next = df.groupby('instrument')['index_time'].shift(-1)
gap_hours = (t_next - df['index_time']).dt.total_seconds() / 3600.0
day_gap = (t_next.dt.normalize() - df['index_time'].dt.normalize()).dt.days
is_night = ((bt >= datetime.time(20, 0)) | (bt <= datetime.time(3, 0)))
night_end = ((gap_hours > 8.0) & is_night & (day_gap >= 2)).fillna(False)
eod_night30 = df.assign(ne=night_end).groupby('instrument')['ne'].transform(
    lambda x: x.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1]
).fillna(0).astype(bool)

# 🚀 极其关键：只保留通过了所有过滤的【纯净有效交易样本】
df['is_traded'] = ~(eod30 | roll30 | eod_night30)
df = df[df['is_traded']].reset_index(drop=True)

print(f"✅ 清洗完成！最终参与超参搜索的【绝对安全样本】行数: {len(df):,}")

# =======================================================
# 6. 时序切分 (全局 80% 训练 / 20% 验证)
# =======================================================
df = df.sort_values('index_time').reset_index(drop=True)
unique_dates = np.sort(df['index_time'].dt.date.unique())
split_date = unique_dates[int(len(unique_dates) * 0.8)]
print(f"⏱️ 2024 样本内全局切分日期: {split_date}")

train_mask = df['index_time'].dt.date < split_date
val_mask = df['index_time'].dt.date >= split_date

# 转换为 np.float32 降低本地进程间序列化负担
X_train = df.loc[train_mask, features].values.astype(np.float32)
y_train = df.loc[train_mask, TARGET].values.astype(np.float32)
X_test = df.loc[val_mask, features].values.astype(np.float32)
y_test = df.loc[val_mask, TARGET].values.astype(np.float32)

val_df = df.loc[val_mask, ['index_time', 'instrument', TARGET]].copy()

del df, unique_dates
if 'table' in locals():
    del table
gc.collect()

# =======================================================
# 7. 定义超参数网格 (针对大数据量进行了加深扩展)
# =======================================================
param_grid = {
    'learning_rate':   [0.01, 0.02, 0.05],
    'max_depth':       [8, 10, 12],          
    'num_leaves':      [31, 63, 127],           
    'min_data_in_leaf': [500, 1000, 2000],  # 针对 ~40万训练样本的配置
    'feature_fraction': [0.6, 0.8],
    'bagging_fraction': [0.7, 0.8],
    'lambda_l1':       [0.1, 1.0],         # 💥 新增：L1正则化，高维特征降噪必备
    'lambda_l2':       [0.1, 1.0],         # 💥 新增：L2正则化，抗共线性
}

param_names = list(param_grid.keys())
param_list = [dict(zip(param_names, combo)) for combo in product(*param_grid.values())]

print(f"\n🔍 共 {len(param_list)} 组超参数组合待搜索，准备提交至本地进程池...")
print("=" * 80)

# =======================================================
# 8. 定义本地 Worker 并行训练函数
# =======================================================
def evaluate_params(params, x_train, y_train, x_test, y_test, v_df, target_col):
    """本地子进程中的独立训练函数。"""
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    
    FIXED_PARAMS = {
        'objective':    'regression',
        'metric':       'rmse',
        'n_estimators': 3000,
        'bagging_freq': 5,
        'n_jobs':       2, 
        'random_state': 42,
        'verbose':      -1,
    }
    lgb_params = {**FIXED_PARAMS, **params}
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    test_df_local = v_df.copy()
    test_df_local['Pred'] = model.predict(x_test)
    
    # 全局 Pooled Full IC
    full_ic = spearmanr(test_df_local['Pred'], test_df_local[target_col])[0]
    
    # 横截面 Daily IC (每天针对池子里的多品种进行 Rank 比较)
    # 💥 切分为 2000 分钟窗口
    time_factor, _ = pd.factorize(test_df_local['index_time'], sort=True)
    test_df_local['window_id'] = time_factor // 2000
    
    period_ic = test_df_local.groupby('window_id').apply(
        lambda x: spearmanr(x['Pred'], x[target_col])[0] if len(x) > 20 and x['Pred'].std() > 1e-8 else np.nan
    ).dropna()
    
    mean_ic = period_ic.mean()
    # 💥 年化乘数改为 30.24 (252天 * 240分 / 2000分)
    ir = mean_ic / period_ic.std() * np.sqrt(30.24) if period_ic.std() > 0 else np.nan
    n_trees = model.best_iteration_ if model.best_iteration_ else model.n_estimators_
    
    return {
        **params,
        'n_trees':       n_trees,
        'full_ic':       full_ic,
        'mean_period_ic': mean_ic,
        'ic_ir':         ir,
        'importances':   model.feature_importances_.tolist()
    }

worker_func = partial(evaluate_params, 
                      x_train=X_train, y_train=y_train, 
                      x_test=X_test, y_test=y_test, 
                      v_df=val_df, target_col=TARGET)

# =======================================================
# 9. 启动本地进程池并映射任务
# =======================================================
print("🚀 正在提交本地并行计算任务...")
import multiprocessing
max_workers = max(1, multiprocessing.cpu_count() - 2)
print(f"🔥 启动本地多进程搜索 (Workers: {max_workers})...")
with ProcessPoolExecutor(max_workers=max_workers) as pool:
    results = list(pool.map(worker_func, param_list))
print("✅ 本地并行计算完成！")

# =======================================================
# 10. 输出结果汇总与最优结果提取
# =======================================================
display_results = [{k: v for k, v in r.items() if k != 'importances'} for r in results]
# 针对混池，由于横截面因子更多依靠全局 Full IC 决定纯粹的绝对预测力，优先按 Full IC 排序
results_df = pd.DataFrame(display_results).sort_values('full_ic', ascending=False)

print("\n" + "=" * 100)
print(f"🏆 [TARGET POOL] 2024样本内 - 本地搜索结果汇总（按 Pooled Full IC 降序）")
print("=" * 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.5f}')
print(results_df.head(100).to_string(index=False))
print("=" * 100)

best_result_full = max(results, key=lambda x: x['full_ic'])
best_ic = best_result_full['full_ic']
best_params = {k: best_result_full[k] for k in param_names}
best_importances = best_result_full['importances']

print(f"\n🥇 [TARGET POOL] 2024 最优超参数 (In-Sample Full IC = {best_ic:.5f}):")
for k, v in best_params.items():
    print(f"    '{k}': {v},")
print(f"    'n_estimators': 3000,")
print("    -----------------------------------------")

# =======================================================
# 11. 可视化
# =======================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sorted_df = results_df.sort_values('full_ic', ascending=True).reset_index(drop=True)
colors = ['steelblue'] * len(sorted_df)
best_idx = sorted_df['full_ic'].idxmax()
if not pd.isna(best_idx):
    colors[best_idx] = 'darkorange'

labels = [f"lr={r['learning_rate']} d={r['max_depth']} l={r['num_leaves']}\n"
          f"ff={r['feature_fraction']} bf={r['bagging_fraction']}"
          for _, r in sorted_df.iterrows()]

# 对于混池子，Full IC 比 Daily IC IR 更稳定（日内常常多品种同时停盘）
ax1 = axes[0]
ax1.barh(range(len(sorted_df)), sorted_df['full_ic'].values, color=colors)
ax1.set_yticks(range(len(sorted_df)))
ax1.set_yticklabels(labels, fontsize=7)
ax1.set_xlabel("Pooled Full IC (Validation Set)", fontsize=12)
ax1.set_title(f"Hyperparameter Search: Full IC by Config (Pooled IS)", fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.5)
ax1.grid(axis='x', linestyle='--', alpha=0.4)

ax2 = axes[1]
feat_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': best_importances
}).sort_values(by='Importance', ascending=False).head(20)

ax2.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1], color='steelblue')
ax2.set_title(f"Top 20 Feature Importances (Best Pooled Model)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Importance (Split Count)", fontsize=12)
ax2.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
pool.shutdown()
