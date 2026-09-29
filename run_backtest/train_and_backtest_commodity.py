"""
文件: future_sim/run/train_and_backtest_commodity.py
=====================================================
1. 加载 parquet 因子数据 → 洗盘(过滤非交易时段) → 训练 LightGBM → 生成样本外 (OOS) 信号
2. NautilusTrader 正式回测：支持 Fixed Grid (最高水位线追加) 与 Dynamic Event (动态沙漏)

池子: 全市场商品期货 (精准匹配 2025 年固定/比例双轨手续费)
使用:
    python train_and_backtest_commodity.py                          # 全量
    python train_and_backtest_commodity.py --train-only             # 只训练
    python train_and_backtest_commodity.py --backtest-only          # 只回测
"""

import os
import re
import glob
import gc
import json
import pickle
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from decimal import Decimal

# ═══════════════════════════════════════════════════════════════
# 0. 配置常量
# ═══════════════════════════════════════════════════════════════

LGB_PARAMS = dict(
    n_estimators=5000,
    objective="regression",
    metric="rmse",
    learning_rate=0.01,
    max_depth=12,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.7,
    bagging_freq=5,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

CN_TZ = timezone(timedelta(hours=8))

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = str(project_root / "data")
PARQUET_PATH = f"{DATA_DIR}/factor_dataset_AllMarket_main_only_2025.parquet"

TARGET = "Ret_30m_Fwd"

# ---- 策略参数 ----
ZSCORE_LOOKBACK = 2000       
ZSCORE_CLIP = 3.0            
MAX_WEIGHT_PER_INST = 1.0    
TOTAL_CAPITAL = 10_000_000   
REBALANCE_INTERVAL_MIN = 30  
GTD_STAGE1_SEC = 45          
GTD_STAGE2_SEC = 45          
MIN_BEST_QTY = 50            
ADDON_WINDOW_MIN = 5         
MAPPING_TYPE = "zscore"      

# ═══════════════════════════════════════════════════════════════
# 1. 交易池与品种规格配置 (纯净版标准前缀)
# ═══════════════════════════════════════════════════════════════

POOL_NAME = "ferrous"

POOL_CONFIG = {
    "precious":     {"pool": ['GL', 'SL']},  # 标准代号替代 GLZ, SLZ
    "non_ferrous":  {"pool": ['AL', 'CU', 'ZN', 'PB', 'NI', 'SN', 'SS', 'BC', 'LC']},
    "ferrous":      {"pool": ['RB', 'HC', 'I', 'J', 'JM', 'SM', 'SF']},
    "chemicals":    {"pool": ['RU', 'BU', 'TA', 'MA', 'EG', 'PP', 'L', 'V', 'EB', 'UR', 'SA', 'FG', 'BR', 'FU', 'LU', 'NR', 'SC', 'EC']}
}
COMM_POOL = POOL_CONFIG[POOL_NAME]["pool"]

FEATURES_PATH = f"{DATA_DIR}/best_features_{POOL_NAME}.json"
SIGNALS_OUT = f"{DATA_DIR}/oos_signals_{POOL_NAME}.parquet"
MODEL_OUT = f"{DATA_DIR}/lgb_model_{POOL_NAME}.pkl"

NIGHT_CLOSE_MINS = {
    'AG': 150, 'AU': 150, 'SC': 150, 
    'CU': 60, 'AL': 60, 'ZN': 60, 'PB': 60, 'NI': 60, 'SN': 60, 'SS': 60, 'BC': 60,
}

# 规格表: (乘数, tick_size, 交易所, 保证金率, 计费模式, 费率数值)
# ⚠️ 这里区分了 "ratio" (万分比) 和 "fixed" (固定绝对金额，单位：元/手)
INSTRUMENT_SPECS = {
    # SHFE
    'AG': (15, 1.0, 'SHFE', 0.15, 'ratio', 0.00005),   # 万0.5
    'AL': (5, 5.0, 'SHFE', 0.15, 'fixed', 3.0),        # 3元/手
    'AO': (20, 1.0, 'SHFE', 0.15, 'ratio', 0.0001),    # 万1
    'AU': (1000, 0.02, 'SHFE', 0.15, 'fixed', 10.0),   # 10元/手
    'BR': (5, 5.0, 'SHFE', 0.15, 'ratio', 0.0001),     # 万1
    'BU': (10, 1.0, 'SHFE', 0.15, 'ratio', 0.0001),    # 万1
    'CU': (5, 10.0, 'SHFE', 0.15, 'ratio', 0.00005),   # 万0.5
    'FU': (10, 1.0, 'SHFE', 0.15, 'ratio', 0.00005),   # 万0.5
    'HC': (10, 1.0, 'SHFE', 0.15, 'ratio', 0.0001),    # 万1
    'NI': (1, 10.0, 'SHFE', 0.15, 'fixed', 3.0),       # 3元/手
    'PB': (5, 5.0, 'SHFE', 0.15, 'ratio', 0.00004),    # 万0.4
    'RB': (10, 1.0, 'SHFE', 0.15, 'ratio', 0.0001),    # 万1
    'RU': (10, 5.0, 'SHFE', 0.15, 'fixed', 3.0),       # 3元/手
    'SN': (1, 10.0, 'SHFE', 0.15, 'fixed', 3.0),       # 3元/手
    'SP': (10, 2.0, 'SHFE', 0.15, 'ratio', 0.00005),   # 万0.5
    'SS': (5, 5.0, 'SHFE', 0.15, 'fixed', 2.0),        # 2元/手
    'WR': (10, 1.0, 'SHFE', 0.15, 'ratio', 0.00004),   # 万0.4
    'ZN': (5, 5.0, 'SHFE', 0.15, 'fixed', 3.0),        # 3元/手
    # DCE
    'A': (10, 1.0, 'DCE', 0.15, 'fixed', 2.0),         # 2元/手
    'B': (10, 1.0, 'DCE', 0.15, 'fixed', 1.0),         # 1元/手
    'BB': (500, 0.05, 'DCE', 0.15, 'ratio', 0.0001),   # 万1
    'C': (10, 1.0, 'DCE', 0.15, 'fixed', 1.2),         # 1.2元/手
    'CS': (10, 1.0, 'DCE', 0.15, 'fixed', 1.5),        # 1.5元/手
    'EB': (5, 1.0, 'DCE', 0.15, 'fixed', 3.0),         # 3元/手
    'EG': (10, 1.0, 'DCE', 0.15, 'fixed', 3.0),        # 3元/手
    'FB': (10, 0.05, 'DCE', 0.15, 'ratio', 0.0001),    # 万1
    'I': (100, 0.5, 'DCE', 0.15, 'ratio', 0.0001),     # 万1
    'J': (100, 0.5, 'DCE', 0.15, 'ratio', 0.0001),     # 万1
    'JD': (10, 1.0, 'DCE', 0.15, 'ratio', 0.00015),    # 万1.5
    'JM': (60, 0.5, 'DCE', 0.15, 'ratio', 0.0001),     # 万1
    'L': (5, 1.0, 'DCE', 0.15, 'fixed', 1.0),          # 1元/手
    'LH': (16, 5.0, 'DCE', 0.15, 'ratio', 0.0001),     # 万1
    'M': (10, 1.0, 'DCE', 0.15, 'fixed', 1.5),         # 1.5元/手
    'P': (10, 2.0, 'DCE', 0.15, 'fixed', 2.5),         # 2.5元/手
    'PG': (20, 1.0, 'DCE', 0.15, 'fixed', 6.0),        # 6元/手
    'PP': (5, 1.0, 'DCE', 0.15, 'fixed', 1.0),         # 1元/手
    'RR': (10, 1.0, 'DCE', 0.15, 'fixed', 1.0),        # 1元/手
    'V': (5, 1.0, 'DCE', 0.15, 'fixed', 1.0),          # 1元/手
    'Y': (10, 2.0, 'DCE', 0.15, 'fixed', 2.5),         # 2.5元/手
    # CZCE
    'AP': (10, 1.0, 'CZCE', 0.15, 'fixed', 5.0),       # 5元/手
    'CF': (5, 5.0, 'CZCE', 0.15, 'fixed', 4.3),        # 4.3元/手
    'CJ': (5, 5.0, 'CZCE', 0.15, 'fixed', 3.0),        # 3元/手
    'FG': (20, 1.0, 'CZCE', 0.15, 'fixed', 3.0),       # 3元/手
    'MA': (10, 1.0, 'CZCE', 0.15, 'fixed', 1.0),       # 1元/手
    'OI': (10, 1.0, 'CZCE', 0.15, 'fixed', 2.0),       # 2元/手
    'PF': (5, 2.0, 'CZCE', 0.15, 'fixed', 3.0),        # 3元/手
    'PK': (5, 2.0, 'CZCE', 0.15, 'fixed', 4.0),        # 4元/手
    'RM': (10, 1.0, 'CZCE', 0.15, 'fixed', 1.5),       # 1.5元/手
    'RS': (10, 1.0, 'CZCE', 0.15, 'fixed', 2.0),       # 2元/手
    'SA': (20, 1.0, 'CZCE', 0.15, 'ratio', 0.0001),    # 万1
    'SF': (5, 2.0, 'CZCE', 0.15, 'fixed', 3.0),        # 3元/手
    'SM': (5, 2.0, 'CZCE', 0.15, 'fixed', 3.0),        # 3元/手
    'SR': (10, 1.0, 'CZCE', 0.15, 'fixed', 3.0),       # 3元/手
    'TA': (5, 2.0, 'CZCE', 0.15, 'fixed', 3.0),        # 3元/手
    'UR': (20, 1.0, 'CZCE', 0.15, 'ratio', 0.0001),    # 万1
    # INE & GFEX
    'BC': (5, 10.0, 'INE', 0.15, 'ratio', 0.00001),    # 万0.1
    'EC': (50, 0.1, 'INE', 0.15, 'ratio', 0.0005),     # 万5
    'LU': (10, 1.0, 'INE', 0.15, 'ratio', 0.00001),    # 万0.1
    'NR': (10, 5.0, 'INE', 0.15, 'fixed', 3.0),        # 3元/手
    'SC': (1000, 0.1, 'INE', 0.15, 'fixed', 20.0),     # 20元/手
    'LC': (1, 50.0, 'GFEX', 0.15, 'ratio', 0.00008),   # 万0.8
}


# ═══════════════════════════════════════════════════════════════
# 2. 工具函数与时段洗盘
# ═══════════════════════════════════════════════════════════════

def parquet_contract_to_futures(prefix: str, contract: str) -> str:
    """parquet 合约 'YYYYMM' → 期货代码 'PREFIXYYM'"""
    yy = contract[2:4]
    mm = contract[4:6]
    return f"{prefix}{yy}{mm}"

def filter_global_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    print(f"执行全市场交易时段清洗前: {len(df)} 行")
    time_str = df['index_time'].dt.strftime('%H:%M:%S')
    
    is_cffex = df['instrument'].isin(['IC', 'IF', 'IH', 'IM'])
    mask_cffex = (
        ((time_str >= '09:30:00') & (time_str <= '11:30:00')) |
        ((time_str >= '13:00:00') & (time_str <= '15:00:00'))
    )
    
    is_commodity = ~is_cffex
    mask_comm_day = (
        ((time_str >= '09:00:00') & (time_str <= '10:15:00')) |
        ((time_str >= '10:30:00') & (time_str <= '11:30:00')) |
        ((time_str >= '13:30:00') & (time_str <= '15:00:00'))
    )
    
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
    
    valid_mask = (is_cffex & mask_cffex) | (is_commodity & mask_comm)

    df_clean = df[valid_mask].copy()
    df_clean = df_clean.sort_values(['index_time', 'instrument', 'contract']).reset_index(drop=True)
    
    print(f"✅ 执行全市场清洗后: {len(df_clean)} 行")
    return df_clean

def load_features() -> list[str]:
    with open(FEATURES_PATH, "r") as f:
        data = json.load(f)
    raw = data["Feature"] if isinstance(data, dict) and "Feature" in data else data
    all_cols = pq.read_schema(PARQUET_PATH).names
    features = [f for f in raw if f in all_cols]
    print(f"✅ {len(features)}/{len(raw)} 特征可用")
    return features


# ═══════════════════════════════════════════════════════════════
# 3. 训练 & 信号生成
# ═══════════════════════════════════════════════════════════════

def train_and_generate_signals() -> pd.DataFrame:
    features = load_features()

    all_cols = ["index_time", "instrument", "contract", TARGET] + features
    table = pq.read_table(
        PARQUET_PATH, columns=[c for c in all_cols if c in pq.read_schema(PARQUET_PATH).names],
        filters=[("instrument", "in", COMM_POOL)],
    )
    df = table.to_pandas()
    df["index_time"] = pd.to_datetime(df["index_time"]).dt.tz_localize(CN_TZ)
    df = df.dropna(subset=[TARGET]).sort_values("index_time").reset_index(drop=True)

    df = filter_global_trading_hours(df)

    print("\n⚙️ 正在计算目标波动率权重 (Sample Weights)...")
    target_vol = df.groupby('instrument')[TARGET].transform(
        lambda x: x.shift(30).rolling(2000, min_periods=100).std()
    )
    df['Weight'] = 1.0 / (target_vol + 1e-8)
    df = df.dropna(subset=['Weight']).reset_index(drop=True)

    print(f"📦 加载: {len(df):,} 行, {len(features)} 特征, {df['instrument'].nunique()} 品种")

    split_idx = int(len(df) * 0.8)
    oos_start_time = df.loc[split_idx, 'index_time']
    print(f"⏱️ 切分点: {oos_start_time}（{split_idx:,}/{len(df):,}）")

    # 💥 核心纠错：加入 30 个 bar 的 Embargo (隔离带) 防泄露
    embargo_idx = split_idx - 30 
    
    X_train, y_train = df.iloc[:embargo_idx][features], df.iloc[:embargo_idx][TARGET]
    w_train = df.iloc[:embargo_idx]['Weight']
    X_test,  y_test  = df.iloc[split_idx:][features],  df.iloc[split_idx:][TARGET]

    print("🧠 训练 带权重的 LightGBM ...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,  
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )
    n_trees = model.best_iteration_ or model.n_estimators_

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 模型: {MODEL_OUT}")

    df["Pred"] = model.predict(df[features])

    df_train = df.iloc[:split_idx]
    inst_directions = {}
    for inst in df_train['instrument'].unique():
        sub = df_train[df_train['instrument'] == inst].dropna(subset=['Pred', TARGET])
        inst_directions[inst] = 1.0 if (len(sub) >= 100 and sub['Pred'].corr(sub[TARGET]) >= 0) else -1.0
    print(f"📌 方向(基于训练集): {inst_directions}")

    oos = df.iloc[split_idx:].copy()
    full_ic, _ = spearmanr(oos["Pred"], oos[TARGET])
    oos["Date"] = oos["index_time"].dt.date
    daily_ic = oos.groupby("Date").apply(
        lambda x: spearmanr(x["Pred"], x[TARGET])[0] if len(x) > 20 else np.nan
    ).dropna()
    mean_ic = daily_ic.mean()
    ir = mean_ic / daily_ic.std() * np.sqrt(252) if daily_ic.std() > 0 else 0
    print(f"📊 OOS: Full IC={full_ic:.5f}, Mean Daily IC={mean_ic:.5f}, IR={ir:.3f}, Trees={n_trees}")

    print(f"🔮 生成 OOS 信号 (全量 warm-up, lookback={ZSCORE_LOOKBACK}) ...")
    parts = []
    for inst in COMM_POOL:
        prefix = inst
        all_data = df[df["instrument"] == inst].copy()
        
        r = all_data["Pred"].rolling(ZSCORE_LOOKBACK, min_periods=ZSCORE_LOOKBACK // 2)
        m = r.mean()
        s = r.std()
        q75, q25 = r.quantile(0.75), r.quantile(0.25)
        q95, q05 = r.quantile(0.95), r.quantile(0.05)

        all_data["zscore"]   = (all_data["Pred"] / (s + 1e-8)).fillna(0)
        all_data["tw_zscore"] = np.clip(all_data["zscore"] / ZSCORE_CLIP, -1.0, 1.0)

        all_data["robust_z"] = (all_data["Pred"] / ((q75 - q25) / 1.349 + 1e-8)).fillna(0)
        all_data["tw_robust"] = np.clip(all_data["robust_z"] / ZSCORE_CLIP, -1.0, 1.0)

        all_data["tw_quantile"] = np.where(
            (all_data["Pred"] > q95) & (all_data["Pred"] > 0), 1.0,
            np.where((all_data["Pred"] < q05) & (all_data["Pred"] < 0), -1.0, 0.0)
        )

        direction = inst_directions.get(inst, 1.0)
        all_data["tw_zscore"] *= direction
        all_data["tw_robust"] *= direction
        all_data["tw_quantile"] *= direction

        all_data["raw_signal"] = all_data["Pred"]
        all_data["futures_code"] = all_data["contract"].apply(
            lambda c: parquet_contract_to_futures(prefix, str(c))
        )

        oos_only = all_data[all_data["index_time"] >= oos_start_time].copy()
        keep = ["index_time", "instrument", "contract", "futures_code",
                "raw_signal", "tw_zscore", "tw_robust", "tw_quantile"]
        parts.append(oos_only[keep])

    signals_df = pd.concat(parts, ignore_index=True)
    signals_df = signals_df.sort_values(["futures_code", "index_time"]).reset_index(drop=True)
    signals_df.to_parquet(SIGNALS_OUT, index=False)
    print(f"💾 信号: {SIGNALS_OUT} ({len(signals_df):,} 行)")
    print(f"   OOS 期间: {signals_df['index_time'].min()} → {signals_df['index_time'].max()}")
    return signals_df


# ═══════════════════════════════════════════════════════════════
# 4. 信号索引
# ═══════════════════════════════════════════════════════════════

def build_signal_index(signals_df: pd.DataFrame):
    minute_idx = {}
    daily_active = {}
    df = signals_df.copy()
    minutes = df["index_time"].dt.hour * 60 + df["index_time"].dt.minute
    ordinals = df["index_time"].map(lambda dt: dt.date().toordinal())
    for i, row in df.iterrows():
        inst = row["instrument"]
        key = (inst, int(ordinals[i]), int(minutes[i]))
        minute_idx[key] = {
            "tw_zscore":   float(row.get("tw_zscore", 0.0)),
            "tw_robust":   float(row.get("tw_robust", 0.0)),
            "tw_quantile": float(row.get("tw_quantile", 0.0)),
            "futures_code": row["futures_code"],
        }
        date_key = (inst, int(ordinals[i]))
        if date_key not in daily_active:
            daily_active[date_key] = row["futures_code"]
    return minute_idx, daily_active


# ═══════════════════════════════════════════════════════════════
# 5. NautilusTrader 回测环境与费率
# ═══════════════════════════════════════════════════════════════

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.data import OrderBookDepth10, BookOrder
from nautilus_trader.model.enums import (
    OrderSide, OrderStatus, TimeInForce,
    AccountType, OmsType, BookType, AssetClass,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, PositionId
from nautilus_trader.model.objects import Price, Quantity, Currency, Money
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.events.order import OrderFilled, OrderExpired
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.backtest.models.fee import FeeModel


class UniversalFuturesFeeModel(FeeModel):
    """支持固定金额/万分比混排的商品期货终极费率模型"""
    def __init__(self, cache=None):
        super().__init__()
        self._cache = cache

    def get_commission(self, order, fill_qty, fill_px, instrument):
        futures_code = instrument.raw_symbol.value
        prefix_match = re.match(r"^([A-Za-z]+)", futures_code)
        prefix = prefix_match.group(1).upper() if prefix_match else ""
        
        # 获取配置：(乘数, tick_size, 交易所, 保证金率, 计费模式, 费率数值)
        spec = INSTRUMENT_SPECS.get(prefix)
        if spec:
            fee_type, fee_val = spec[4], spec[5]
        else:
            fee_type, fee_val = "ratio", 0.0001
        
        if fee_type == "ratio":
            # 比例收费: 名义本金 × 万分比
            notional = instrument.multiplier.as_double() * fill_qty.as_double() * fill_px.as_double()
            commission = notional * float(fee_val)
        else: 
            # 固定收费: 手数 × 固定金额
            commission = fill_qty.as_double() * float(fee_val)
            
        return Money(commission, instrument.quote_currency)


QUANT360_COLUMN_MAP = {
    "pursbuy_one_prc": "bid_1_prc",   "pursbuy_one_qty": "bid_1_qty",
    "purssell_one_prc": "ask_1_prc",  "purssell_one_qty": "ask_1_qty",
    "pursbuy_two_prc": "bid_2_prc",   "pursbuy_two_qty": "bid_2_qty",
    "purssell_two_prc": "ask_2_prc",  "purssell_two_qty": "ask_2_qty",
    "pursbuy_three_prc": "bid_3_prc", "pursbuy_three_qty": "bid_3_qty",
    "purssell_three_prc": "ask_3_prc","purssell_three_qty": "ask_3_qty",
    "pursbuy_four_prc": "bid_4_prc",  "pursbuy_four_qty": "bid_4_qty",
    "purssell_four_prc": "ask_4_prc", "purssell_four_qty": "ask_4_qty",
    "pursbuy_five_prc": "bid_5_prc",  "pursbuy_five_qty": "bid_5_qty",
    "purssell_five_prc": "ask_5_prc", "purssell_five_qty": "ask_5_qty",
}

def _parse_qt(qt):
    s = str(int(qt)).zfill(17)
    return datetime(
        year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]),
        hour=int(s[8:10]), minute=int(s[10:12]), second=int(s[12:14]),
        microsecond=int(s[14:17]) * 1000, tzinfo=CN_TZ,
    )

def _safe_pq(prc, qty):
    if pd.isna(prc) or prc <= 0:
        return 0.0, 0.0
    if pd.isna(qty) or qty <= 0:
        return float(prc), 0.0
    return float(prc), float(qty)

def load_snapshots_from_files(
    target_contracts: set[str],
    file_paths: list[str]
) -> dict[str, list[OrderBookDepth10]]:
    snaps_by_inst: dict[str, list[OrderBookDepth10]] = {}
    
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
            
        # 💡 终极修复：统一大写，并将郑商所的 3 位代码 (如 TA601) 强制补齐为 4 位 (TA2601)
        df["secu_cd"] = df["secu_cd"].astype(str).str.upper()
        df["secu_cd"] = df["secu_cd"].apply(lambda x: re.sub(r'^([A-Z]+)(\d{3})$', r'\g<1>2\2', x))
            
        df = df[df["secu_cd"].isin(target_contracts)]
        if len(df) == 0:
            continue
            
        df["_timestamp"] = df["quot_time"].apply(_parse_qt)
        df = df.sort_values("_timestamp").rename(columns=QUANT360_COLUMN_MAP)

        for ctr, grp in df.groupby("secu_cd"):
            prefix_match = re.match(r"^([A-Za-z]+)", str(ctr))
            prefix = prefix_match.group(1).upper() if prefix_match else ""
            exchange = INSTRUMENT_SPECS.get(prefix, (0, 0, "SIM", 0, "ratio", 0))[2]
            
            inst_str = f"{ctr}.{exchange}"
            snaps = snaps_by_inst.setdefault(inst_str, [])
            for _, row in grp.iterrows():
                ts_ns = int(row["_timestamp"].timestamp() * 1e9)
                bids, asks, b_cts, a_cts = [], [], [], []
                for i in range(1, 6):
                    bp, bq = _safe_pq(row.get(f"bid_{i}_prc"), row.get(f"bid_{i}_qty"))
                    ap, aq = _safe_pq(row.get(f"ask_{i}_prc"), row.get(f"ask_{i}_qty"))
                    bids.append(BookOrder(OrderSide.BUY, Price.from_str(str(bp)), Quantity.from_str(str(int(bq))), 0))
                    asks.append(BookOrder(OrderSide.SELL, Price.from_str(str(ap)), Quantity.from_str(str(int(aq))), 0))
                    b_cts.append(0); a_cts.append(0)
                zero = BookOrder(OrderSide.BUY, Price.from_str("0"), Quantity.from_str("0"), 0)
                for _ in range(6, 11):
                    bids.append(zero); asks.append(zero)
                    b_cts.append(0); a_cts.append(0)
                snaps.append(OrderBookDepth10(
                    InstrumentId.from_str(inst_str), bids, asks, b_cts, a_cts, 0, 0, ts_ns, ts_ns
                ))

    return snaps_by_inst


# ---- 5b. NautilusTrader Strategy 子类 ----

def _top_of_book(snap: OrderBookDepth10):
    best_bid, best_ask = 0.0, 0.0
    for b in snap.bids:
        bp, bs = b.price.as_double(), b.size.as_double()
        if bp > 0 and bs > 0:
            best_bid = bp; break
    for a in snap.asks:
        ap, as_ = a.price.as_double(), a.size.as_double()
        if ap > 0 and as_ > 0:
            best_ask = ap; break
    mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
    return best_bid, best_ask, mid

def _best_size(snap: OrderBookDepth10, side: OrderSide) -> float:
    orders = snap.bids if side == OrderSide.BUY else snap.asks
    for o in orders:
        p, s = o.price.as_double(), o.size.as_double()
        if p > 0 and s > 0: return s
    return 0.0

def compute_target_lots(target_weight: float, mid_price: float, prefix: str, 
                        capital: float = TOTAL_CAPITAL) -> int:
    multiplier = INSTRUMENT_SPECS.get(prefix, (10, 1.0))[0]
    notional = mid_price * multiplier
    max_allowed = max(1, int(MAX_WEIGHT_PER_INST * capital / notional))
    return int(round(target_weight * max_allowed))


class CommodityIntradayStrategy(Strategy):
    _SEC_UTC_TO_CN = 28800          
    _SEC_PER_DAY    = 86400
    _EPOCH_ORDINAL  = 719163        

    def __init__(self, minute_idx: dict, daily_active: dict, all_instruments: list[str], 
                 mapping_type: str = "zscore", config=None, per_inst_capital: float = TOTAL_CAPITAL):
        super().__init__(config)
        self.minute_idx = minute_idx
        self.daily_active = daily_active
        self._all_instruments = all_instruments
        self._prefix_to_parquet = COMMODITY_MAPPING 

        self._mapping_type = mapping_type
        self._mapping_field = f"tw_{mapping_type}"

        self._active_contract: dict[str, str] = {}     
        self._next_day_contract: dict[str, str] = {}   

        # 基础状态控制
        self._current_day_ordinal: Optional[int] = None
        self._last_minute: Optional[int] = None        
        self._last_slot: Optional[int] = None          
        
        self._eod_closed_insts: set[str] = set()       
        self._active_windows: dict[str, dict] = {}     

        # 轨道 A (Fixed Grid) 的最高水位线追仓配置
        self._addon_window_min = ADDON_WINDOW_MIN                   
        self._current_slot_start_min = None          
        self._slot_extreme_signals: dict[str, float] = {} 
        self._slot_target_lots: dict[str, int] = {}  # 💡 记录当前窗格累计下达的目标持仓

        # 💡 并发订单追踪字典 {inst_str: {client_order_id: {"phase": 1, "order": order, "target_lots": lots}}}
        self._gtq_orders: dict[str, dict] = {}
        self._latest_depth: dict[str, OrderBookDepth10] = {}

        self._per_inst_capital = per_inst_capital
        self._commission_by_inst: dict[str, float] = {}

    def on_start(self) -> None:
        for inst_str in self._all_instruments:
            inst_id = InstrumentId.from_str(inst_str)
            self.subscribe_order_book_depth(
                instrument_id=inst_id, book_type=BookType.L2_MBP, depth=0, managed=False
            )
        self.log.info(f"Strategy started [{self._mapping_type} mode]")

    def on_stop(self) -> None:
        for inst_str in list(self._gtq_orders.keys()):
            self._cancel_all_gtds(inst_str)
        self._gtq_orders.clear()

    # ============================================================
    # 核心回调 — 分钟级调度
    # ============================================================
    def on_order_book_depth(self, depth: OrderBookDepth10) -> None:
        inst_str = str(depth.instrument_id)
        self._latest_depth[inst_str] = depth

        cn_seconds = depth.ts_event // 1_000_000_000 + self._SEC_UTC_TO_CN
        day_ordinal = cn_seconds // self._SEC_PER_DAY + self._EPOCH_ORDINAL
        minute_of_day = (cn_seconds % self._SEC_PER_DAY) // 60  

        if self._current_day_ordinal != day_ordinal:
            self._on_new_day(day_ordinal)
            self._current_day_ordinal = day_ordinal
            self._eod_closed_insts.clear()
            self._active_windows.clear()
            self._slot_target_lots.clear()
            self._last_minute = None
            self._last_slot = None
            self._current_slot_start_min = None  

        if self._last_minute is None or minute_of_day != self._last_minute:
            self._last_minute = minute_of_day
            self._evaluate_minute_logic(day_ordinal, minute_of_day, cn_seconds)

        self._check_gtq(inst_str, depth)

    # ============================================================
    # 双轨制调度中心
    # ============================================================
    def _evaluate_minute_logic(self, day_ordinal: int, minute_of_day: int, cn_seconds: int):
        import datetime as pydt
        d_date = pydt.date.fromordinal(day_ordinal)
        is_friday_or_holiday = (d_date.weekday() == 4) 
        is_saturday = (d_date.weekday() == 5)

        # ── 1. 强制平仓拦截器 (按品种独立判断) ──
        for prefix, parquet_inst in self._prefix_to_parquet.items():
            active_inst = self._active_contract.get(prefix)
            if not active_inst or active_inst in self._eod_closed_insts:
                continue

            force_flat_now = False
            
            # 【规则 A】：每天下午 14:30 (870) 强平
            if 870 <= minute_of_day < 900:
                force_flat_now = True
            
            # 【规则 B】：周末或节假日前的夜盘强平
            elif is_friday_or_holiday and minute_of_day >= 1260:
                close_min = NIGHT_CLOSE_MINS.get(prefix, 1380)    
                if close_min >= 1380 and minute_of_day >= close_min - 30:
                    force_flat_now = True
            elif is_saturday and minute_of_day <= 150:            
                close_min = NIGHT_CLOSE_MINS.get(prefix, 150)
                if minute_of_day >= close_min - 30:
                    force_flat_now = True
                    
            if force_flat_now:
                depth = self._latest_depth.get(active_inst)
                if depth:
                    self._eod_closed_insts.add(active_inst)
                    self._active_windows.pop(active_inst, None)  
                    current_lots = self._get_current_lots(active_inst)
                    self.log.info(f"⚙ REBAL {prefix} [FORCED FLAT] cur={current_lots}")
                    
                    self._cancel_all_gtds(active_inst)
                    if current_lots != 0:
                        side = OrderSide.SELL if current_lots > 0 else OrderSide.BUY
                        self._issue_order(active_inst, abs(current_lots), side, depth, 0)

        # ── 2. 正常交易时段判定 (天然跳过开盘第一分钟) ──
        is_trading = (541 <= minute_of_day < 870) or (1261 <= minute_of_day <= 1439) or (1 <= minute_of_day <= 150)
        
        if is_trading:
            if self._mapping_type in ["zscore", "robust"]:
                slot = day_ordinal * 48 + minute_of_day // 30
                is_new_slot = (self._last_slot is None or slot != self._last_slot)
                
                if is_new_slot:
                    self._last_slot = slot
                    self._current_slot_start_min = minute_of_day
                    self._slot_extreme_signals.clear()
                    
                    for prefix, parquet_inst in self._prefix_to_parquet.items():
                        if self._active_contract.get(prefix) in self._eod_closed_insts: continue
                        self._rebalance_fixed_grid(prefix, parquet_inst, day_ordinal, minute_of_day, is_new_slot=True)
                else:
                    if self._addon_window_min > 0 and (minute_of_day - self._current_slot_start_min) <= self._addon_window_min:
                        for prefix, parquet_inst in self._prefix_to_parquet.items():
                            if self._active_contract.get(prefix) in self._eod_closed_insts: continue
                            self._rebalance_fixed_grid(prefix, parquet_inst, day_ordinal, minute_of_day, is_new_slot=False)
            
            elif self._mapping_type == "quantile":
                for prefix, parquet_inst in self._prefix_to_parquet.items():
                    if self._active_contract.get(prefix) in self._eod_closed_insts: continue
                    self._rebalance_dynamic_event(prefix, parquet_inst, day_ordinal, minute_of_day, cn_seconds)

    # ============================================================
    # 轨道 A：带追加机制的定时调仓 (High-Water Mark 模式)
    # ============================================================
    def _rebalance_fixed_grid(self, prefix, parquet_inst, day_ordinal, minute_of_day, is_new_slot: bool):
        active_inst = self._active_contract.get(prefix)
        if not active_inst: return
        depth = self._latest_depth.get(active_inst)
        if not depth: return
        mid = _top_of_book(depth)[2]
        if mid <= 0: return

        signal = self._lookup_signal(parquet_inst, day_ordinal, minute_of_day)
        if not signal: return
        sig_val = signal[self._mapping_field]
        current_lots = self._get_current_lots(active_inst)

        if is_new_slot:
            self._slot_extreme_signals[active_inst] = sig_val
            target_lots = compute_target_lots(sig_val, mid, prefix, self._per_inst_capital)
            self._slot_target_lots[active_inst] = target_lots
            
            self._cancel_all_gtds(active_inst)
            delta = target_lots - current_lots
            self.log.info(f"⚙ GRID REBAL {prefix} sig={sig_val:.3f} target={target_lots} cur={current_lots}")
            
            if delta != 0:
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                self._issue_order(active_inst, abs(delta), side, depth, target_lots)
        else:
            extreme_sig = self._slot_extreme_signals.get(active_inst, 0.0)
            if (extreme_sig > 0 and sig_val > extreme_sig) or (extreme_sig < 0 and sig_val < extreme_sig):
                self._slot_extreme_signals[active_inst] = sig_val
                new_target_lots = compute_target_lots(sig_val, mid, prefix, self._per_inst_capital)
                old_target_lots = self._slot_target_lots.get(active_inst, current_lots)
                addon_delta = new_target_lots - old_target_lots
                
                if addon_delta != 0:
                    is_adding_long = (extreme_sig > 0 and addon_delta > 0)
                    is_adding_short = (extreme_sig < 0 and addon_delta < 0)
                    if is_adding_long or is_adding_short:
                        self.log.info(f"🚀 GRID ADD-ON {prefix} sig: {extreme_sig:.3f} -> {sig_val:.3f}, adding {addon_delta} lots")
                        self._slot_target_lots[active_inst] = new_target_lots
                        side = OrderSide.BUY if addon_delta > 0 else OrderSide.SELL
                        # 💡 绝不撤旧单！仅发差额新单并发排队
                        self._issue_order(active_inst, abs(addon_delta), side, depth, new_target_lots)

    # ============================================================
    # 轨道 B：事件驱动的 30 分钟动态沙漏
    # ============================================================
    def _rebalance_dynamic_event(self, prefix, parquet_inst, day_ordinal, minute_of_day, cn_seconds):
        active_inst = self._active_contract.get(prefix)
        if not active_inst: return
        depth = self._latest_depth.get(active_inst)
        if not depth: return
        mid = _top_of_book(depth)[2]
        if mid <= 0: return

        signal = self._lookup_signal(parquet_inst, day_ordinal, minute_of_day)
        sig_val = signal[self._mapping_field] if signal else 0.0
        current_lots = self._get_current_lots(active_inst)
        window = self._active_windows.get(active_inst)

        if window is None:
            if sig_val != 0:
                target_lots = compute_target_lots(sig_val, mid, prefix, self._per_inst_capital)
                self._active_windows[active_inst] = {
                    'expire_sec': self._add_trading_seconds(cn_seconds, 30 * 60),
                    'target_lots': target_lots
                }
                self.log.info(f"⚙ EVENT TRIGGER {prefix} sig={sig_val} -> target={target_lots}")
                self._cancel_all_gtds(active_inst)
                delta = target_lots - current_lots
                if delta != 0:
                    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                    self._issue_order(active_inst, abs(delta), side, depth, target_lots)
        else:
            if cn_seconds >= window['expire_sec']:
                self._active_windows.pop(active_inst)
                self._cancel_all_gtds(active_inst)
                if sig_val != 0:
                    target_lots = compute_target_lots(sig_val, mid, prefix, self._per_inst_capital)
                    self._active_windows[active_inst] = {
                        'expire_sec': self._add_trading_seconds(cn_seconds, 30 * 60),
                        'target_lots': target_lots
                    }
                    self.log.info(f"⚙ EVENT ROLL {prefix} sig={sig_val} target={target_lots}")
                    delta = target_lots - current_lots
                    if delta != 0:
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        self._issue_order(active_inst, abs(delta), side, depth, target_lots)
                else:
                    self.log.info(f"⚙ EVENT EXPIRE {prefix} closing pos")
                    if current_lots != 0:
                        side = OrderSide.SELL if current_lots > 0 else OrderSide.BUY
                        self._issue_order(active_inst, abs(current_lots), side, depth, 0)
            else:
                if sig_val == 0:
                    if window['target_lots'] != 0:
                        self.log.info(f"⚙ EVENT CANCEL {prefix} sig->0. Keeping filled {current_lots} lots")
                        window['target_lots'] = 0
                        self._cancel_all_gtds(active_inst)
                else:
                    new_target = compute_target_lots(sig_val, mid, prefix, self._per_inst_capital)
                    old_target = window['target_lots']
                    if old_target != new_target:
                        addon = new_target - old_target
                        window['target_lots'] = new_target
                        if addon != 0:
                            self.log.info(f"⚙ EVENT UPDATE {prefix} sig->{sig_val}. New target={new_target}")
                            is_reversal = (old_target > 0 and new_target < 0) or (old_target < 0 and new_target > 0)
                            if is_reversal or old_target == 0:
                                self._cancel_all_gtds(active_inst)
                                delta = new_target - current_lots
                                if delta != 0:
                                    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                                    self._issue_order(active_inst, abs(delta), side, depth, new_target)
                            else:
                                # 💡 纯粹的同向追仓：绝不撤旧单！仅发差额新单并发排队
                                side = OrderSide.BUY if addon > 0 else OrderSide.SELL
                                self._issue_order(active_inst, abs(addon), side, depth, new_target)

    def _add_trading_seconds(self, start_sec: int, delta_sec: int) -> int:
        curr = start_sec
        rem = delta_sec
        while rem > 0:
            curr += 60
            mod = (curr % 86400) // 60
            if (540 <= mod < 615) or (630 <= mod < 690) or \
               (810 <= mod < 900) or (1260 <= mod < 1440) or (0 <= mod <= 150):
                rem -= 60
        return curr

    # ============================================================
    # 通用的下单与状态更新执行引擎 (1-to-N 架构)
    # ============================================================
    def _issue_order(self, active_inst: str, qty: int, side: OrderSide, depth: OrderBookDepth10, target_lots: int):
        """发单引擎：绝不撤单，支持并发行走"""
        if qty <= 0: return
        best_bid, best_ask, mid = _top_of_book(depth)
        limit_price = best_bid if side == OrderSide.BUY else best_ask
        if _best_size(depth, side) < MIN_BEST_QTY:
            limit_price = best_ask if side == OrderSide.BUY else best_bid

        expire_dt = self.clock.utc_now() + timedelta(seconds=GTD_STAGE1_SEC)
        order = self.order_factory.limit(
            instrument_id=InstrumentId.from_str(active_inst),
            order_side=side,
            quantity=Quantity.from_str(str(qty)),
            price=Price.from_str(str(limit_price)),
            time_in_force=TimeInForce.GTD,
            expire_time=expire_dt,
        )
        self.submit_order(order)
        
        if active_inst not in self._gtq_orders:
            self._gtq_orders[active_inst] = {}
        self._gtq_orders[active_inst][order.client_order_id.value] = {
            "phase": 1, "order": order, "target_lots": target_lots
        }

    def _cancel_all_gtds(self, inst_str):
        active_dict = self._gtq_orders.get(inst_str, {})
        for cid, gtd in list(active_dict.items()):
            order = gtd["order"]
            if order.status in (OrderStatus.INITIALIZED, OrderStatus.ACCEPTED, OrderStatus.SUBMITTED, OrderStatus.PENDING_UPDATE):
                try: self.cancel_order(order)
                except Exception: pass
        self._gtq_orders[inst_str] = {}

    def _cancel_single_gtd(self, inst_str, client_order_id):
        active_dict = self._gtq_orders.get(inst_str, {})
        gtd = active_dict.pop(client_order_id, None)
        if gtd:
            order = gtd["order"]
            if order.status in (OrderStatus.INITIALIZED, OrderStatus.ACCEPTED, OrderStatus.SUBMITTED, OrderStatus.PENDING_UPDATE):
                try: self.cancel_order(order)
                except Exception: pass

    def _get_current_lots(self, active_inst: str) -> int:
        positions = self.cache.positions_open(instrument_id=InstrumentId.from_str(active_inst))
        if positions:
            qty = int(positions[0].quantity.as_double())
            return -qty if positions[0].side == PositionSide.SHORT else qty
        return 0

    def _on_new_day(self, day_ordinal: int):
        next_ordinal = day_ordinal + 1
        for prefix, parquet_inst in self._prefix_to_parquet.items():
            today_fc = self.daily_active.get((parquet_inst, day_ordinal))
            if today_fc:
                exchange = INSTRUMENT_SPECS.get(prefix, (0, 0, "SIM"))[2]
                self._active_contract[prefix] = f"{today_fc}.{exchange}"
            tomorrow_fc = self.daily_active.get((parquet_inst, next_ordinal))
            if tomorrow_fc:
                exchange = INSTRUMENT_SPECS.get(prefix, (0, 0, "SIM"))[2]
                self._next_day_contract[prefix] = f"{tomorrow_fc}.{exchange}"
        self.log.info(f"📅 day_ordinal={day_ordinal} active={len(self._active_contract)} insts")

    def _lookup_signal(self, inst: str, day_ordinal: int, minute_of_day: int) -> dict | None:
        sig = self.minute_idx.get((inst, day_ordinal, minute_of_day))
        if sig is not None: return sig
        for back_m in range(1, 6):
            alt = minute_of_day - back_m
            if alt < 0: break
            sig = self.minute_idx.get((inst, day_ordinal, alt))
            if sig is not None: return sig
        return None

    # ============================================================
    # GTD 三阶段追价逻辑 (支持并发订单)
    # ============================================================
    def _check_gtq(self, inst_str, depth):
        active_dict = self._gtq_orders.get(inst_str, {})
        for cid, gtd in list(active_dict.items()):
            if gtd["phase"] != 1: continue
            order = gtd["order"]
            
            live_order = self.cache.order(order.client_order_id)
            if not live_order or live_order.status not in (OrderStatus.INITIALIZED, OrderStatus.ACCEPTED, OrderStatus.SUBMITTED, OrderStatus.PENDING_UPDATE):
                continue
                
            side, limit = order.side, order.price.as_double()
            best_bid, best_ask, _ = _top_of_book(depth)
            
            if side == OrderSide.BUY and best_ask > 0 and best_ask <= limit: continue
            if side == OrderSide.SELL and best_bid > 0 and best_bid >= limit: continue
            
            if side == OrderSide.BUY and best_bid > 0 and best_bid > limit:
                self._upgrade_to_stage2(inst_str, cid, best_ask)
            elif side == OrderSide.SELL and best_ask > 0 and best_ask < limit:
                self._upgrade_to_stage2(inst_str, cid, best_bid)

    def _upgrade_to_stage2(self, inst_str, cid, new_price):
        active_dict = self._gtq_orders.get(inst_str, {})
        gtd = active_dict.get(cid)
        if not gtd or new_price <= 0: return
        
        old_order = gtd["order"]
        target_lots = gtd["target_lots"]
        self._cancel_single_gtd(inst_str, cid)
        
        live_order = self.cache.order(old_order.client_order_id)
        if live_order:
            rem_qty = live_order.quantity.as_double() - live_order.filled_qty.as_double()
        else:
            rem_qty = old_order.quantity.as_double()
            
        if rem_qty <= 0: return
        
        side = old_order.side
        qty = Quantity.from_str(str(int(rem_qty)))
        expire_dt = self.clock.utc_now() + timedelta(seconds=GTD_STAGE2_SEC)
        
        new_order = self.order_factory.limit(
            instrument_id=InstrumentId.from_str(inst_str), order_side=side,
            quantity=qty, price=Price.from_str(str(new_price)),
            time_in_force=TimeInForce.GTD, expire_time=expire_dt,
        )
        self.submit_order(new_order)
        active_dict[new_order.client_order_id.value] = {"phase": 2, "order": new_order, "target_lots": target_lots}

    def on_order_filled(self, event: OrderFilled) -> None:
        inst_str = str(event.instrument_id)
        comm = getattr(event, "commission", None)            # 💡 新增
        if comm is not None:
            if isinstance(comm, (list, tuple)):
                amt = sum(c.as_double() for c in comm)
            else:
                amt = comm.as_double()
            self._commission_by_inst[inst_str] = \
                self._commission_by_inst.get(inst_str, 0.0) + amt
        cid = event.client_order_id.value
        
        live_order = self.cache.order(event.client_order_id)
        if live_order and live_order.status == OrderStatus.FILLED:
            active_dict = self._gtq_orders.get(inst_str, {})
            active_dict.pop(cid, None)
            
        self.log.info(f"✅ FILLED {inst_str} qty={event.last_qty.as_double()} px={event.last_px.as_double():.2f}")

    def on_order_expired(self, event: OrderExpired) -> None:
        inst_str = str(event.instrument_id)
        cid = event.client_order_id.value
        
        active_dict = self._gtq_orders.get(inst_str, {})
        gtd = active_dict.pop(cid, None)
        if not gtd: return
        
        phase = gtd["phase"]
        old_order = gtd["order"]
        target_lots = gtd["target_lots"]

        depth = self._latest_depth.get(inst_str)
        if depth is None: return
        best_bid, best_ask, _ = _top_of_book(depth)
        if best_bid <= 0 or best_ask <= 0: return

        live_order = self.cache.order(old_order.client_order_id)
        if live_order:
            rem_qty = live_order.quantity.as_double() - live_order.filled_qty.as_double()
        else:
            rem_qty = old_order.quantity.as_double()
            
        if rem_qty <= 0: return

        side = old_order.side
        qty = Quantity.from_str(str(int(rem_qty)))
        new_price = best_ask if side == OrderSide.BUY else best_bid

        if phase == 1:
            expire_dt = self.clock.utc_now() + timedelta(seconds=GTD_STAGE2_SEC)
            new_order = self.order_factory.limit(
                instrument_id=InstrumentId.from_str(inst_str), order_side=side,
                quantity=qty, price=Price.from_str(str(new_price)),
                time_in_force=TimeInForce.GTD, expire_time=expire_dt,
            )
            self.submit_order(new_order)
            active_dict[new_order.client_order_id.value] = {"phase": 2, "order": new_order, "target_lots": target_lots}
        else:
            new_order = self.order_factory.limit(
                instrument_id=InstrumentId.from_str(inst_str), order_side=side,
                quantity=qty, price=Price.from_str(str(new_price)),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(new_order)
            active_dict[new_order.client_order_id.value] = {"phase": 3, "order": new_order, "target_lots": target_lots}


# ---- 5c. 品种注册 + 策略构造 ----
def _make_instrument(futures_code: str) -> FuturesContract:
    prefix_match = re.match(r"^([A-Za-z]+)", futures_code)
    prefix = prefix_match.group(1).upper() if prefix_match else ""
    
    # 获取字典中的元组，默认给万1比例收费
    spec = INSTRUMENT_SPECS.get(prefix)
    if spec:
        multiplier, tick_size, exchange, margin, fee_type, fee_val = spec
    else:
        multiplier, tick_size, exchange, margin, fee_type, fee_val = 10, 1.0, "SIM", 0.15, "ratio", 0.0001
        
    price_inc = Price.from_str(str(tick_size))

    code = futures_code[len(prefix):]
    yy, mm = 2000 + int(code[:2]), int(code[2:4])
    expiry_dt = datetime(yy, mm, 15, tzinfo=CN_TZ)
    expiration_ns = int(expiry_dt.timestamp() * 1e9)

    return FuturesContract(
        instrument_id=InstrumentId.from_str(f"{futures_code}.{exchange}"),
        raw_symbol=Symbol(futures_code),
        asset_class=AssetClass.COMMODITY,
        currency=Currency.from_str("CNY"),
        price_precision=price_inc.precision,
        price_increment=price_inc,
        multiplier=Quantity.from_str(str(multiplier)),
        lot_size=Quantity.from_str("1"),
        underlying=prefix,
        activation_ns=0, expiration_ns=expiration_ns, ts_event=0, ts_init=0,
        margin_init=Decimal(str(margin)), margin_maint=Decimal(str(margin)),
        # 💡 Nautilus自带的 fee 字段是强行当做万分比算的。
        # 为了兼容绝对金额收费(Fixed)，这里直接赋0。实际扣费完全由上面的 UniversalFuturesFeeModel 接管！
        maker_fee=Decimal("0"), taker_fee=Decimal("0"),
    )


# ═══════════════════════════════════════════════════════════════
# 6. 回测主入口 (Memory-Safe 每日独立循环模式)
# ═══════════════════════════════════════════════════════════════

def run_nt_backtest_multi_day(signals_df: pd.DataFrame, start_date_str: str = None, end_date_str: str = None):
    minute_idx, daily_active = build_signal_index(signals_df)
    print(f"📋 信号索引: {len(minute_idx)} 分钟级 + {len(daily_active)} 日级")

    all_futures = set(daily_active.values()) | {s["futures_code"] for s in minute_idx.values()}
    
    exchanges_needed = set()
    for fc in all_futures:
        prefix_match = re.match(r"^([A-Za-z]+)", fc)
        prefix = prefix_match.group(1).upper() if prefix_match else ""
        exchanges_needed.add(INSTRUMENT_SPECS.get(prefix, (0,0,"SIM",0,"ratio",0))[2])

    # 💡 核心突破：无视文件名和物理路径，直接读 CSV 第一行拿官方交易日
    print(f"\n🔍 开始全局扫描文件树，建立官方交易日(Trading Day)索引 (自动解耦夜盘错位)...")
    base_dir = str(project_root / "tick_data" / "2025")
    date_to_files = {}
    
    import time
    scan_t0 = time.time()
    
    for exch in exchanges_needed:
        exch_lower = exch.lower()
        exch_upper = exch.upper()
        
        day_pattern = os.path.join(base_dir, "*", "*", f"*q.{exch_lower}.futr", f"L2_new_FUTR_{exch_upper}_day_*.csv")
        night_pattern = os.path.join(base_dir, "*", "*", "Night", f"*q.{exch_lower}.futr", f"L2_new_FUTR_{exch_upper}_night_*.csv")
        
        for pattern in [day_pattern, night_pattern]:
            for f in glob.glob(pattern):
                try:
                    # 💡 只读取 1 列、1 行，网络 I/O 速度极快，找出它真正的归属
                    trd_day_raw = pd.read_csv(f, usecols=["trd_day"], nrows=1).iloc[0, 0]
                    # 将 "2025-12-01" 转换为 "20251201"
                    trd_day = str(trd_day_raw).replace("-", "")
                    
                    # 在第一道关卡就做拦截，大幅节省后续开销
                    if start_date_str and trd_day < str(start_date_str):
                        continue
                    if end_date_str and trd_day > str(end_date_str):
                        continue
                        
                    date_to_files.setdefault(trd_day, []).append(f)
                except Exception as e:
                    # 如果读不到 trd_day 或者文件损坏，直接跳过
                    continue

    sorted_trd_dates = sorted(list(date_to_files.keys()))
    scan_t1 = time.time()
    print(f"✅ 扫描完成 (耗时 {scan_t1 - scan_t0:.1f}s)，共匹配到 {len(sorted_trd_dates)} 个有效交易日的数据！")

    if not sorted_trd_dates:
        print("❌ 严重警告：未匹配到任何文件！请检查数据路径或命令行日期参数！")
        return None

        # ====================================================================
    # 💡 终极架构重构：按官方交易日循环，连续无损沙盒
    # ====================================================================
    print(f"\n🚀 开始按日独立沙盒循环回测 (极致省内存模式)...")

    # 全局指标记录仪
    global_stats = {
        "total_orders": 0, "filled": 0, "expired": 0, "canceled": 0,
        "net_pnl": 0.0, "total_ticks": 0,
        "daily_equity": {} # 💡 新增：用于记录每天跑完后的总资产（资金曲线缝合）
    }
    
    start_balance = TOTAL_CAPITAL * len(COMM_POOL)
    start_time_all = pd.Timestamp.now()
    
    # 💡 我们自己维护一个累加的真实净值水池
    running_total_equity = float(start_balance)
    inst_stats: dict[str, dict] = {}                     # 分品种累计

    def _blank():
        return {"pnl_gross": 0.0, "commission": 0.0, "volume": 0.0,
                "orders": 0, "filled": 0, "days": 0, "win_days": 0, "loss_days": 0}

    for trd_date in sorted_trd_dates:
        t_day_start = pd.Timestamp.now()
        # 💡 当日资金基准 = 昨日收盘净值
        day_start_equity = running_total_equity
        per_inst_capital = day_start_equity / len(COMM_POOL)

        file_paths = date_to_files[trd_date]
        snaps_by_inst = load_snapshots_from_files(all_futures, file_paths)
        day_snaps = sum(len(v) for v in snaps_by_inst.values())
        if day_snaps == 0:
            print(f"   [{trd_date}] ⚠️ 数据为空，跳过。")
            continue

        engine = BacktestEngine(config=BacktestEngineConfig(trader_id=f"BT-{trd_date}"))
        for ex in exchanges_needed:
            engine.add_venue(
                venue=Venue(ex), oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                # 💡 关键改动：每日初始资金 = 昨日净值
                starting_balances=[Money.from_str(f"{day_start_equity:.2f} CNY")],
                base_currency=Currency.from_str("CNY"), book_type=BookType.L2_MBP,
                fee_model=UniversalFuturesFeeModel(cache=engine.cache),
            )

        registered = {}
        for fc in sorted(all_futures):
            instrument = _make_instrument(fc)
            engine.add_instrument(instrument)
            registered[fc] = str(instrument.id)

        for inst_str, snap_list in snaps_by_inst.items():
            engine.add_data(snap_list)

        all_inst_strs = list(registered.values())
        strategy = CommodityIntradayStrategy(
            minute_idx, daily_active, all_inst_strs, mapping_type=MAPPING_TYPE,
            per_inst_capital=per_inst_capital,                # 💡 注入当日资金
            config=StrategyConfig(strategy_id=f"Strat-{trd_date}", order_id_tag="001",
                                  oms_type=OmsType.NETTING, manage_gtd_expiry=True),
        )
        engine.add_strategy(strategy)

        # 运行当日回测
        engine.run()

        # ── 当日结算（唯一一次，基准 = 昨日净值） ──
        day_net_pnl = 0.0
        for acc in engine.cache.accounts():
            bal = acc.balances_total().get(Currency.from_str("CNY"))
            if bal is not None:
                day_net_pnl += float(bal) - day_start_equity

        orders = engine.cache.orders()
        n_orders = len(orders)
        n_filled = sum(1 for o in orders if o.status == OrderStatus.FILLED)
        n_expired = sum(1 for o in orders if o.status == OrderStatus.EXPIRED)
        n_canceled = sum(1 for o in orders if o.status == OrderStatus.CANCELED)

        # ── 分品种聚合 ──
        day_inst_pnl = {}
        for pos in engine.cache.positions():                    # EOD 平仓后含已平仓位
            iid = str(pos.instrument_id)
            try:
                pnl = pos.realized_pnl.as_double()
            except Exception:
                pnl = 0.0
            day_inst_pnl[iid] = day_inst_pnl.get(iid, 0.0) + pnl

        for iid, pnl in day_inst_pnl.items():
            s = inst_stats.setdefault(iid, _blank())
            s["pnl_gross"] += pnl
            s["days"] += 1
            if pnl > 0: s["win_days"] += 1
            elif pnl < 0: s["loss_days"] += 1

        for iid, comm in strategy._commission_by_inst.items():
            inst_stats.setdefault(iid, _blank())["commission"] += comm

        for o in engine.cache.orders():
            s = inst_stats.setdefault(str(o.instrument_id), _blank())
            s["orders"] += 1
            if o.status == OrderStatus.FILLED:
                s["filled"] += 1
                s["volume"] += o.filled_qty.as_double()

        # 累加到全局追踪器
        global_stats["total_orders"] += n_orders
        global_stats["filled"] += n_filled
        global_stats["expired"] += n_expired
        global_stats["canceled"] += n_canceled
        global_stats["net_pnl"] += day_net_pnl
        global_stats["total_ticks"] += day_snaps
        
        # 💡 核心缝合：更新大水池的水位，并记录到当日账本
        running_total_equity += day_net_pnl
        global_stats["daily_equity"][trd_date] = running_total_equity

        t_day_end = pd.Timestamp.now()
        day_elapsed = (t_day_end - t_day_start).total_seconds()

        print(f"   [{trd_date}] ✓ 日盈亏: {day_net_pnl:>9,.0f} | 净值: {running_total_equity:>11,.0f} | 本金: {day_start_equity:>11,.0f} | ...")
        # ... engine.dispose() / gc 不变 ..
        engine.dispose()
        del engine
        del strategy
        del snaps_by_inst
        gc.collect()  

    elapsed_all = (pd.Timestamp.now() - start_time_all).total_seconds()
    print(f"\n✅ 全部交易日遍历完成 (总耗时: {elapsed_all:.0f}s)")

    # 💡 传入初始资金和统计字典，打印最专业的报告！
    _print_global_stitched_performance(global_stats, start_balance)
    _print_per_instrument_stats(inst_stats, len(sorted_trd_dates))
    return None

# ═══════════════════════════════════════════════════════════════
# 7. 业绩打印 (资金曲线缝合专业报告)
# ═══════════════════════════════════════════════════════════════

def _print_per_instrument_stats(inst_stats: dict, n_total_days: int):
    print("\n" + "=" * 96)
    print("📊 分品种业绩统计 (Gross PnL 未含手续费 / Net = Gross - 手续费)")
    print("=" * 96)
    header = (f"{'品种':<10}{'净盈亏(CNY)':>14}{'毛盈亏':>13}{'手续费':>11}"
              f"{'成交量(手)':>11}{'订单':>7}{'成交率':>8}{'交易天数':>9}{'胜日':>7}")
    print(header)
    print("-" * 96)

    rows = []
    for iid, s in inst_stats.items():
        net = s["pnl_gross"] - s["commission"]
        fill_rate = s["filled"] / s["orders"] * 100 if s["orders"] else 0
        rows.append((net, iid, s, fill_rate))

    for net, iid, s, fill_rate in sorted(rows, key=lambda r: -r[0]):
        print(f"{iid:<10}{net:>14,.0f}{s['pnl_gross']:>13,.0f}{s['commission']:>11,.0f}"
              f"{s['volume']:>11,.0f}{s['orders']:>7}{fill_rate:>7.1f}%"
              f"{s['days']:>9}{s['win_days']:>7}")

    total_net = sum(r[0] for r in rows)
    total_comm = sum(s["commission"] for s in inst_stats.values())
    print("-" * 96)
    print(f"{'合计':<10}{total_net:>14,.0f}{'' :>13}{total_comm:>11,.0f}"
          f"{sum(s['volume'] for s in inst_stats.values()):>11,.0f}")
    print("=" * 96)

def _print_global_stitched_performance(stats, start_balance):
    print("\n" + "=" * 80)
    print("📊 终极全局回测业绩摘要 (Stitched Curve)")
    print("=" * 80)
    
    # 提取时间序列并转换为 Pandas Series
    dates = list(stats['daily_equity'].keys())
    equities = list(stats['daily_equity'].values())
    
    if not equities:
        print("❌ 没有产生任何资金变动数据。")
        return

    # 前面插一个基准点 (第0天的初始资金)
    s_eq = pd.Series([start_balance] + equities)
    
    # 计算日收益率序列 (过滤掉未开仓导致的 0 收益日，让统计更真实)
    s_ret = pd.Series(equities).diff().dropna() / start_balance
    active_days = len(s_ret[s_ret != 0])
    
    # 计算累计绝对收益
    total_net_pnl = equities[-1] - start_balance
    total_ret = (equities[-1] / start_balance) - 1.0
    
    # 年化指标 (假设一年 252 个交易日)
    n_days = len(equities)
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1.0 if n_days > 0 else 0
    ann_vol = s_ret.std() * np.sqrt(252) if len(s_ret) > 1 else 0
    
    # 夏普比率 (无风险利率设为 0)
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0
    
    # 最大回撤 (MDD) 和 卡玛比率 (Calmar)
    running_max = s_eq.cummax()
    drawdowns = (s_eq - running_max) / running_max
    mdd = drawdowns.min()
    calmar = (ann_ret / abs(mdd)) if mdd < 0 else (ann_ret / 1e-6)
    
    # 日度胜率
    win_days = len(s_ret[s_ret > 0])
    loss_days = len(s_ret[s_ret < 0])
    win_rate = win_days / (win_days + loss_days) if (win_days + loss_days) > 0 else 0
    
    print(f"  测试模式:       {MAPPING_TYPE.upper()} (追加窗口: {ADDON_WINDOW_MIN}分钟)")
    print(f"  累计处理快照:   {stats['total_ticks']:,} 帧")
    print(f"  测试交易日数:   {n_days} 天 (其中有真实仓位变动: {active_days} 天)")
    print(f"")
    print(f"  初始总资金:     {start_balance:,.2f} CNY")
    print(f"  期末总净值:     {equities[-1]:,.2f} CNY")
    print(f"  累计净利润:     {total_net_pnl:,.2f} CNY (已由底层完美扣除手续费)")
    print(f"")
    print(f"  累计收益率:     {total_ret * 100:.2f}%")
    print(f"  年化收益率:     {ann_ret * 100:.2f}%")
    print(f"  年化波动率:     {ann_vol * 100:.2f}%")
    print(f"  最大回撤:       {mdd * 100:.2f}%")
    print(f"")
    print(f"  夏普比率(Sharpe): {sharpe:.3f}")
    print(f"  卡玛比率(Calmar): {calmar:.3f}")
    print(f"  日度胜率:         {win_rate * 100:.1f}%")
    
    print(f"\n订单执行统计:")
    total = stats['total_orders']
    filled = stats['filled']
    expired = stats['expired']
    canceled = stats['canceled']
    
    print(f"  总发出订单: {total}")
    print(f"  完全成交:   {filled} | 自然过期: {expired} | 撤销/追价: {canceled}")
    
    if total > 0:
        filled_rate = filled / total * 100
        print(f"  整体成交率: {filled_rate:.1f}%")
        
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════
# 7. 业绩打印 (适配分块执行的聚合报告)
# ═══════════════════════════════════════════════════════════════

def _print_global_performance(stats):
    print("\n" + "=" * 80)
    print("📊 终极全局回测业绩摘要 (Memory-Safe Mode)")
    print("=" * 80)
    print(f"  测试模式:       {MAPPING_TYPE.upper()} (追加窗口: {ADDON_WINDOW_MIN}分钟)")
    print(f"  累计处理快照:   {stats['total_ticks']:,} 帧")
    print(f"  累计发生手续费: {stats['total_commission']:,.2f} CNY")
    
    pnl = stats['net_pnl']
    print(f"  累计绝对净利润: {pnl:,.2f} CNY (已扣除手续费)")
    
    if pnl > 0:
        print("  最终评价:       ⭐⭐⭐ 盈利")
    elif pnl < 0:
        print("  最终评价:       ⚠️ 亏损")
    else:
        print("  最终评价:       持平无交易")

    print(f"\n订单统计:")
    total = stats['total_orders']
    filled = stats['filled']
    expired = stats['expired']
    canceled = stats['canceled']
    
    print(f"  总发出订单: {total}")
    print(f"  完全成交:   {filled}")
    print(f"  自然过期:   {expired}")
    print(f"  撤销/追价:  {canceled}")
    
    if total > 0:
        filled_rate = filled / total * 100
        print(f"  整体成交率: {filled_rate:.1f}%")
        
    print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# 8. 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    train_only = "--train-only" in sys.argv
    backtest_only = "--backtest-only" in sys.argv
    
    start_date_str = "20251024"
    end_date_str = "20251231"
    
    for a in sys.argv:
        if a.startswith("--start-date="):
            start_date_str = a.split("=", 1)[1]
        elif a.startswith("--end-date="):
            end_date_str = a.split("=", 1)[1]
        elif a.startswith("--date="):
            start_date_str = a.split("=", 1)[1]

    if not backtest_only:
        print("=" * 80)
        print("阶段 1: 训练 LightGBM & 生成 OOS 信号")
        print("=" * 80)
        signals_df = train_and_generate_signals()
    else:
        print("📂 加载已有信号文件 ...")
        signals_df = pd.read_parquet(SIGNALS_OUT)
        print(f"   加载: {len(signals_df):,} 行")

    if not train_only:
        print("\n" + "=" * 80)
        print("阶段 2: NautilusTrader 回测")
        print("=" * 80)
        run_nt_backtest_multi_day(signals_df, start_date_str, end_date_str)

    print("\n✅ 全部工作已完成")