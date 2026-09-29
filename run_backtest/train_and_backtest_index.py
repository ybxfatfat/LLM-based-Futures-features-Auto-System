"""
=====================================================
1. 加载 parquet 因子数据 → 训练 LightGBM → 生成样本外 (OOS) 信号
2. NautilusTrader 正式回测：30分钟调仓 + GTD 限价单执行

池子: index (CSZ→IF, C500Z→IC, SS50Z→IH, C1000Z→IM)

使用:
    python train_and_backtest_index.py                          # 全量
    python train_and_backtest_index.py --train-only             # 只训练
    python train_and_backtest_index.py --backtest-only          # 只回测
    python train_and_backtest_index.py --date=20251231          # 指定回测日期
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
from datetime import datetime, timezone, timedelta, time as dt_time
from decimal import Decimal

# ═══════════════════════════════════════════════════════════════
# 0. 配置常量
# ═══════════════════════════════════════════════════════════════

CN_TZ = timezone(timedelta(hours=8))

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = str(project_root / "data")
PARQUET_PATH = f"{DATA_DIR}/factor_dataset_AllMarket_main_only_1Year.parquet"
FEATURES_PATH = f"{DATA_DIR}/best_features_index.json"
SIGNALS_OUT = f"{DATA_DIR}/oos_signals_index.parquet"
MODEL_OUT = f"{DATA_DIR}/lgb_model_index.pkl"

TARGET = "Ret_30m_Fwd"

# ---- 池子映射: 纯净版标准前缀 ----
POOL_CONFIG = {
    "IF": "IF",    # 沪深300
    "IC": "IC",    # 中证500
    "IH": "IH",    # 上证50
    "IM": "IM",    # 中证1000
}
# 期货品种参数: (乘数, tick_size)
INSTRUMENT_SPECS = {
    "IF": (300, 0.2),
    "IH": (300, 0.2),
    "IC": (200, 0.2),
    "IM": (200, 0.2),
}

FEE_RATE = 0.00030   # CFFEX 基准费率

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

# ---- 策略参数 ----
ZSCORE_LOOKBACK = 2000       # 滚动 z-score 窗口 (分钟 bar)
ZSCORE_CLIP = 3.0            # z-score 截断
MAX_WEIGHT_PER_INST = 1.0    # 单品种满仓
TOTAL_CAPITAL = 10_000_000   # 总资金 (CNY)
REBALANCE_INTERVAL_MIN = 30  # 调仓间隔
GTD_STAGE1_SEC = 45          # 阶段一：挂买一/卖一等 45 秒
GTD_STAGE2_SEC = 45          # 阶段二：追价再等 45 秒
MIN_BEST_QTY = 50            # 最低深度才挂单（否则直接吃）
ADDON_WINDOW_MIN = 5
MAPPING_TYPE = "zscore"  # zscore, robust, quantile


# ═══════════════════════════════════════════════════════════════
# 1. 工具函数
# ═══════════════════════════════════════════════════════════════

def parquet_contract_to_futures(prefix: str, contract: str) -> str:
    """parquet 合约 'YYYYMM' → 期货代码 'PREFIXYYM'"""
    yy = contract[2:4]
    mm = contract[4:6]
    return f"{prefix}{yy}{mm}"


# ═══════════════════════════════════════════════════════════════
# 2. 特征加载
# ═══════════════════════════════════════════════════════════════

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
    """训练 LightGBM（带波动率权重），生成 OOS 信号（支持多种 Mapping）。"""
    features = load_features()
    pool_instruments = list(POOL_CONFIG.keys())

    all_cols = ["index_time", "instrument", "contract", TARGET] + features
    table = pq.read_table(
        PARQUET_PATH, columns=[c for c in all_cols if c in pq.read_schema(PARQUET_PATH).names],
        filters=[("instrument", "in", pool_instruments)],
    )
    df = table.to_pandas()
    df["index_time"] = pd.to_datetime(df["index_time"]).dt.tz_localize(CN_TZ)
    df = df.dropna(subset=[TARGET]).sort_values("index_time").reset_index(drop=True)

    # 👑 核心绝招：不改 TARGET！计算历史波动率倒数作为 LightGBM 的 Sample Weight
    print("\n⚙️ 正在计算目标波动率权重 (Sample Weights)...")
    target_vol = df.groupby('instrument')[TARGET].transform(
        lambda x: x.shift(30).rolling(2000, min_periods=100).std()
    )
    df['Weight'] = 1.0 / (target_vol + 1e-8)
    df = df.dropna(subset=['Weight']).reset_index(drop=True)

    print(f"📦 加载: {len(df):,} 行, {len(features)} 特征, {df['instrument'].nunique()} 品种")
    print(f"   时间: {df['index_time'].min()} → {df['index_time'].max()}")

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
        sample_weight=w_train,  # 👑 喂给模型
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )
    n_trees = model.best_iteration_ or model.n_estimators_

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 模型: {MODEL_OUT}")

    # ⚡ 全量预测（训练期 + OOS，用于 z-score warm-up）
    df["Pred"] = model.predict(df[features])

    # ✅ 方向必须只依赖训练集，绝对不能看全量数据的 IC
    df_train = df.iloc[:split_idx]
    inst_directions = {}
    for inst in df_train['instrument'].unique():
        sub = df_train[df_train['instrument'] == inst].dropna(subset=['Pred', TARGET])
        inst_directions[inst] = 1.0 if (len(sub) >= 100 and sub['Pred'].corr(sub[TARGET]) >= 0) else -1.0
    print(f"📌 方向(基于训练集): {inst_directions}")

    # OOS IC 评估（只用 OOS 行）
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
    for inst in pool_instruments:
        prefix = POOL_CONFIG[inst]
        all_data = df[df["instrument"] == inst].copy()
        
        # 基础统计量
        r = all_data["Pred"].rolling(ZSCORE_LOOKBACK, min_periods=ZSCORE_LOOKBACK // 2)
        s = r.std()
        q75 = r.quantile(0.75)
        q25 = r.quantile(0.25)
        q95 = r.quantile(0.9)  # 对应你新代码里的 0.9
        q05 = r.quantile(0.1)  # 对应你新代码里的 0.1

        # 💡 1: Z-Score，不减均值 m，仅用标准差 s 缩放
        all_data["zscore"] = (all_data["Pred"] / (s + 1e-8)).fillna(0)
        all_data["tw_zscore"] = np.clip(all_data["zscore"] / ZSCORE_CLIP, -1.0, 1.0)

        # 💡 2: Robust Z，不减中位数 med，仅用 IQR 极差缩放
        all_data["robust_z"] = (all_data["Pred"] / ((q75 - q25) / 1.349 + 1e-8)).fillna(0)
        all_data["tw_robust"] = np.clip(all_data["robust_z"] / ZSCORE_CLIP, -1.0, 1.0)

        # 💡 3: Extreme 10%，加入了绝对符号约束
        all_data["tw_quantile"] = np.where(
            (all_data["Pred"] > q95) & (all_data["Pred"] > 0), 1.0,
            np.where((all_data["Pred"] < q05) & (all_data["Pred"] < 0), -1.0, 0.0)
        )

        # 乘上方向因子
        direction = inst_directions.get(inst, 1.0)
        all_data["tw_zscore"] *= direction
        all_data["tw_robust"] *= direction
        all_data["tw_quantile"] *= direction

        all_data["raw_signal"] = all_data["Pred"]
        all_data["futures_code"] = all_data["contract"].apply(
            lambda c: parquet_contract_to_futures(prefix, str(c))
        )

        # 按时间过滤 OOS
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
# 5. NautilusTrader 回测
# ═══════════════════════════════════════════════════════════════

# --- 所有 import 放这里，避免训练阶段加载 NautilusTrader ---
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.data import OrderBookDepth10, BookOrder
from nautilus_trader.model.enums import (
    OrderSide, OrderStatus, TimeInForce,
    AccountType, OmsType, BookType, AssetClass,
)
from nautilus_trader.model.identifiers import (
    InstrumentId, Symbol, Venue, PositionId,
)
from nautilus_trader.model.objects import Price, Quantity, Currency, Money
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.events.order import OrderFilled, OrderExpired
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.backtest.models.fee import FeeModel


# ---- 5a-0. CFFEX 自定义费率模型 ----

CFFEX_OPEN_BPS = Decimal("0.000023")       # 开仓/平昨: 0.23‱
CFFEX_CLOSE_TODAY_BPS = Decimal("0.00023") # 平今: 2.3‱ (×10)


class CffexIndexFeeModel(FeeModel):
    """
    CFFEX 股指期货真实费率：开仓/平昨 = 0.23‱，平今 = 2.3‱。

    需要注入 cache 引用才能在 get_commission 里查持仓判断开/平/今。
    """

    def __init__(self, cache=None):
        super().__init__()
        self._cache = cache

    def get_commission(self, order, fill_qty, fill_px, instrument):
        # NETTING 模式 position_id = "{inst}-{strategy}"
        pos_id = PositionId(f"{order.instrument_id}-{order.strategy_id}")

        # 判断平今（纯整数运算，不构造 Pandas Timestamp）
        is_close_today = False
        if self._cache is not None:
            pos = self._cache.position(pos_id)
            if pos is not None and pos.signed_qty != 0:
                if (pos.side == PositionSide.LONG and order.side == OrderSide.SELL) or \
                   (pos.side == PositionSide.SHORT and order.side == OrderSide.BUY):
                    ts_ns = order.ts_accepted if order.ts_accepted > 0 else order.ts_init
                    # UTC ns → UTC days: pos opened vs order accepted
                    pos_day = pos.ts_opened // 86400_000_000_000
                    order_day = ts_ns // 86400_000_000_000
                    if pos_day == order_day:
                        is_close_today = True

        rate = CFFEX_CLOSE_TODAY_BPS if is_close_today else CFFEX_OPEN_BPS
        # instrument.notional_value() 内置合约乘数（IF×300/IC×200/…）
        notional = instrument.multiplier.as_double() * fill_qty.as_double() * fill_px.as_double()
        commission = notional * float(rate)
        return Money(commission, instrument.quote_currency)


# ---- 5a. 快照加载（与 test.py 相同的逻辑） ----

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


def load_snapshots_for_date(
    target_contracts: set[str],    # ← 改：不再从 signal_index 取，直接传合约列表
    snapshot_date: str,
    base_dir: str = str(project_root / "tick_data" / "2025"),
) -> dict[str, list[OrderBookDepth10]]:
    signal_contracts = target_contracts
    snaps_by_inst: dict[str, list[OrderBookDepth10]] = {}

    for session in ["day", "night"]:
        pattern = os.path.join(
            base_dir, "**", f"*.q.cffex.futr",
            f"L2_new_FUTR_CFFEX_{session}_{snapshot_date}.csv",
        )
        for fp in sorted(glob.glob(pattern, recursive=True)):
            print(f"  ✓ {os.path.basename(fp)} ...")
            try:
                df = pd.read_csv(fp)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                print(f"  ⚠ 跳过空文件/缺失: {os.path.basename(fp)}")
                continue
            
            # 💥 引入防暴雷清洗：统一大写，并将可能出现的 3 位短代码强制补齐为 4 位
            df["secu_cd"] = df["secu_cd"].astype(str).str.upper()
            df["secu_cd"] = df["secu_cd"].apply(lambda x: re.sub(r'^([A-Z]+)(\d{3})$', r'\g<1>2\2', x))
                
            df = df[df["secu_cd"].isin(signal_contracts)]
            if len(df) == 0:
                continue
            df["_timestamp"] = df["quot_time"].apply(_parse_qt)
            df = df.sort_values("_timestamp")
            df = df.rename(columns=QUANT360_COLUMN_MAP)

            for ctr, grp in df.groupby("secu_cd"):
                inst_str = f"{ctr}.CFFEX"
                snaps = snaps_by_inst.setdefault(inst_str, [])
                for _, row in grp.iterrows():
                    ts_ns = int(row["_timestamp"].timestamp() * 1e9)
                    bids, asks, b_cts, a_cts = [], [], [], []
                    for i in range(1, 6):
                        bp, bq = _safe_pq(row.get(f"bid_{i}_prc"), row.get(f"bid_{i}_qty"))
                        ap, aq = _safe_pq(row.get(f"ask_{i}_prc"), row.get(f"ask_{i}_qty"))
                        bids.append(BookOrder(OrderSide.BUY,
                            Price.from_str(str(bp)), Quantity.from_str(str(int(bq))), 0))
                        asks.append(BookOrder(OrderSide.SELL,
                            Price.from_str(str(ap)), Quantity.from_str(str(int(aq))), 0))
                        b_cts.append(0); a_cts.append(0)
                    zero = BookOrder(OrderSide.BUY, Price.from_str("0"), Quantity.from_str("0"), 0)
                    for _ in range(6, 11):
                        bids.append(zero); asks.append(zero)
                        b_cts.append(0); a_cts.append(0)
                    snaps.append(OrderBookDepth10(
                        InstrumentId.from_str(inst_str),
                        bids, asks, b_cts, a_cts,
                        0, 0, ts_ns, ts_ns,
                    ))

    total = sum(len(v) for v in snaps_by_inst.values())
    print(f"   总计: {total:,} 快照, {len(snaps_by_inst)} 合约")
    return snaps_by_inst


# ---- 5b. NautilusTrader Strategy 子类 ----

def _top_of_book(snap: OrderBookDepth10):
    """安全获取 top-of-book: (best_bid, best_ask, mid)"""
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
        if p > 0 and s > 0:
            return s
    return 0.0


def compute_target_lots(target_weight: float, mid_price: float, prefix: str) -> int:
    """target_weight (-1~1) → 整数目标手数"""
    multiplier, _ = INSTRUMENT_SPECS.get(prefix, (1, 0.01))
    notional = mid_price * multiplier
    max_allowed = max(1, int(MAX_WEIGHT_PER_INST * TOTAL_CAPITAL / notional))
    return int(round(target_weight * max_allowed))


class IndexIntradayStrategy(Strategy):
    # ---- 常量 (纳秒基准) ----
    _SEC_UTC_TO_CN = 28800          
    _SEC_PER_DAY    = 86400
    _EPOCH_ORDINAL  = 719163        

    def __init__(self, minute_idx: dict, daily_active: dict, all_instruments: list[str], 
                 mapping_type: str = "zscore", config=None):
        super().__init__(config)
        self.minute_idx = minute_idx
        self.daily_active = daily_active
        self._all_instruments = all_instruments
        self._prefix_to_parquet = {v: k for k, v in POOL_CONFIG.items()} 

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
        # ── 1. 14:30 以后强制全平 (股指规则) ──
        if minute_of_day >= 870:
            for prefix, parquet_inst in self._prefix_to_parquet.items():
                active_inst = self._active_contract.get(prefix)
                if active_inst and active_inst not in self._eod_closed_insts:
                    depth = self._latest_depth.get(active_inst)
                    if depth:
                        self._eod_closed_insts.add(active_inst)
                        self._active_windows.pop(active_inst, None)
                        current_lots = self._get_current_lots(active_inst)
                        self.log.info(f"⚙ REBAL {prefix} [EOD FLAT] cur={current_lots}")
                        
                        self._cancel_all_gtds(active_inst)
                        if current_lots != 0:
                            side = OrderSide.SELL if current_lots > 0 else OrderSide.BUY
                            self._issue_order(active_inst, abs(current_lots), side, depth, 0)
            return

        # ── 2. 正常交易时段 (股指 09:31 - 14:29) ──
        if 571 <= minute_of_day < 870:
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
            target_lots = compute_target_lots(sig_val, mid, prefix)
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
                new_target_lots = compute_target_lots(sig_val, mid, prefix)
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
                target_lots = compute_target_lots(sig_val, mid, prefix)
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
                    target_lots = compute_target_lots(sig_val, mid, prefix)
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
                    new_target = compute_target_lots(sig_val, mid, prefix)
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
                                side = OrderSide.BUY if addon > 0 else OrderSide.SELL
                                self._issue_order(active_inst, abs(addon), side, depth, new_target)

    def _add_trading_seconds(self, start_sec: int, delta_sec: int) -> int:
        """解决股指中午休市 11:30 (690) 到 13:00 (780) 的时间跨越"""
        curr = start_sec
        rem = delta_sec
        while rem > 0:
            curr += 60
            mod = (curr % 86400) // 60
            if (570 <= mod < 690) or (780 <= mod < 900):
                rem -= 60
        return curr

    # ============================================================
    # 通用的下单与状态更新执行引擎 (1-to-N 架构)
    # ============================================================
    def _issue_order(self, active_inst: str, qty: int, side: OrderSide, depth: OrderBookDepth10, target_lots: int):
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
                self._active_contract[prefix] = f"{today_fc}.CFFEX"
            tomorrow_fc = self.daily_active.get((parquet_inst, next_ordinal))
            if tomorrow_fc:
                self._next_day_contract[prefix] = f"{tomorrow_fc}.CFFEX"
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
    multiplier, tick_size = INSTRUMENT_SPECS.get(prefix, (1, 0.01))
    price_inc = Price.from_str(str(tick_size))

    code = futures_code[len(prefix):]
    yy, mm = 2000 + int(code[:2]), int(code[2:4])
    expiry_dt = datetime(yy, mm, 15, tzinfo=CN_TZ)
    expiration_ns = int(expiry_dt.timestamp() * 1e9)

    # CFFEX 保证金 + 费率（2025）
    # spec: (保证金%, 开仓‱, 平今‱)  来源：中金所
    spec = {
        "IF": (12, 0.23, 2.3),
        "IH": (12, 0.23, 2.3),
        "IC": (14, 0.23, 2.3),
        "IM": (15, 0.23, 2.3),
    }
    margin_pct, open_bps, close_today_bps = spec.get(prefix, (12, 0.23, 2.3))

    return FuturesContract(
        instrument_id=InstrumentId.from_str(f"{futures_code}.CFFEX"),
        raw_symbol=Symbol(futures_code),
        asset_class=AssetClass.INDEX,
        currency=Currency.from_str("CNY"),
        price_precision=price_inc.precision,
        price_increment=price_inc,
        multiplier=Quantity.from_str(str(multiplier)),
        lot_size=Quantity.from_str("1"),
        underlying=prefix,
        activation_ns=0,
        expiration_ns=expiration_ns,
        ts_event=0,
        ts_init=0,
        margin_init=Decimal(str(margin_pct / 100)),
        margin_maint=Decimal(str(margin_pct / 100)),
        maker_fee=Decimal(str(open_bps / 10000)),         # 开仓/平昨: 0.23‱
        taker_fee=Decimal(str(close_today_bps / 10000)),  # 平今: 2.3‱ (×10)
    )


# ═══════════════════════════════════════════════════════════════
# 6. 回测主入口
# ═══════════════════════════════════════════════════════════════

from concurrent.futures import ThreadPoolExecutor, as_completed

def _load_day_task(day_ordinal, daily_active, epoch_date):
    """多线程 Worker 函数：加载某一天的快照"""
    d_date = epoch_date.fromordinal(day_ordinal)
    date_str = d_date.strftime("%Y%m%d")
    
    day_futures = set()
    for (inst, d), fc in daily_active.items():
        if d == day_ordinal:
            day_futures.add(fc)
            
    if not day_futures:
        return date_str, None
        
    snaps = load_snapshots_for_date(day_futures, date_str)
    return date_str, snaps

def run_nt_backtest_multi_day(signals_df: pd.DataFrame):
    minute_idx, daily_active = build_signal_index(signals_df)
    print(f"📋 信号索引: {len(minute_idx)} 分钟级 + {len(daily_active)} 日级")

    # 提取所有需要的合约
    all_futures = set(daily_active.values()) | {s["futures_code"] for s in minute_idx.values()}
    # daily_active key = (inst, day_ordinal) — 提取所有唯一 day_ordinal
    oos_ordinals = sorted(set(d for (_, d) in daily_active.keys()))
    # oos_ordinals = oos_ordinals[:4]
    # 转回 date 对象用于打印
    from datetime import date as _date
    _epoch = _date(1, 1, 1)
    oos_dates_list = [_epoch.fromordinal(d) for d in oos_ordinals]
    print(f"📅 OOS: {len(oos_ordinals)} 天, {len(all_futures)} 合约")
    print(f"   范围: {oos_dates_list[0]} → {oos_dates_list[-1]}")

    # ── 构建引擎 ──
    config = BacktestEngineConfig(trader_id="BACKTEST-001")
    engine = BacktestEngine(config=config)
    engine.add_venue(
        venue=Venue("CFFEX"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money.from_str(f"{TOTAL_CAPITAL} CNY")],
        base_currency=Currency.from_str("CNY"),
        book_type=BookType.L2_MBP,
        fee_model=CffexIndexFeeModel(cache=engine.cache),
    )

    # ── 预注册所有合约 ──
    registered = {}
    for fc in sorted(all_futures):
        instrument = _make_instrument(fc)
        engine.add_instrument(instrument)
        registered[fc] = f"{fc}.CFFEX"

        # ── 多线程并行加载快照 ──
    total_snaps = 0
    max_workers = min(os.cpu_count() or 4, 8)  # 最多 8 线程
    print(f"\n🚀 启动多线程加载 (Workers: {max_workers})...")
    
    futures_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for day_ordinal in oos_ordinals:
            fut = executor.submit(_load_day_task, day_ordinal, daily_active, _epoch)
            futures_dict[fut] = day_ordinal
            
        date_results = []
        for fut in as_completed(futures_dict):
            date_str, snaps = fut.result()
            if snaps is not None:
                date_results.append((date_str, snaps))

    # NautilusTrader 要求 add_data 的数据必须按时间严格递增
    # 所以必须对多线程乱序返回的结果，按日期重新排序！
    date_results.sort(key=lambda x: x[0])
    
    print("\n💿 正在将排序后的数据注入引擎...")
    for date_str, snaps in date_results:
        for inst_str, snap_list in snaps.items():
            engine.add_data(snap_list)
            total_snaps += len(snap_list)
            
    print(f"\n📊 总计 {total_snaps:,} 快照, {len(registered)} 注册合约")

    # ── 构造策略 ──
    all_inst_strs = list(registered.values())  # ["IC2512.CFFEX", "IC2603.CFFEX", ...]

    strategy = IndexIntradayStrategy(
        minute_idx, daily_active,
        all_instruments=all_inst_strs,
        mapping_type=MAPPING_TYPE,             
        config=StrategyConfig(
            strategy_id="IndexIntraday-001",
            order_id_tag="001",
            oms_type=OmsType.NETTING,
            manage_gtd_expiry=True,
        ))
    engine.add_strategy(strategy)

    # ── 运行 ──
    print(f"\n🔧 运行 {len(oos_ordinals)} 天连续回测 ...")
    t0 = pd.Timestamp.now()
    engine.run()
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    print(f"✅ 回测完成 ({elapsed:.0f}s)")

    _print_multi_day_performance(engine, strategy, registered)
    return engine


# ═══════════════════════════════════════════════════════════════
# 7. 业绩打印
# ═══════════════════════════════════════════════════════════════

def _print_performance(engine: BacktestEngine, strategy: IndexIntradayStrategy,
                       registered_insts: list[str]):
    print("\n" + "=" * 80)
    print("📊 回测业绩摘要")
    print("=" * 80)

    # 查询每个品种的期末持仓
    for inst_str in sorted(registered_insts):
        inst_id = InstrumentId.from_str(inst_str)
        positions = strategy.cache.positions_open(
            instrument_id=inst_id,
            strategy_id=strategy.id,
        )
        if positions:
            pos = positions[0]
            qty = int(pos.quantity.as_double())
            avg_px = float(pos.avg_px_open)
            prefix = strategy._prefixes.get(inst_str, "")
            m, _ = INSTRUMENT_SPECS.get(prefix, (1, 0.01))
            mv = abs(qty) * avg_px * m
            print(f"  {inst_str}: {qty:+d} 手 @ {avg_px:.2f} (市值 {mv:,.0f})")
        else:
            print(f"  {inst_str}: 无持仓")

    # 账户信息（通过 engine 的 trader account(s) 获取）
    accounts = strategy.cache.accounts()
    for acc in accounts:
        print(f"\n账户:{str(acc.id)}")
        print(f"  余额: {[m.to_formatted_str() for m in acc.balances_total().values()]}")

    # 订单/成交统计
    orders = strategy.cache.orders(strategy_id=strategy.id)
    filled = [o for o in orders if o.status == OrderStatus.FILLED]
    expired = [o for o in orders if o.status == OrderStatus.EXPIRED]
    canceled = [o for o in orders if o.status == OrderStatus.CANCELED]
    print(f"\n订单统计:")
    print(f"  总订单: {len(orders)}")
    print(f"  成交: {len(filled)} | 过期: {len(expired)} | 撤单: {len(canceled)}")
    if filled:
        print(f"  成交率: {len(filled) / max(len(filled)+len(expired)+len(canceled), 1) * 100:.1f}%")

def _print_multi_day_performance(engine, strategy, registered):
    """多日版业绩打印。用 registered dict {futures_code: inst_str} 反推 prefix。"""
    inst_prefix = {}
    for fc, inst_str in registered.items():
        prefix_match = re.match(r"^([A-Za-z]+)", fc)
        if prefix_match:
            inst_prefix[inst_str] = prefix_match.group(1).upper()

    print("\n" + "=" * 80)
    print("📊 回测业绩摘要")
    print("=" * 80)
    for inst_str in sorted(registered.values()):
        inst_id = InstrumentId.from_str(inst_str)
        positions = strategy.cache.positions_open(
            instrument_id=inst_id, strategy_id=strategy.id)
        if positions:
            pos = positions[0]
            qty = int(pos.quantity.as_double())
            avg_px = float(pos.avg_px_open)
            prefix = inst_prefix.get(inst_str, "")
            m, _ = INSTRUMENT_SPECS.get(prefix, (1, 0.01))
            mv = abs(qty) * avg_px * m
            print(f"  {inst_str}: {qty:+d} 手 @ {avg_px:.2f} (市值 {mv:,.0f})")
        else:
            print(f"  {inst_str}: 无持仓")

    accounts = strategy.cache.accounts()
    for acc in accounts:
        print(f"\n账户:{str(acc.id)}")
        balances = [m.to_formatted_str() for m in acc.balances_total().values()]
        print(f"  余额: {balances}")

    orders = strategy.cache.orders(strategy_id=strategy.id)
    filled = [o for o in orders if o.status == OrderStatus.FILLED]
    expired = [o for o in orders if o.status == OrderStatus.EXPIRED]
    canceled = [o for o in orders if o.status == OrderStatus.CANCELED]
    print(f"\n订单统计:")
    print(f"  总订单: {len(orders)}")
    print(f"  成交: {len(filled)} | 过期: {len(expired)} | 撤单: {len(canceled)}")
    if filled:
        total = len(filled) + len(expired) + len(canceled)
        print(f"  成交率: {len(filled) / max(total, 1) * 100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 8. 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    train_only = "--train-only" in sys.argv
    backtest_only = "--backtest-only" in sys.argv
    snapshot_date = "20251231"
    for a in sys.argv:
        if a.startswith("--date="):
            snapshot_date = a.split("=", 1)[1]

    # ═══ 阶段 1: 训练 ═══
    if not backtest_only:
        print("=" * 80)
        print("阶段 1: 训练 LightGBM & 生成 OOS 信号")
        print("=" * 80)
        signals_df = train_and_generate_signals()
    else:
        print("📂 加载已有信号文件 ...")
        signals_df = pd.read_parquet(SIGNALS_OUT)
        print(f"   加载: {len(signals_df):,} 行")

    # ═══ 阶段 2: NautilusTrader 回测 ═══
    if not train_only:
        print("\n" + "=" * 80)
        print("阶段 2: NautilusTrader 回测")
        print("=" * 80)
        run_nt_backtest_multi_day(signals_df)

    print("\n✅ 全部完成")