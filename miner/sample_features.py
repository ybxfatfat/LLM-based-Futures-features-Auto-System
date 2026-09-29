"""
sample_features.py - Curated LLM-Generated High-Frequency Features
===========================================================================
This file contains a hand-picked subset of the best performing, highly interpretable 
micro-structure features automatically discovered by the LLM-GA Pipeline.

Selection Criteria for these examples:
  - Built heavily upon Custom Base Fields (e.g., BuyRatio, MaxConsecUpAmp, ExpOBI)
  - Strong Economic Intuition (Clear Counterparty & Mechanism)
  - Strict Mathematical Elegance (No deep nesting, no curve-fitted magic constants)
  - Robust OOS Metrics (Full Year IC > 0.03, Ann IR > 2.0)
"""

# ======================================================================
# LLM-MINED ALPHA DICTIONARY (Ready for C++ DAG Engine execution)
# ======================================================================
FACTOR_FORMULAS = {

    # 🌟 1. 纯粹主动扫单意愿的平滑动量 (Aggressive Sweep Momentum)
    # 🎯 使用基建: BuyRatio (主动买单成交占比)
    # 💡 经济学逻辑: 将主动买卖比例 (BuyRatio) 对成交量进行线性回归剥离残差 (TsFacNeutralRes)，
    # 提取出完全独立于放量/缩量的“纯粹主动扫单意愿”。Z-Score 标准化后，使用快速 10 分钟 EWMA 平滑微观噪音，
    # 捕捉机构隐蔽扫单的真实动量。
    # 📊 Metrics: Mean IC = -0.04020 | Full Year IC = -0.04065 | Ann IR = -2.260
    "aggressive_sweep_ewma_10": "EWMA(TsZscore(TsFacNeutralRes(BuyRatio, volume / (TsMean(volume, 60) + 0.00000001), 60), 60), 10)",

    # 🌟 2. 连续扫盘极值的“高地驻留”密度 (Sweep Plateau Density)
    # 🎯 使用基建: MaxConsecUpAmp (连续上涨游程的极限振幅)
    # 💡 经济学逻辑: 针对你底层算出的游程振幅，计算它在 60 分钟窗口内的分位数，并检查它是否“停留在前 10% 的高位 (0.9)”。
    # 然后用 TsMean 统计过去 15 分钟内这种“高地驻留”的时间占比。这完美过滤了单次瞬间的虚假拉升，专门捕捉持续性的爆破买盘。
    # 📊 Metrics: Mean IC = 0.04907 | Full Year IC = 0.05073 | Ann IR = 2.558
    "sweep_plateau_zscore_smooth": "EWMA(TsZscore(TsMean(MoreThan(TsRank(MaxConsecUpAmp, 60), 0.9), 15), 120), 30)",

    # 🌟 3. 盘口失衡的偏度突破 (OBI Skewness Breakout Plateau)
    # 🎯 使用基建: ExpOBI (加权盘口深度失衡期望)
    # 💡 经济学逻辑: 将 OBI 乘以其自身的偏度 (TsSkewness)。如果盘口失衡本身存在极强的非对称偏移，
    # 说明挂单撤单极度活跃（如诱多/诱空）。系统计算该突破指标的 60 分钟 Rank，并提取“停留在 80% 分位以上的时间比例”。
    # 隔离出真实的、持续的欺骗性挂单墙，而非一触即散的假墙。
    # 📊 Metrics: Mean IC = 0.03596 | Full Year IC = 0.03937 | Ann IR = 3.297
    "obi_skew_breakout_plateau": "TsMean(IndicatorMoreThanZero(TsRank(TsMean(ExpOBI, 60) * TsSkewness(ExpOBI, 60), 60) - 0.8), 30)",

    # 🌟 4. 微观盘口差价压力的 MACD (Micro Spread Pressure EWMAC)
    # 🎯 使用基建: ask1 - bid1 (买一卖一微观价差)
    # 💡 经济学逻辑: 剥离成交量影响后的纯粹“价差拉宽”(Liquidity Vacuum)。系统大模型极其聪明地运用了
    # EWMAC(5, 20) 的快慢线交叉，计算出流动性真空的“加速度”，并通过 20 分钟的线性衰减 (TsDecay) 
    # 赋予近期突然发生的盘口崩溃以极高的权重，捕捉极致的流动性枯竭反转。
    # 📊 Metrics: Mean IC = 0.03970 | Full Year IC = 0.03904 | Ann IR = 2.941
    "idio_spread_press_decay": "TsDecay(TsRank(EWMA(TsZscore(TsFacNeutralRes(ask1 - bid1, volume / (TsMean(volume, 30) + 0.00000001), 30), 30), 5) - EWMA(TsZscore(TsFacNeutralRes(ask1 - bid1, volume / (TsMean(volume, 30) + 0.00000001), 30), 30), 20), 60), 20)",

    # 🌟 5. 扫单比例偏度爆发的衰减记忆 (BuyRatio Skew Breakout Decay)
    # 🎯 使用基建: BuyRatio (主动买单成交占比)
    # 💡 经济学逻辑: 大模型发现直接用主动买卖比例不足以描述极值。它将 BuyRatio 的均值与其偏度 (Skewness) 相乘，
    # 定位极端的买盘爆发区间。由于这种微观突破极易回撤，模型使用绝对值的 Z-score 量化其强度，并果断施加 
    # 15 分钟的线性衰减 (TsDecay)，确保这种瞬间的微观结构崩溃能迅速反映在收益预测上。
    # 📊 Metrics: Mean IC = -0.03449 | Full Year IC = -0.03383 | Ann IR = -4.295
    "skew_breakout_decay_smooth": "TsDecay(Abs(TsZscore(TsMean(BuyRatio, 20) * TsSkewness(BuyRatio, 20), 60)), 15)"
}