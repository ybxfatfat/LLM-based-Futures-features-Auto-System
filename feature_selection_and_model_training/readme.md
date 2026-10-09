## 🔍 High-Resolution Feature Evaluation & Mapping / 高分辨率特征评价与仓位映射

### 1. Vectorized Feature Quality Assessment / 向量化特征质量评估

`feature_quality.py` is a local multiprocessed diagnostics engine for the **2024 in-sample research period**. It compares generated features through several complementary statistics; it does not, by itself, establish out-of-sample predictive power. A genuine OOS claim requires a later period that was not used for feature generation, direction selection, threshold tuning, or model validation.

`feature_quality.py` 是面向 **2024 年样本内研究期**的本地多进程诊断引擎。它通过多组互补指标比较生成特征，但这些结果本身不构成样本外预测力证明。真正的 OOS 结论需要使用一个未参与特征生成、方向选择、阈值调优或模型验证的后续时期。

Key diagnostics include (核心诊断指标包括):

*   **Period IC (2000-minute window)**: Computes IC over non-overlapping 2000-minute windows so that the evaluation unit is consistent across instruments with different daily session lengths. (使用 2000 分钟不重叠窗口计算 IC，使不同日内交易时长的品种采用一致的评价尺度)。
*   **Decile Return Spreads (Q10 - Q1)**: Buckets a feature into 10 quantiles to inspect monotonicity and the long-short return spread in addition to linear correlation. (将特征分为 10 组，在相关系数之外检查单调性与多空收益差)。
*   **Vectorized Simulation Metrics (向量化简易回测指标)**: The NumPy-based diagnostic simulation reports annualized return in bps, Sharpe ratio, and turnover under the script's stated assumptions. (基于 NumPy 的诊断性模拟在脚本所列假设下输出年化收益 bps、夏普比率与换手率)。

### 2. AlphaMiner Pro Dashboard / 交互式特征分析看板

`dashboard_app.py` provides a Streamlit interface for reviewing the metrics exported by `feature_quality.py`. / `dashboard_app.py` 使用 Streamlit 展示 `feature_quality.py` 导出的评测结果。

*   **Feature-Centric View (单特征下钻视图)**: Shows decile-return charts, cross-instrument IC distributions, and H1-versus-H2 distribution shifts for a selected feature. (展示指定特征的分位数收益、跨品种 IC 分布及上下半年分布变化)。
*   **Scope-Centric View (全局横向视图)**: Ranks generated features by selected metrics within the chosen commodity or index pool and supports threshold filtering. (在选定的商品或指数品种池内按指标比较特征，并支持阈值筛选)。

## 3. Graph-Based Feature Selection / 阶段三：基于图网络的特征正交与分层初筛

Before model training, `select_feature.py` uses a correlation network and hierarchical voting to reduce redundancy and retain representative candidates. / 在模型训练前，`select_feature.py` 使用相关性网络与分层投票降低冗余，并保留具有代表性的候选特征。

*   **Hierarchical Community Discovery & Voting (社区发现与分层投票筛选)**:
    1. **Single-Instrument Pre-selection**: Within each highly correlated community found by Greedy Modularity, the system selects a representative for each instrument from historical in-sample metrics. (在 Greedy Modularity 识别出的高相关社区内，依据历史样本内指标为各品种选择代表特征)。
    2. **Target Pool Coverage Retention**: A community is retained only when its candidates cover a configured share of the target pool, such as 20%. (只有当候选特征覆盖达到设定比例，例如目标品种池的 20%，对应社区才会被保留)。
    3. **Intra-Community Voting & Tie-breaking**: Instrument-level votes determine the representative; ties are resolved with pooled IC under the configured sample mask. (使用品种级投票选择社区代表，平票时以既定样本掩码下的 Pooled IC 决胜)。
*   **Tradability Masking (可交易样本屏蔽)**: The pooled calculation excludes configured end-of-day, illiquid-night-session, and rollover samples to reduce contamination from periods outside the intended trading scope. (混池计算按配置剔除尾盘、低流动性夜盘与换月样本，以减少非目标交易时段造成的污染)。

### 4. Position Mapping Logic in Tree Models / 模型预测值的仓位映射

In quantitative trading, raw model predictions must be mapped to target positions in `[-1.0, 1.0]`. `tree_models.py` implements a 2×2 family of mapping choices so that the assumptions about reference history and scaling are explicit. / 在量化交易中，原始模型预测值需要映射为 `[-1.0, 1.0]` 范围内的目标仓位。`tree_models.py` 实现了一组 2×2 映射选择，使时间基准与数值缩放假设保持显式可审查。

#### Dimension 1: Reference Timeframe / 维度一：时间基准

*   **Real-time Rolling Windows (实时长窗口滚动)**: Calculates distribution statistics from a trailing historical window to react to recent regime changes. (从滚动历史窗口计算分布统计量，以响应近期市场状态变化)。
*   **In-Sample Static Anchoring (样本内静态锚定)**: Estimates mean, standard deviation, or quantiles from the training sample and freezes those constants for later application. This avoids using future observations in the mapping statistics, provided the training boundary is respected. (从训练样本估计均值、标准差或分位数，并将这些常数冻结后用于后续时期；只要严格遵守训练边界，映射统计量就不会读取未来观测)。

#### Dimension 2: Scaling Mathematics / 维度二：数学缩放

*   **Z-Score Mapping (Z-Score 映射)**:
    *   *Demeaned*: Centers the signal before scaling to emphasize relative strength. (先减均值再缩放，强调相对强弱)。
    *   *No-Demean*: Scales without centering to preserve the directional level of the raw prediction. (不减均值，仅进行尺度调整，以保留原始预测的方向水平)。
*   **Robust Extremum / Quantile Lock (稳健极值与分位数锁定)**: Uses IQR-based bounds or fixed tail quantiles instead of relying only on standard deviation, reducing the influence of extreme observations and concentrating exposure in configured high-conviction regions. (使用 IQR 边界或固定尾部分位数，而非只依赖标准差，从而降低极端观测的影响，并将仓位集中在设定的高确信度区域)。
