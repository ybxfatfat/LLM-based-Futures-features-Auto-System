## 🔍 High-Resolution Feature Evaluation & Mapping / 高分辨率特征评价与仓位映射

### 1. Vectorized Vectorized Feature Quality Assessment (向量化特征质量评估)
To strictly prevent look-ahead bias and accurately measure the out-of-sample (OOS) predictive power of the generated features, we designed a highly optimized, multiprocessed evaluation engine (`feature_quality_2.py`[cite: 17]).
为了严格防止未来函数泄露并准确衡量生成特征的样本外 (OOS) 预测能力，我们设计了一个高度优化的多进程评估引擎。

Key metrics computed include (核心计算指标包括):
*   **Period IC (2000-minute window)**: Replaced the traditional Daily IC to perfectly align with the 2000-minute non-overlapping evaluation window used in the core backtester. This ensures statistical robustness regardless of daily trading hours[cite: 17]. (将传统的 Daily IC 替换为 **Period IC**，完美对齐核心回测器中使用的 2000 分钟无重叠评价窗口，确保跨品种评估的统计学稳健性)。
*   **Decile Return Spreads (Q10 - Q1)**: Beyond linear correlation (IC), the engine automatically buckets features into 10 quantiles to track monotonicity and long-short return spreads[cite: 17]. (超越线性相关性，自动将特征分为 10 组，追踪多空收益差与单调性)。
*   **Vectorized Simulation Metrics (向量化简易回测指标)**: The script runs a lightning-fast NumPy-based vectorized backtest computing:
    *   **Annualized Return (bps)** (年化收益率 bps)[cite: 17]
    *   **Sharpe Ratio** (夏普比率)[cite: 17]
    *   **Turnover Rate** (换手率)[cite: 17]

### 2. AlphaMiner Pro Dashboard (交互式特征分析看板)
To make sense of the massive evaluation metrics across hundreds of instruments, we built an interactive Streamlit dashboard (`dashboard_app_2.py`[cite: 16]).
为了直观地分析数百个品种的海量评估指标，我们构建了一个基于 Streamlit 的交互式数据看板。

*   **Feature-Centric View (单特征下钻视图)**: Deep dive into a specific feature's performance, displaying Decile Return bar charts, Cross-Instrument Stability (IC distribution across different futures), and Half-Year (H1 vs H2) distributional shifts[cite: 16]. (深入剖析特定特征的表现，展示分位数收益柱状图、跨品种稳定性以及上下半年分布漂移)。
*   **Scope-Centric View (全局横向排行榜)**: Rank all generated features based on specific metrics (e.g., Period IC, Decile Spreads) across the entire commodity or index pool, complete with threshold filtering[cite: 16]. (在全市场池子中，基于特定指标对所有生成的特征进行横向排行与阈值筛选)。

## 3: Graph-Based Feature Selection / 阶段三：基于图网络的特征正交与分层初筛

Before feeding features into complex machine learning models, the system employs rigorous network theory and a hierarchical voting mechanism to deduplicate and extract orthogonal Alphas (`select_feature_2.py`). / 在进入树模型前，利用图网络与分层投票机制进行严格的因子正交去重与可用性初筛。

*   **Hierarchical Community Discovery & Voting (社区发现与分层投票筛选)**:
    1. **Single-Instrument Pre-selection**: Within each highly-collinear feature community (extracted via Greedy Modularity), the system pre-selects the best performing feature for *each individual instrument* based on historical metrics. (单品种社区内预选：在每个高内聚的共线性特征社区内，首先针对单个品种，依据历史指标挑选出最优特征)。
    2. **Target Pool Coverage Retention**: A community is only retained if it provides valid Alpha for a sufficient ratio (e.g., 20%) of the entire target pool. (全池覆盖率保留：如果某个社区选出的特征无法覆盖目标品种池中足够比例的品种，该特征社区将被整体淘汰)。
    3. **Intra-Community Voting & Tie-breaking**: Features receive "votes" from different instruments within the community. Ties are broken using a globally computed **Pooled IC**, ensuring the final selected representative is the most universally robust Alpha across the entire asset universe. (社区内投票与终选比较：统计社区内各特征获得的“品种选票”。平局时，利用全市场严格洗膜后的 **Pooled IC** 进行绝对值对比决出胜者，确保每个正交社区只保留唯一的、普适性最强的超级特征)。
*   **Strict Tradability Masking (严苛的可交易池屏蔽)**: Evaluates Pooled IC strictly on clean, tradable samples—masking out End-of-Day (EOD) closures, illiquid night sessions, and contract rollover periods, preventing illusory performance. (在计算混池 Pooled IC 时，强制施加极其严格的时空掩码——精准剔除尾盘、长假前夜盘以及换月跳空期，只在“绝对安全可交易”的净样本中验证预测力)。

### 4. Position Mapping Logic in Tree Models (模型预测值的仓位映射)
In quantitative trading, a raw model prediction (e.g., an expected return) must be converted into an actual target position ([-1.0, 1.0]). In `tree_models_2.py`[cite: 21], we implemented a robust "2x2 Orthogonal Matrix" of mapping strategies to handle low signal-to-noise ratio (SNR) environments without data leakage.
在量化交易中，原始的模型预测值必须被转换为实际的目标仓位 ([-1.0, 1.0])。在树模型回测器中，我们实现了一个强大的“2×2 正交矩阵式”映射策略，以在无未来函数泄露的前提下处理低信噪比环境。

#### Dimension 1: The Reference Timeframe (时间基准维度)
*   **Real-time Rolling Windows (实时长窗口滚动)**: Dynamically calculates distribution statistics over a long lookback period (e.g., pooled over N instruments * W minutes). This allows the model to adapt to recent market regime shifts[cite: 21]. (在一个长回看窗口内动态计算分布统计量，允许模型自适应近期的市场状态漂移)。
*   **In-Sample (IS) Static Anchoring (样本内静态锚定)**: Extracts exact distribution statistics (mean, standard deviation, quantiles) directly from the In-Sample (IS) training folds. These constants are strictly carried forward into the Out-of-Sample (OOS) period, preventing dynamic-window-induced turnover and strictly eliminating look-ahead bias[cite: 21]. (直接从样本内训练集中提取确切的分布统计量，并将其作为绝对常数应用到样本外。这种极度严苛的做法有效防止了滚动窗口带来的高换手率)。

#### Dimension 2: The Scaling Mathematics (数学缩放维度)
Both of the timeframes above (Rolling & IS) are tested against two mathematical scaling paradigms[cite: 21] (上述两种时间基准，均分别结合了以下两种数学缩放范式进行测试):

*   **Z-Score Mapping (Z-Score 映射)**: Scales predictions using standard deviation. 
    *   *Demeaned*: Centers the signal by subtracting the mean to capture relative momentum[cite: 21].
    *   *No-Demean*: Only scales by volatility, preserving the absolute directional drift of the raw prediction[cite: 21].
    (利用标准差进行缩放。**减均值版**捕捉相对动量；**不减均值版**保留原始预测的绝对方向漂移)。
*   **Robust Extremum / Quantile Lock (稳健极值与分位数锁定)**: Utilizes the IQR (Interquartile Range) or fixed hard-quantiles (e.g., Top 10% and Bottom 10%) instead of standard deviation to scale predictions. This neutralizes the impact of extreme outliers and restricts trading strictly to high-conviction zones[cite: 21]. (使用 IQR 或固定的硬分位数替代标准差进行缩放或锁仓触发，完美免疫极端异常值的干扰，并将交易严格限制在高确信度区间)。