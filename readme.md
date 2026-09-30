# 🧬 LLM-GA & High-Frequency Futures Alpha Engine / 大模型高频微观因子挖掘与极速计算系统

**[EN]** This project is a high/mid-frequency quantitative feature mining and execution pipeline that processes raw snapshot data into trading signals ready for simulated backtesting. The system features a **dual-phase, dual-engine architecture**: The bottom layer utilizes a C++ Symbolic Tracing computation engine alongside a RAG-based agent to design complex dimensionality reduction algorithms, transforming 500ms snapshots into micro-features (base fields); The upper layer (further computation at the minute frequency) leverages a lightning-fast GPU compilation engine combined with a multi-round 5-layer genetic evolution LLM to mine deep Alpha. Finally, the pipeline is equipped with hierarchical feature redundancy removal based on community networks tailored for specific target instrument pools, professional feature evaluation dashboards, model training for signal generation and target position mapping (showcasing tree models only), and a NautilusTrader simulation backtesting environment.

**[CN]** 本项目是一个从原始快照数据到可以通过模拟交易系统回测的交易信号的高/中频量化特征挖掘与执行流水线。系统具备**两阶段双引擎架构**：底层利用 C++ 符号追踪计算引擎与 RAG based 智能体，将 500ms 快照设计复杂的降维算法得到微观特征（基础字段）；上层(分钟频上的进一步计算)利用 GPU 极速编译引擎与多轮运行的 5 层遗传演化大模型，挖掘深层 Alpha。最后搭载了基于社区网络的分层特征去冗余等针对特定目标品种池的筛选、专业的特征评价看板，模型训练产出信号并映射成目标仓位（仅展示树模型）以及 NautilusTrader 仿真回测环境。

> ⚠️ **Disclaimer / 特别声明**
> * **[EN]** This repository serves strictly as a replication and showcase of past quantitative research methodologies. It is **not** plug-and-play out of the box due to strict dependencies on proprietary institutional data interfaces. Furthermore, certain core proprietary libraries (e.g., specific foundation templates in `base_features`) have been stripped or reduced to bare minimums for demonstration purposes, and as a replication effort, some useful technical nuances and optimizations from the original system have not yet been fully added in.
> * **[CN]** 本仓库仅作为过往量化研究经验与系统架构设计的复现与展示。由于高度依赖特定的底层数据接口，本项目**并非开箱即用**。同时，部分核心底层依赖（如 `base_features` 中的具体基础因子库）已被剥离或仅保留最小化示例，且由于是复现工程，部分曾实现过的实用细节技巧尚未加入其中。
> 

> 📚 **Documentation Note / 文档指引**
> * **[EN]** For more rich and detailed technical information regarding specific components, please note that **each subfolder in this repository contains its own dedicated and comprehensive `README.md`**.
> * **[CN]** 为了提供更加丰富、硬核的技术细节，**本仓库的每一个子文件夹下都配有专属且详细的 `README.md` 说明文档**，欢迎深入各个模块查阅。

---

## 🔬 Phase 1: Microstructure Aggregation (500ms ➔ 1-Min) / 阶段一：微观快照降频系统

### 1.1 C++ Symbolic Tracing Engine / C++ 零拷贝符号追踪底座
Designed to process high/mid-frequency Level-2 snapshot data into minute-level aggregated features. / 专为处理 500ms L2 快照设计的高频聚合底座。

*   **Zero-Copy Memory & C++ Native Execution**: Massive Tick data is loaded via Python (NumPy), but PyBind11 simply passes raw memory pointers to the C++ engine. Iterations bypass the Python GIL completely. (Python 仅做“指挥官”，C++ 闭环遍历数百万行 Tick，绕过 GIL 锁实现绝对零拷贝)。
*   **Symbolic Tracing Graph**: Python acts as a blueprint drawer (`ce.IfElse`, `ce.ConstNode`). The C++ engine dynamically allocates memory and processes the data pipeline based on this DAG. (Python 端仅作“画图连线”，底层 C++ 引擎据此动态构建状态算子流水线)。
*   **Atomic to Complex (原子算子构建复杂逻辑)**: Although the engine relies on atomic operators, these primitives can be flexibly combined to construct highly complex conditional judgments and stateful logic (e.g., using `TsResetable` for conditional run-length computations). (虽然底层调用的是原子化算子，但通过灵活组合，已足以构造出极度复杂的条件判断与状态机逻辑，如带条件打断的连涨/连跌游程计算)。

#### 💡 Research Rationale & Microstructure Nuances / 研究考量与微观处理:
*   **Missing Ticks Philosophy (缺失快照的处理)**: A minute theoretically has 120 snapshots (at 500ms), but missing ticks are common. While a lack of updates usually implies an unchanged order book—making forward-filling a standard industry practice—**we explicitly reject this naive approach**. For high-frequency aggregation, padding the window with redundant static snapshots artificially inflates the sample size and severely distorts dynamic statistical metrics like volatility and variance. Instead, we strictly preserve data sparsity and maintain a true `tick_count` to dynamically adjust the aggregation denominator `n`, ensuring pristine samples. (1分钟理论上有120个快照，少量缺失现象很常见。虽然无推送通常意味着盘口无变动，这也是许多常规处理选择“前向填充 / Forward-fill”的逻辑前提，**但我们坚决摒弃了这种做法**。因为对于高频特征聚合而言，强行填充静态快照在缺失较多时会人为制造大量冗余样本，严重扭曲波动率、方差等动态指标的真实性。我们在降频时严格保留了数据的稀疏性，仅通过维护真实的 `tick_count` 来对分母 `n` 进行动态矫正，确保样本绝对纯净)。
*   **Small Sample & Opening Bar Protection (小样本与开盘保护)**: At market open or during extreme illiquidity, a minute might only receive a single auction tick or highly sparse updates. We implement a strict valid tick count threshold (e.g., >20 valid ticks). If not met, the feature is safely masked as `NaN` to prevent extreme mathematical distortion. (刚开盘时往往只有集合竞价后的单笔 tick，或遭遇极端冷门时段。我们在计算特定字段时设置了“有效快照数”门槛。样本量不足时强制置为 `NaN`，防止单一样本引发极端失真的离群值)。
*   **Limit Moves & Zero-Division (涨跌停与防零除保护)**: Modern data feeds often continue pushing limit up/down states without defaulting sides to 0 or NaN. Thus, no artificial truncation is applied during limit moves. However, strict zero-division protection via `ce.IfElse` is enforced at the compiler level across all dynamic denominators, guaranteeing execution safety. (现代快照数据源在涨跌停时通常会继续推送盘口状态，因此特征计算时不做人工的涨跌停截断。但我们在底层强制实施了严格的防零除保护，确保 C++ 引擎在任何极端挂单真空状态下都不会崩溃)。

### 1.2 Autonomous RAG Agent / RAG 微观逻辑智能挖掘体
Bridges the gap between natural language economic intuition and high/mid-frequency C++ execution. / 打通“自然语言微观直觉”到“极高频 C++ 算子落地”的次元壁。

*   **Economic Intuition (逻辑三问)**: Forces the LLM to articulate: *What drives returns? Who is the loser? Why does it persist?* before generating any code. (强制大模型在写代码前回答：赚谁的钱？对手盘为何亏损？异象为何持续？确保逻辑先于代码)。
*   **Dual-Payload Semantic Retrieval (双载荷语义检索)**: Queries the local RAG Vector Database using only the *economic mechanism* as the embedding vector, while storing the actual C++ code and backtest metrics as payload metadata. (仅对自然语言假设进行向量化 Embedding，而将代码和历史评测指标作为元数据载荷。完美规避代码语法对语义检索的干扰，实现基于“经济学原理”的精准查重)。
*   **Closed-Loop Reflection Database (闭环自进化反思库)**: Post-evaluation, the system feeds the rigorous performance metrics (e.g., Period IC, Q10-Q1 spread) back to the LLM to generate a diagnostic reflection. This "post-mortem" is injected back into the RAG database. (在特征评测后，系统会将指标喂给大模型进行“验尸反思”，并将反思报告打入 RAG 经验库，形成越挖越聪明的自我进化闭环)。

#### 💡 Research Rationale / 研究考量:
*   **Mitigating LLM Hallucination via Institutional Memory (通过“机构级记忆”抑制大模型幻觉)**: Left unconstrained, an LLM acts like a naive beginner, repeatedly generating curve-fitted illusions or basic moving-average crossovers that fail in noisy live markets. By forcing the LLM to consult a persistent RAG database containing hard historical backtest metrics, we endow the Agent with a "Quant's Grudge Book" (错题本), fundamentally preventing it from stepping into the same microstructure noise traps twice. (没有实盘记忆的大模型会无限重复生成在真实市场中必亏的简单线性逻辑或严重过拟合的废代码。通过强制大模型查阅带有真实回测指标的 RAG 经验库，我们实质上给了它一本“量化错题本”，避免它在同一个微观结构噪音的坑里踩两次，从而极大地提高了高频因子的信噪比)。

### 1.3 Core Base Fields: Active Buy/Sell Decomposition / 核心底层数据 (Base Fields)：主动买卖意愿精细分解
**[EN]** Highlight: The **Active Buy/Sell Decomposition** base fields. Instead of naively relying on order book volumes (which are easily spoofed), the system derives fundamental physics of the order book by computing the instantaneous Tick VWAP and comparing it against the top-of-book quotes and Mid-Price. This non-linear slicing cleanly separates pure aggressive sweeps from passive matching, serving as a highly reliable foundational feature.
**[CN]** 成果展示：系统内置了顶级的**主动买卖意愿分解基础字段 (Base Fields)**。摒弃了容易被撤单操纵的原始盘口挂单量，精准计算瞬间 Tick VWAP，并将其与买一/卖一价及中间价的偏离度进行非线性切割。极度精准地剥离出“纯粹主动扫单”与“被动成交”，真实还原主力机构的微观方向性意图，作为后续演化最坚实的物理底座。

---

## 🚀 Phase 2: Alpha Evolution (1-Min ➔ Deep Alpha) / 阶段二：分钟频高阶演化系统

### 2.1 GPU-Accelerated DAG Compiler / GPU 常驻显存编译引擎
An ultra-low latency, highly scalable GPU subsystem designed for complex minute-level feature engineering. / 专为分钟频高阶特征组合设计的 GPU 极速算力底座。

*   **AST-Driven Compilation & CSE**: Utilizes a Python AST `NodeTransformer` to parse formulas. Enforces Common Subexpression Elimination (CSE). (基于抽象语法树的 DAG 编译，全局公共子表达式消除，极致压缩计算图)。
*   **Zero-IO Formula Chunking**: Entire panels are pinned in GPU VRAM (Pin Memory). Formulas are dispatched in batches, transforming the GPU into a pure execution state. (常驻显存的列式切分，彻底消灭 CPU-GPU 间的 PCIe 数据搬运损耗)。

<p align="center">
  <img src="./miner/figure.png" alt="LLM-GA Architecture" width="100%">
</p>

### 2.2 The 5-Layer Genetic Pipeline / 5层定向基因突变架构
Instead of blind formula generation, the LLM progresses through a strictly scaffolded pipeline. The primary objective of this system is to proactively search for **"Strong Features" (Factors)** that can be directly consumed by tree-based models, while strictly capping mathematical complexity to prevent overfitting.
系统禁止大模型一口气写完逻辑，将其拆解为 5 道靶向流水线。本系统的核心目标是主动定向搜索**“强特征（因子）”**以供树模型直接使用，同时严格限制公式的数学复杂度以防过拟合。

*   **[L1] Logical Genesis (基础逻辑构建层)**: Goes beyond mere template injection. It leverages the `Base Features` library to construct foundational, economically meaningful operations. This involves both deep time-series computations on a single base field and rich cross-sectional interactions across multiple base fields to form a solid logical bedrock. (绝非简单的模板注入，而是利用 `Base Features` 库构造具备坚实微观逻辑的初始运算。这不仅包含单一基础字段的复杂时序逻辑计算，更包含多个基础字段间的丰富交互组合，为后续演化奠定最具物理意义的基石)。
*   **[L2] Relativization (截面相对化层)**: Eliminates absolute price/volume scale differences to ensure cross-sectional comparability. The LLM is forced to use economically logical scaling rather than naive mathematical transformations. (强制施加规模剥离，消除跨品种间的绝对价格与流动性差异。系统要求 LLM 必须采用具有实际经济学意义的相对化操作，而非盲目的数学变换，如单纯取 Abs)。
*   **[L3] Monotonicity & Distribution (分布与单调性重塑层)**: Feeds 10-quantile return profiles (mean return per bin) back to the LLM. The goal is not just numerical distribution normalization, but mathematically reshaping the feature to correct U-shaped anomalies and strictly enforce predictive monotonicity. (向 LLM 喂入 10 分组收益率均值图。核心目的不仅仅是调整特征数值本身的分布，更重要的是让 LLM 像人类研究员一样“看图治病”，针对性地修复非线性异象，强行重塑特征与收益率之间的单调性)。
*   **[L4] Temporal Interaction (复杂时序交互层)**: Opens access to the `Compound Operators` library. The LLM is inspired to self-create deep-nested recursive structures (e.g., EWMAC on volatility-adjusted momentum), discovering higher-order market microstructure dynamics. (开放 `Compound Operators` 高阶复合算子库。启发 LLM 在保留前期物理逻辑的前提下，自创深度嵌套的复杂时序计算，挖掘更高阶的微观博弈动态)。
*   **[L5] Smoothing & Sweeping (降噪与参数网格扫荡)**: Enforces exponential smoothing to filter microstructure noise and mandates the generation of multiple time-window variants for grid sweeping to verify the alpha's survival boundaries. (强制施加指数平滑过滤微观噪音，并强行衍生出多个不同数量级的时间窗口进行网格扫荡，以探索该因子在不同周期下的存活边界)。

#### ⚙️ Evolutionary Infrastructure / 演化基建与 E&E 博弈
**[EN]** *Throughout all 5 layers, the LLM is strictly prompted to maintain economic interpretability, rejecting meaningless math operations.* Furthermore, the pipeline is supported by an intricate infrastructure:
1.  **Component Libraries**: The `Base Features` and `Compound Operators` libraries serve as the system's fundamental building blocks.
2.  **Inter-Layer Material Pools (蓄水池)**: Retains successfully mutated genes across layers.
3.  **Lessons Learned Database (经验库)**: Every attempted feature is analyzed by the LLM, logging its mathematical successes or reasons for failure.
The system leverages these facilities to dynamically trade off between **Exploitation** (deep-diving into newly discovered promising branches) and **Exploration** (reviving and mutating dormant, orthogonal ideas from the pool to escape local optima).
**[CN]** *在每一层的演化中，系统都严苛要求 LLM 必须注重公式的物理意义，拒绝无逻辑的纯数学拼接（如简单嵌套函数）。* 此外，支撑这 5 层架构的是一套极其精密的底层设施：作为积木的**基础字段与高阶算子库**、跨层流转的**原料蓄水池**，以及由 LLM 独立分析并沉淀每一次失败与成功原因的**历史经验库 (Lessons Learned)**。系统利用这些设施，在**深度利用 (Exploitation)**（顺藤摸瓜打磨高潜新分支）与**广度探索 (Exploration)**（从蓄水池唤醒沉睡的旧思路以跳出局部最优）之间进行精妙的博弈平衡。

#### 💡 Research Rationale / 研究考量:
*   **Strong Feature vs. Weak Feature Paradigm (强特征 vs 弱特征的量化流派博弈)**: Modern high-frequency research is bifurcated into "Strong Feature + Weak Model" (e.g., Tree models) and "Weak Feature + Strong Model" (e.g., Time-Series Deep Learning). Evaluating the true quality of "weak features" (base fields) using traditional statistical metrics is notoriously difficult. This pipeline explicitly targets the **"Strong Feature"** paradigm. The explicit temporal and smoothing operations generated in Layers 4 and 5 are essentially manual algorithmic extractions of patterns that Time-Series DL models attempt to learn implicitly. By generating these explicitly, we provide ultra-high-quality inputs for Tree Models while avoiding the DL "black-box" trap and preventing severe time-series overfitting. (当前中高频量化研究主要分为“强特征+弱模型（如树模型）”与“弱特征+强模型（如时序深度学习）”两派。传统评价指标很难衡量基础“弱特征”的好坏。本系统坚定选择**“强特征”**路线。事实上，流水线第 4、5 层的时序交互与平滑操作，正是时序深度学习模型在黑盒中试图隐式学习的逻辑。我们将其显式提取出来，直接为树模型提供极高质量的输入，同时警惕时序计算过度复杂化带来的过拟合风险)。
*   **AST Lineage Tracing for Deep Learning (面向深度学习的 AST 血缘溯源展望)**: While this pipeline outputs fully-formed Strong Features for tree models, we recognize their underlying structural value for Deep Learning. An established research methodology from our past experiences involves parsing the Abstract Syntax Tree (AST) lineage of these highly complex features to construct a feature dependency graph. By using LLMs to analyze these lineage graphs, we can identify and extract high-frequency co-occurring "Root Formulas". These pruned, foundational roots inherently possess high Signal-to-Noise Ratios (SNR) and serve as the perfect "Weak Features" for robust Time-Series DL inputs. *(Note: This specific AST graph-extraction module is a conceptual replication of past research and is not fully integrated into the current open-source repository).* (虽然系统当下直接产出强特征，但基于过往研究经验，我们将大模型生成的复杂公式解析为 AST 抽象语法树，绘制特征血缘网络，并借助 LLM 辅助分析找出高频共现的“公共 Root 公式”。这些去芜存菁的基础 Root 具有极高的信噪比，是未来作为时序深度学习模型输入的绝佳“弱特征”材料。这一前瞻性溯源思路构成了系统演进的重要基石，但作为复现工程，相关代码暂未完全整合入本项目中)。

---

## 🕸️ Phase 3: Graph-Based Feature Selection / 阶段三：基于社区网络的特征正交与分层初筛

Before feeding features into complex machine learning models, the system employs rigorous network theory and a hierarchical voting mechanism to deduplicate and extract orthogonal Alphas (`select_feature.py`). / 在进入树模型前，利用社区网络与分层投票机制进行严格的因子正交去重与可用性初筛。

*   **Hierarchical Community Discovery & Voting (社区发现与分层投票筛选)**:
    1. **Single-Instrument Pre-selection**: Within each highly-collinear feature community (extracted via Greedy Modularity), the system pre-selects the best performing feature for *each individual instrument* based on historical metrics. (单品种社区内预选：在每个高内聚的共线性特征社区内，首先针对单个品种，依据历史指标挑选出最优特征)。
    2. **Target Pool Coverage Retention**: A community is only retained if it provides valid Alpha for a sufficient ratio (e.g., 20%) of the entire target pool. (全池覆盖率保留：如果某个社区选出的特征无法覆盖目标品种池中足够比例的品种，该特征社区将被整体淘汰)。
    3. **Intra-Community Voting & Tie-breaking**: Features receive "votes" from different instruments within the community. Ties are broken using a globally computed **Pooled IC**, ensuring the final selected representative is the most universally robust Alpha across the entire asset universe. (社区内投票与终选比较：统计社区内各特征获得的“品种选票”。平局时，利用全市场严格洗膜后的 **Pooled IC** 进行绝对值对比决出胜者，确保每个正交社区只保留唯一的、普适性最强的超级特征)。
*   **Strict Tradability Masking (严苛的可交易池屏蔽)**: Evaluates Pooled IC strictly on clean, tradable samples—masking out End-of-Day (EOD) closures, illiquid night sessions, and contract rollover periods, preventing illusory performance. (在计算混池 Pooled IC 时，强制施加极其严格的时空掩码——精准剔除尾盘、长假前夜盘以及换月跳空期，只在“绝对安全可交易”的净样本中验证预测力)。

---

## 🔍 Phase 4: High-Resolution Evaluation & Dashboard / 阶段四：高分辨率特征评价与看板

To strictly prevent look-ahead bias, we built a robust multiprocessed evaluation suite (`feature_quality.py`). / 为严格防止未来函数泄露，设计了极速多进程评估引擎。

*   **Period IC (2000-minute window)**: Replaced traditional Daily IC to perfectly align with a 2000-minute non-overlapping evaluation window. (采用 2000 分钟无重叠评价窗口，确保跨品种统计稳健性)。
*   **Decile Return Spreads (Q10 - Q1)**: Automatically buckets features into 10 quantiles to track monotonicity. (自动划分为 10 组追踪单调性与多空收益差)。
*   **AlphaMiner Pro Dashboard**: An interactive Streamlit app (`dashboard_app.py`) providing Feature-Centric Drilldowns and Scope-Centric Rankings. (交互式看板，支持单特征下钻与全市场横向阈值排行榜)。

---

## 📈 Phase 5: Position Mapping & Execution / 阶段五：模型预测值的仓位映射与仿真回测

Converting raw model predictions (which typically represent expected returns, especially when trained with RMSE-like loss functions) into actual trading target positions ([-1.0, 1.0]) is a critical final step. `tree_models.py` implements several rigorous position mapping methods rather than a single naive scaling approach.
将模型输出的原始预测值（尤其是在基于 RMSE 类损失函数训练时，预测值自带“预期收益率”的物理意义）转化为实际的交易目标仓位 ([-1.0, 1.0]) 是至关重要的一步。我们在 `tree_models.py` 中实现了多种严谨的仓位映射方法。

#### 1. Position Mapping Methods / 核心仓位映射方法
*   **Timeframe Anchoring (时间基准提取)**:
    *   *Real-time Rolling Windows (实时滚动回看)*: Calculates scaling statistics (mean, std, quantiles) based on a dynamic trailing historical window. (基于一段动态滚动的历史窗口来计算缩放所需的统计量)。
    *   *In-Sample Static Anchoring (样本内静态锚定)*: Extracts exact distribution constants directly from the entire In-Sample (IS) training period and freezes them for Out-of-Sample (OOS) application. (直接从完整的样本内全量数据中提取统计量，并作为绝对常数冻结应用到样本外)。
*   **Scaling Mathematics (数值缩放方式)**:
    *   *Z-Score (Demean vs. No-Demean)*: Standardizes the raw predictions. Demeaning targets cross-sectional relative strength, while No-Demean preserves absolute momentum. (利用标准差对预测值进行 Z-Score 映射。分为减均值与不减均值两种处理)。
    *   *Non-Linear Extremum / Quantile Mapping*: Uses hard percentiles or Interquartile Range (IQR) to map predictions, applying distinct transformations to extreme versus median values. (基于分位数或 IQR 进行非线性映射，对极值区间和中位数区间施加不同的转换)。

#### 💡 Research Rationale & Pitfalls / 研究考量与实盘陷阱:
*   **The "Demean" Dilemma in One-Sided Markets (减均值困境与单边极端行情)**: Models trained to predict returns will naturally output uniformly signed predictions (e.g., all positive) during extreme OOS one-sided bull markets. In such regimes, applying a "Demean" operation artificially forces half of the predictions into negative values, causing the strategy to take disastrous short positions against a strong trend. Choosing between Demean and No-Demean is fundamentally a trade-off between cross-sectional neutrality and absolute directional momentum, requiring holistic adjustments in target engineering and OOS risk management. (在遇到极端的单边行情如持续大涨时，优秀的模型本身会给出大量一致的单边符号预测值。此时如果强制进行 Demean 减均值操作，反而会人为制造出负数仓位导致逆势做空亏损。减均值与否，本质上是截面相对动量与绝对方向收益之间的权衡，需要结合预测目标的修改与整体策略风控来综合解决)。
*   **Distribution Shift Assumptions (应对分布漂移的时间基准博弈)**: The choice between IS Static and Rolling Windows relies on different assumptions about data non-stationarity. IS Static anchoring benefits from a massive, stable sample size, immune to local noise. Conversely, Rolling Windows operate under the assumption that distribution shifts are continuous—meaning the recent historical distribution is the closest proxy to the immediate OOS future. (使用全量样本内信息 (IS) 映射的好处在于样本量极大且稳定；而每次回看一定长度的历史 (Rolling) 则是基于“分布变动是连续的”这一假设，认为离待预测时间越近的历史分布，越接近未来的真实分布)。
*   **Weak Signal Suppression & Non-Linearity (弱信号区的非线性抑制)**: Quantile return analysis reveals that only extreme model predictions (high absolute values) yield statistically significant returns. The mid-range "weak signals" are predominantly noise. Therefore, naive linear mapping is suboptimal. Superior mapping must be non-linear—aggressively squashing weak signals into disproportionately low or absolute zero positions to save transaction friction, while fully capitalizing on high-conviction zones. (分组收益率均值图表明，通常只有绝对数值极大的预测区间，其收益率均值才显著。中间的弱信号区本质上是噪音。因此仓位映射不应是简单的线性缩放，采用非线性的映射方法，让弱信号部分映射出极低仓位甚至是直接归 0 仓位，往往能更有效规避摩擦成本)。

> **Advanced Simulation**: The execution engine feeds these signals into a highly rigorous **Memory-Safe Daily Sandbox NautilusTrader Engine**, handling Dynamic Event (hourglass) execution and Continuous Fixed-Grid add-on behaviors with realistic slippage matching. (最终信号喂入极度严苛的 NautilusTrader 每日独立沙盒，支持动态沙漏与固定网格追加追价，完美缝合日频净值曲线)。

---

## 🛠️ Quick Start / 快速启动指南

### 1. Phase 1: Microstructure Tick Mining (500ms ➔ 1Min)
```bash
# Compile the C++ DAG Engine
./build.sh

# Run the RAG Agent to mine new C++ operators using natural language
python miner_agent.py

# Process L2 snapshots into minute-frequency Base Panel features
nohup python run.py > run_tick.log 2>&1 &