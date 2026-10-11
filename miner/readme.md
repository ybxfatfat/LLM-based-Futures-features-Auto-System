# 🧬 LLM-GA Futures Alpha Miner / 大模型分层遗传因子挖掘系统

**[EN]** This directory implements the **second-stage, minute-frequency miner**: it starts from the Level-2-derived base panels produced upstream, then combines LLM-guided hypothesis generation with a PyTorch Directed Acyclic Graph (DAG) engine to propose, evaluate, and organize symbolic Alphas and market-regime candidates.

**[CN]** 本目录实现的是**第二阶段的分钟频挖掘**：以上游从 L2 快照降频得到的 Base Panel 为输入，将 LLM 引导的假设生成与 PyTorch DAG 计算图结合，通过五层分工、探索与利用（E&E）抽样及量化约束，提出、评估并组织符号化 Alpha 与市场状态候选。

![LLM-GA 架构图](./figure.png)

**[EN]** The flowchart is saved as both [`figure.png`](./figure.png) and the editable [`figure.html`](./figure.html). It describes the minute-frequency evolutionary miner in this directory; the upstream snapshot-to-minute RAG miner is a separate system under [`../tick_to_min/`](../tick_to_min/).

**[CN]** 流程图同时保存为 [`figure.png`](./figure.png) 与可编辑的 [`figure.html`](./figure.html)。它描述的是本目录中的分钟频演化挖掘；上游“快照→分钟基础字段”的 RAG 挖掘是 [`../tick_to_min/`](../tick_to_min/) 下的另一套系统。

---

## 🔥 Executive Overview / 架构与研究方法速览

### The 5-Layer Pipeline at a Glance / 五层流程概览
**[EN]** Instead of blind formula generation, the LLM progresses through a strictly scaffolded 5-step pipeline.
**[CN]** 系统不让大模型一次性写完所有逻辑，而是将研究过程拆为 5 个定向阶段：
*   **[L1] Raw Ingredients (基础原料层)**: Lets the LLM directly reuse or modify sampled Base/Foundation Features, while requiring it to learn from their economic reasoning and also originate new, potentially profitable mechanisms from raw fields. / 允许 LLM 直接使用或微调抽样得到的 Base/Foundation Features，同时要求它学习这些特征背后的经济推理方式，并从原始字段中原创具有潜在盈利逻辑的新机制。
*   **[L2] Relativization (尺度相对化层)**: Makes candidates dimensionless or otherwise scale-comparable for mixed-instrument training, preferably through economically meaningful relative quantities. / 为混品种训练把候选改造成无量纲或至少尺度可比的表达，并优先使用具有经济含义的相对量。
*   **[L3] Distribution Transformation (分布—机理诊断层)**: Requires the LLM to jointly read the formula, economic rationale, and 10-quantile return profile; it must explain why the observed shape arises before proposing a logically consistent transformation. / 要求 LLM 联合阅读公式、经济逻辑与 10 分组收益图，先解释单调、U 型、单侧尾部或噪声形态的成因，再提出与机制一致的变换。
*   **[L4] Temporal & Interaction (复杂时序交互层)**: Exposes compound operators such as EWMAC to support nested momentum, reversal, and interaction hypotheses, with recursion depth capped by the prompt. / 提供 EWMAC 等复合算子，用于构造嵌套的动量、反转与交互假设，并通过 Prompt 限制递归深度。
*   **[L5] Smoothing & Window Sweeping (降噪与窗口扫描层)**: Adds smoothing and generates 5-8 time-window variants so the same mechanism can be compared across different horizons. / 增加平滑处理并生成 5-8 组时间窗口变体，用于比较同一机制在不同周期下的表现。

### Exploration vs. Exploitation / 蓄水池的 E&E 抽样
**[EN]** The system dynamically balances deep exploitation of new genes and broad exploration of dormant genes across persistent pools.
**[CN]** 系统在层级间维护持久化基因池，在生成新一代 Prompt 时进行抽样平衡：
*   **Exploitation (深度利用)**: Extracts the freshly generated "new generation" genes to polish promising branches continuously. / 强制提取上一层刚刚生成的“新生代”基因，顺藤摸瓜将潜力分支一路打磨到底。
*   **Exploration (广度探索)**: Randomly revives dormant genes from historical pools to reduce repeated convergence on the same local branches. / 随机从历史沉淀池中唤醒“沉睡基因”重新变异，减少研究反复收敛到同一局部分支。

### LLM-As-A-Reviewer & Domain Constraints / 领域约束与大模型审查
**[EN]** The feature interfaces are causal by construction: snapshot classes update sequentially from current/past observations, while minute-frequency feature formulas are built from contemporaneous fields and backward-looking operators. Future-return operators are reserved for label construction and are not exposed as feature-building primitives in the miner prompt. The LLM is used as one review signal when pruning redundant candidates, not as proof of validity.
**[CN]** 特征接口在构造上遵守因果时间顺序：快照类只能随当前/历史观测逐步更新，分钟频特征公式则由当期字段与后向时序算子组成；未来收益算子只负责生成标签，不作为挖掘 Prompt 中的特征构造原语。候选去冗余时，LLM 只是一个审查信号，并不等于有效性证明。

### Symbolic Feature Discovery for DL / 面向深度学习的符号降维
**[EN]** The auxiliary `parse_factors.py` script analyzes formula lineage and recurring subexpressions as candidate roots for simpler downstream inputs. It is exploratory tooling; recurrence alone does not establish signal quality.
**[CN]** 辅助脚本 `parse_factors.py` 分析公式血缘与重复子表达式，将其作为更简洁下游输入的候选根节点。这属于探索性工具，重复出现本身并不能证明信号质量。

### Two Mining Stages, Two Memory Systems / 两个挖掘阶段、两种记忆机制

| Stage / 阶段 | Research object / 研究对象 | Candidate representation / 候选表达 | Memory mechanism / 记忆机制 |
|---|---|---|---|
| Snapshot → minute base fields / 快照→分钟基础字段 | Stateful microstructure measurements accumulated from snapshots. / 从快照流累积得到的状态化微观结构测量。 | A reviewable Python class implementing the `BaseTickFeature` interface. / 实现 `BaseTickFeature` 接口、可人工审查的 Python 类。 | **Vector RAG** in `tick_to_min`: embed the economic hypothesis, retrieve semantically similar attempts, and carry code, metrics, and reflections as payload. / `tick_to_min` 中的**向量 RAG**：对经济假设做嵌入检索，并把代码、指标与反思作为载荷保存。 |
| Minute base panel → higher-order factors / 分钟基础面板→高阶因子 | Relativization, nonlinear reshaping, temporal interaction, smoothing, and horizon variation. / 尺度相对化、非线性变换、时序交互、降噪与周期变体。 | A symbolic formula compiled, macro-expanded, and executed by the DAG engine. / 经宏展开并由 DAG 引擎执行的字符串公式。 | **Lineage-aware evolutionary memory** in `miner`: persistent layer reservoirs, parent/formula references, recurring AST subgraphs, metrics, and failure lessons. It behaves more like an LLM-readable wiki or graph than an embedding-based RAG index. / `miner` 中的**血缘感知演化记忆**：持久化分层蓄水池、父代/公式引用、重复 AST 子图、指标与失败教训；它更接近供 LLM 阅读的 Wiki 或图结构，而不是基于 Embedding 的 RAG 索引。 |

**[EN]** The output boundary is explicit: the first system discovers auditable base fields; after review and registration, those fields enter the Base Panel and become raw material for the second system. Calling both systems “RAG” would erase an important architectural distinction.

**[CN]** 两者的交接边界是明确的：第一套系统挖掘可审查的分钟基础字段，经人工审核与显式注册后写入 Base Panel，再成为第二套系统的原料。把两者都笼统称作“RAG”会掩盖这项关键架构差异。

---

## 🏗️ System Modules / 核心组件说明

*   **`evolutionary_miner.py`**
    **[EN]** The orchestrator of the 5-layer evolutionary network. It normalizes formulas with AST-based rewrites and rejects candidates that fail parsing or execution checks; these checks reduce common failures but do not guarantee semantic correctness.
    **[CN]** 流水线调度器。它通过 AST 重写规范化公式，并拒绝未通过解析或执行检查的候选；这些检查能减少常见故障，但不保证公式在语义上正确。
*   **`evaluator.py`**
    **[EN]** The CPU/GPU Vectorized Evaluation Engine utilizing 2000-minute non-overlapping rolling windows (Period IC) and 10-quantile Monotonicity checks.
    **[CN]** 向量化评测引擎。采用 2000 分钟不重叠窗口（Period IC），并结合 10 分组结果检查单调性与阶段稳定性。
*   **`state_miner.py`**
    **[EN]** Market Regime Discovery. Reverses the Alpha question by asking the LLM to hypothesize observable minute-frequency environments associated with realized winning and losing trade samples.
    **[CN]** 微观市场环境挖掘器。它从已实现的盈利／亏损交易样本反推可观测的分钟频市场环境，让大模型充当“侦探”生成刻画特定状态（如流动性真空）的张量；这里的样本不是上游原始快照挖掘。
*   **`parse_factors.py`**
    **[EN]** AST Lineage Parser. Extracts high-scoring Root components and generates interactive ECharts HTML graphs showing genetic diversity.
    **[CN]** AST 血缘解析器。拆解高分 Root 组件，生成带有强弱节点（基于 Max Descendant IR）的 ECharts 交互式关系网大屏。
*   **`manage_libraries.py`**
    **[EN]** Interactive CLI. Allows researchers to describe ideas in natural language, then validates the LLM-produced JSON structure before writing it to the configured library.
    **[CN]** 人机交互 CLI。允许研究员用自然语言描述灵感，并在写入配置库前校验 LLM 生成的 JSON 结构。

---

## 🚀 Minimal Invocation / 最小运行方式

### Configuration / 环境配置
**[EN]** Ensure your environment variables are set for the standard LLM client:
**[CN]** 确保环境中已配置原生的 LLM 接口凭据：
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4o"
```

Run commands from the repository root so package imports resolve consistently. / 请从仓库根目录运行，以保证包导入路径一致。

```bash
python -m miner.evolutionary_miner
```

---

## 🧭 Human-in-the-Loop: What Is Being Automated—and What Is Not / 人机协同：自动化对象与边界

**[EN]** This is a **human-in-the-loop** research system. The miner automates hypothesis expansion, symbolic reformulation, parameter-family generation, and first-pass diagnostics; the researcher defines the research grammar, curates the Base/Compound libraries and failure lessons, interprets the evidence, and decides what is promoted or frozen. The LLM does **not** automate scientific truth: it cannot turn an in-sample correlation into causal evidence or certify that a signal will survive costs and regime change.

**[CN]** 这是一套 **human-in-the-loop（人在回路）** 的研究系统。挖掘器自动化的是假设扩展、符号化改写、参数族生成与第一轮诊断；研究员负责定义研究语法、维护 Base/Compound 库与失败教训、解释证据，并决定哪些候选可以晋级或冻结。LLM 并不“自动化科学真理”：它不能把样本内相关性变成因果证据，也不能保证信号在交易成本与市场状态变化后继续有效。

The intended division of labor is: / 本系统希望形成的分工是：

| Participant | Responsibility / 职责 |
|---|---|
| Researcher / 研究员 | Curates priors and reusable libraries, defines sample boundaries and evaluation rules, reviews generated mechanisms, and owns promotion/freeze decisions. / 维护经济先验与可复用知识库，定义样本边界和评价规则，审查生成机制，并对晋级与冻结决策负责。 |
| LLM | Generates explicit, challengeable hypotheses and alternative symbolic implementations. / 生成显式、可质疑的假设与多种符号化实现。 |
| DAG engine | Applies the stated formula consistently and reports calculation failures. / 一致执行既定公式，并暴露计算失败。 |
| Evaluator | Produces diagnostics under a fixed sample definition; it does not declare a universal Alpha. / 在固定样本定义下生成诊断，但不宣称发现了普适 Alpha。 |
| OOS simulation | Tests a frozen research decision under later, untouched data and execution assumptions. / 在后续未触碰数据及执行假设下检验冻结后的研究决策。 |

## 1. Why a Five-Layer Process? / 为什么采用五层研究流程？

**[EN]** One-shot generation mixes economic hypothesis, normalization, nonlinear transformation, temporal structure, and parameter search into a single opaque answer. The five-layer design separates those decisions so that each mutation has a stated purpose and can be rolled back independently.

**[CN]** 一次性生成会把经济假设、尺度调整、非线性变换、时序结构与参数搜索混在一个黑盒答案里。五层设计把这些决定拆开，使每一次变异都有明确目的，也能单独回退和复盘。

### L1 — Logical Genesis / 逻辑起点

**Research question / 研究问题:** What observable market mechanism might precede returns, and which participant or constraint creates the delay? / 哪种可观测市场机制可能领先收益，哪类参与者或约束造成了反应延迟？

**[EN]** L1 deliberately supports two complementary routes. First, the LLM may directly call, slightly modify, or combine sampled Base/Foundation Features whose mechanism is already documented. Second—and more importantly for novelty—the library acts as an analogy set: the LLM is asked to imitate the *style of economic reasoning* behind those examples or build an entirely new potentially profitable mechanism from raw fields. The prompt explicitly discourages over-reliance on templates and asks roughly half of the output to be original constructions from raw inputs. Every candidate must still explain the counterparty, the delayed adjustment, and why the relationship could persist.

**[CN]** L1 有意保留两条互补路径。第一，LLM 可以直接调用、轻度调整或组合已写明经济机理的 Base/Foundation Features。第二，也是创新上更重要的一条：特征库同时承担“类比样本”的作用，要求 LLM 模仿其背后的经济推理方式，或者直接从原始字段中原创具有潜在盈利逻辑的市场机制。Prompt 明确要求不能过度依赖模板，并尽量让约一半输出来自原始字段的原创构造。无论走哪条路径，候选都必须解释谁是潜在对手方、价格为何没有立即调整，以及这种关系为何可能持续。

### L2 — Relativization / 尺度相对化

**Research question / 研究问题:** How can the same mechanism become dimensionless or otherwise scale-comparable enough for mixed-instrument training? / 如何把同一种机制改造成无量纲或至少尺度可比的表达，以便进行混品种训练？

**[EN]** The primary purpose of L2 is cross-instrument comparability. Raw price, volume, open interest, and book depth live on very different scales across futures; allowing those levels to pass through unchanged can make a pooled model identify the contract rather than the economic state. The prompt therefore asks for meaningful relative quantities—such as ratios to price, rolling volatility, historical volume baselines, open interest, or another economically justified anchor. If an L1 feature is already a ratio or normalized state, it may pass through without an artificial second normalization. The requirement is comparable semantics, not division for its own sake; every denominator must still be checked for noise amplification and unstable tails.

**[CN]** L2 的首要目的就是为混品种训练建立可比性。不同期货品种的价格、成交量、持仓量与盘口深度天然不在同一尺度上；若原始绝对量直接进入混合模型，模型很可能只是在识别“这是哪个品种”，而不是识别经济状态。因此 Prompt 优先要求使用有意义的相对量，例如相对价格、滚动波动率、历史成交量基准、持仓量或其他具有经济解释的锚点。若 L1 候选本身已经是比例或标准化状态，则可以直接通过，不必为了形式再除一次。硬要求是语义与尺度可比，而不是机械做除法；每个分母仍需检查噪声放大和尾部不稳定问题。

### L3 — Distribution and Monotonicity Diagnosis / 分布与单调性诊断

**Research question / 研究问题:** Given the formula's economic meaning, why does its empirical distribution map into this particular future-return shape, and what transformation—if any—preserves that meaning while improving usability? / 结合公式的经济含义，为什么它的经验分布会对应这种未来收益形状；若需要变换，怎样在保留原机制的前提下提高可用性？

**[EN]** L3 never reads the decile chart in isolation. The prompt passes forward the parent formula, its rationale, core metrics, and the 10-quantile future-return profile, then requires formula-level attribution of the observed shape. A U-shape may reflect magnitude rather than direction; a one-sided tail may arise from asymmetric liquidity constraints; weak amplitude may indicate rapid decay or excessive noise; a zig-zag profile may simply be unrepairable noise. Only after explaining the shape should the LLM propose `Abs`, sign separation, ranking, clipping, a threshold-like transform, or rejection. The aim is an economically consistent and more usable response—not cosmetic monotonicity at any cost. Because this diagnosis uses 2024 outcomes, the chosen transformation is an in-sample research decision and must be frozen before later OOS evaluation.

**[CN]** L3 从不孤立地“看图改公式”。Prompt 会同时传入父代公式、原始 rationale、核心指标与 10 分组未来收益图，并强制要求从公式实质出发解释形状成因：U 型可能说明真正有信息的是幅度而非方向；单侧尾部可能源于流动性约束的不对称；振幅过弱可能意味着信号衰减太快或噪声过大；锯齿形也可能只是无法修复的噪声。只有完成这种经济逻辑与分布形状的联合归因后，LLM 才能提出 `Abs`、正负侧拆分、排序、截尾、阈值化或直接淘汰。目标是得到经济含义一致且更可用的响应，而不是不惜代价制造表面单调。由于诊断使用了 2024 年结果，最终变换属于样本内研究决策，必须在后续 OOS 前冻结。

### L4 — Temporal and Interaction Structure / 时序与交互结构

**Research question / 研究问题:** Is the state persistent, accelerating, decaying, or conditional on another market variable? / 该状态具有持续、加速、衰减特征，还是依赖另一个市场变量？

Compound operators encode reusable research motifs such as path efficiency, fast-slow interactions, decay, or volatility adjustment. Their purpose is twofold: reduce repeated implementation effort and show the LLM examples of useful structural complexity. They are not a license for unlimited nesting. Depth limits, macro expansion, and final primitive formulas keep the result inspectable and executable. / Compound Operator 编码路径效率、快慢线交互、衰减、波动率调整等可复用研究母题。它们既减少重复实现，也向 LLM 展示“有用途的复杂度”；但这不意味着可以无限嵌套。深度限制、宏展开与最终基础公式共同保证结果仍可解释、可执行。

### L5 — Smoothing and Window Families / 平滑与窗口族

**Research question / 研究问题:** At which horizon is the mechanism visible, and is the apparent optimum a broad plateau or a single lucky parameter? / 机制在哪个时间尺度上可见，最优点是宽阔平台还是单一幸运参数？

Window sweeping is used to inspect parameter stability, not merely to select the maximum IC. A mechanism that survives across neighboring horizons is easier to trust than an isolated spike. The number of tested variants is also part of the multiple-testing burden and should be logged. / 窗口扫描用于检查参数稳定性，而不只是挑最大 IC。能够在相邻周期存活的机制通常比孤立尖峰更可信；同时，测试过多少参数本身也是多重检验负担的一部分，必须记录。

## 2. Creativity Scaffolds / 创新脚手架

### Foundation features / 基础特征库

**[EN]** Foundation Features are not merely convenient starting values. They have three roles: reusable formula components, compact carriers of prior microstructure knowledge, and worked examples from which the LLM can imitate a style of mechanism construction. This combination lets L1 both exploit established ideas and use them as inspiration for original hypotheses built from raw fields.

**[CN]** Foundation Features 不只是方便调用的起点，而同时承担三种作用：可复用的公式组件、已有微观结构知识的压缩载体，以及供 LLM 模仿“如何构造市场机制”的示例。正因为这三种角色并存，L1 才能一边复用成熟思路，一边把它们当作从原始字段原创假设的灵感来源。

### Compound operators / 复合算子库

These package reusable temporal structures and serve as examples that encourage the LLM to construct richer—but still auditable—logic. Before execution, macros are expanded into operators known by the DAG registry. / 复合算子封装可复用时序结构，也作为示例启发 LLM 构造更丰富但仍可审计的逻辑；执行前，宏会展开为 DAG 注册表认识的基础算子。

### Lineage-aware wiki/graph memory / 血缘式 Wiki／图结构记忆

**[EN]** The minute-frequency miner does not use the snapshot miner's embedding-based RAG database as its main memory. Its knowledge is organized around formula lineage: persistent `L2_pool`–`L5_pool` reservoirs retain candidates from previous rounds; parent names and formulas make descendants reconstructable; metrics and 10-quantile profiles travel with the candidate; `parse_factors.py` can extract recurring AST subexpressions; and global `kb_lessons` inject remembered failures into every prompt. Conceptually, this is closer to an LLM-readable research wiki plus a formula graph.

**[CN]** 分钟频挖掘器并不把快照挖掘器的 Embedding RAG 当作主记忆。它围绕公式血缘组织知识：持久化的 `L2_pool`–`L5_pool` 蓄水池保留跨轮候选；父代名称与公式使后代路径可以重建；指标与 10 分组形状随候选一同传递；`parse_factors.py` 可以抽取反复出现的 AST 子表达式；全局 `kb_lessons` 则把历史失败教训注入每一层 Prompt。从概念上看，它更接近一个可供 LLM 阅读的研究 Wiki 加公式关系图。

### Reservoir mechanics and E&E / 层间蓄水池与探索—利用

**[EN]** Each layer boundary is a persistent reservoir rather than a one-shot queue. After L1–L4 evaluation, candidates are appended to the next layer's pool and the new names are recorded in `latest_outputs`. When the next layer is prompted, the sampler first takes up to ten latest descendants—**exploitation**, which keeps a newly promising lineage moving forward—then randomly fills the remaining slots to a target of ten from older pool members—**exploration**, which can reactivate dormant branches. The full historical pools remain available for parent-formula resolution even when only a subset enters the prompt.

**[CN]** 每个层间边界都是持久化蓄水池，而不是一次性队列。L1–L4 评估完成后，候选会追加到下一层池中，新生成名称同时写入 `latest_outputs`。下一层构造 Prompt 时，系统先抽取最多十个最新后代——这是 **Exploitation**，保证新出现的潜力血脉能够继续向下深挖；若不足十个，再从旧池随机补齐——这是 **Exploration**，让沉睡分支有机会重新被激活。即使只有一个子集进入当轮 Prompt，完整历史池仍用于解析父代公式与恢复血缘。

**[EN]** Reservoir capacity is also part of the search policy. When a layer pool exceeds 100 candidates, an LLM-assisted review scores logic, redundancy, Full/Period IC consistency, nonlinearity, and tail behavior, then prunes the pool to 50. This is a bounded-memory E&E mechanism: retaining only recent winners would collapse diversity, while uniform random carry-forward would spend too much budget on weak branches. L5 survivors enter the round-local `active_pool`; candidates passing quantitative gates at any layer are separately copied to the Hall of Fame, so a strong intermediate cannot be lost merely because a later mutation fails.

**[CN]** 蓄水池容量本身也是搜索策略的一部分。当某层池超过 100 个候选时，系统让 LLM 基于逻辑、同质化、Full/Period IC 一致性、非线性与尾部表现辅助打分，再把池裁剪到 50 个。这是一种有界记忆的 E&E：如果只保留最新赢家，多样性会快速塌缩；若完全随机传递，又会把大量预算浪费在弱分支上。L5 存活者进入本轮临时 `active_pool`；任何层级中通过量化门槛的候选还会被单独复制到 Hall of Fame，避免一个优秀中间体仅因后续变异失败而丢失。

**[EN]** Randomness is useful only when it is attributable. For strict reproducibility, the random seed, sampled pool members, prompt text, parent identifiers, and generated descendants should be logged together. / **[CN]** 随机性只有在可归因时才有研究价值。为了严格复现，应把随机种子、当轮抽到的池成员、Prompt 文本、父代标识与生成后代一起记录。

### Snapshot-stage RAG / 快照阶段的 RAG

**[EN]** The true retrieval-augmented experience library lives in `tick_to_min/rag_vector_store.py`. It embeds a proposed microstructure mechanism and counterparty description, retrieves semantically similar historical attempts, and returns their metrics and reflections before a new `BaseTickFeature` class is drafted. After evaluation, the complete class code can be written back with the post-mortem. This RAG loop is specific to discovering snapshot-aggregation base fields; it should not be conflated with the formula-lineage reservoirs above.

**[CN]** 真正基于检索增强的经验库位于 `tick_to_min/rag_vector_store.py`。它对候选微观机制与对手方描述做向量化，在生成新的 `BaseTickFeature` 类之前检索语义相近的历史尝试及其指标、反思；评估后还可以把完整类代码与复盘结果写回。这条 RAG 闭环专门服务于“快照降频基础字段”的挖掘，不应与上面的分钟公式血缘蓄水池混为一谈。

### Text-only idea cards / 纯文字 Idea 卡片（实验方向）

A useful extension is a researcher-maintained library of observations that describe *what may be happening* without prescribing a formula. Randomly sampled cards can stimulate alternative implementations while remaining distinguishable from reusable code templates. This direction has been explored conceptually but is not presented as a fully tuned component of the current repository. / 一个值得继续推进的方向，是由研究员维护只描述“市场可能发生什么”、但不预设公式的文字观察库。每轮随机注入少量卡片，可以启发不同实现，同时与代码模板清晰区分。该方向已经做过概念性尝试，但本仓库不把它描述为已完整调优的模块。

## 3. Evaluation Semantics / 评测指标应该怎样解释

### 2024 is the research sample / 2024 是研究样本

**[EN]** Candidate generation, formula transformation, quantile diagnosis, parameter search, and state discovery currently use 2024 data. Metrics from these stages are in-sample evidence used to make research decisions. They can rank or reject candidates, but they are not OOS performance claims.

**[CN]** 当前候选生成、公式变换、分组诊断、参数搜索与市场状态发现都使用 2024 数据。这些指标是用于做研究决策的样本内证据，可以用来排序或淘汰候选，但不能被描述为样本外表现。

### Full IC and Period IC / 全期 IC 与分段 IC

Full vs. Period and pooled vs. per-instrument are **two independent axes**, producing four complementary views rather than one pooled statistic plus one symbol average. / Full 与 Period、混品种与单品种是**两条相互独立的维度**，共同形成四种互补视角，而不是“一个混合指标加一个品种均值”。

| Scope / 统计范围 | Full IC / 全期 IC | Period IC / 分段 IC |
|---|---|---|
| **Pooled universe / 混品种池** | Correlation over all valid rows in the selected Index or Commodity pool for the research year. It tests whether one common formula has a pooled association, but may be dominated by instruments or regimes with more observations. / 在研究年度内，把所选股指池或商品池的全部有效行合并后计算相关性；检验同一公式的混合样本关系，但可能受样本更多的品种或状态主导。 | For each global 2000-minute block, compute one IC across the selected pool, then summarize the block-IC series by mean, dispersion, win rate, and IR. / 在每个全局 2000 分钟区间内对所选品种池计算一次 IC，再汇总区间 IC 序列的均值、波动、胜率与 IR。 |
| **Per instrument / 单品种** | Correlation over the full research year separately for each instrument. It reveals sign disagreement and concentration that pooled Full IC can hide. / 对每个品种分别使用完整研究年度计算相关性，用于暴露混品种 Full IC 可能掩盖的符号分歧与品种集中。 | Within each instrument, compute IC separately for every 2000-minute block, then summarize that instrument's block-IC series. It diagnoses temporal stability without mixing contracts. / 在每个品种内部逐个 2000 分钟区间计算 IC，再汇总该品种自己的区间 IC 序列，用于在不混合品种的情况下检查时间稳定性。 |

**[EN]** The pooled and per-instrument versions answer different questions and neither replaces the other. The 2000-minute blocks are non-overlapping in their feature timestamps, but forward-return labels may still overlap near adjacent observations; block counts must therefore not be interpreted as a matching number of independent bets.

**[CN]** 混品种版与单品种版回答的是不同问题，不能互相替代。2000 分钟区间在特征时间戳上不重叠，但区间内相邻样本的未来收益标签仍可能重叠，因此不能把区间数量直接解释成同等数量的独立下注。

### Quantile returns / 分组收益

Quantile means help distinguish linear association, tail concentration, asymmetry, and non-monotonic response. They are especially useful for tree-model features because a feature may be valuable through thresholds even when Pearson IC is modest. The same chart is also easy to overfit; every transform chosen after viewing it consumes research degrees of freedom. / 分组均值能够区分线性关联、尾部集中、非对称与非单调响应。对树模型而言，即使 Pearson IC 不高，阈值结构也可能有价值。但分组图同样容易被过拟合；每一次“看图后改公式”都会消耗研究自由度。

### IR and annualization / IR 与年化

**[EN]** IR is calculated from the Period-IC sequence in both the pooled-universe and per-instrument reports. It is a scale-dependent diagnostic for comparing candidates under one fixed convention, not a count of independent trades. Minute-level forward labels overlap, so the absolute IR must be read together with the holding horizon, block construction, sample coverage, and cost model.

**[CN]** IR 在混品种报告与单品种报告中，都是基于各自的 Period IC 序列计算的。它是在统一口径下比较候选的尺度化诊断量，而不是独立交易次数的映射。分钟级未来收益标签彼此重叠，因此 IR 的绝对值必须结合持有周期、分段方式、样本覆盖与费用模型一起解释。

## 4. Market-State Discovery / 市场状态挖掘

**[EN]** `state_miner.py` searches for conditions under which an existing baseline behaves differently. This reverses the usual question: rather than asking only “what predicts return?”, it asks “under which observable environment does this strategy win, lose, or become unreliable?” Candidate state formulas are cleaned, compound macros are expanded, and the resulting tensors are evaluated against the configured 2024 sample.

**[CN]** `state_miner.py` 寻找基准策略表现发生变化的条件。它把问题从“什么预测收益”反转为“在什么可观测环境下，这个策略赚钱、亏损或失效”。候选状态公式会先清洗并展开复合宏，再在配置的 2024 样本上计算状态张量。

qcut thresholds, mutual information, KS statistics, state IR, and final state selection are currently exploratory statistics calculated on the same research sample. They are useful for hypothesis formation, not OOS confirmation. A promoted state should freeze its formula and thresholds, then be applied unchanged to a later period. / 当前 qcut 阈值、互信息、KS 统计量、状态 IR 与最终选择都来自同一研究样本，因此适合形成假设，而不是确认 OOS。状态一旦晋级，应冻结公式与阈值，再原样应用到后续时期。

## 5. Traceability and Failure Memory / 可追溯性与失败记忆

**[EN]** Traceability is the operational side of human-in-the-loop research: the model proposes and critiques, but a researcher must be able to reconstruct what information entered the decision and why a branch was kept, transformed, or rejected. The current state files already retain core candidate formulas, rationales, metrics, layer pools, and lineage references; the list below is the fuller reproducibility target.

**[CN]** 可追溯性是 human-in-the-loop 在工程上的落点：模型负责提出与反思，但研究员必须能重建一次决策看到了什么信息，以及某条分支为何被保留、变换或淘汰。当前状态文件已经保存候选公式、rationale、指标、分层池与血缘引用等核心内容；下面列出的是更完整的可复现目标。

Every promoted or rejected candidate should ideally retain: / 每个晋级或被淘汰的候选，理想情况下都应保留：

1. Candidate ID, parent ID, layer, prompt material, and random seed. / 候选 ID、父代 ID、所属层、Prompt 材料与随机种子。
2. Economic hypothesis, counterparty explanation, and persistence argument. / 经济假设、对手方解释与持续性理由。
3. Original formula, macro-expanded formula, and final normalized formula. / 原始公式、宏展开公式与最终规范化公式。
4. Data version, date range, instrument pool, label horizon, and masks. / 数据版本、日期区间、品种池、标签周期与掩码。
5. Full metrics, rejection reason, LLM reflection, and human decision. / 完整指标、淘汰原因、LLM 反思与人工决定。
6. Generated Python class code where the candidate is represented as code. / 当候选以代码表达时，保存完整 Python 类实现。

This is why the reflection library stores actual generated code rather than a placeholder such as `"Class Code Saved"`. A failure becomes reusable knowledge only when the failed implementation can be reconstructed. / 因此反思库需要保存真实生成代码，而不是 `"Class Code Saved"` 之类占位符。只有能重建失败实现，失败经验才具有复用价值。

## 6. Module Map / 文件职责

| Module | Responsibility / 职责 |
|---|---|
| `evolutionary_miner.py` | Five-layer orchestration, material pools, prompt construction, selection, and experiment persistence. / 五层调度、材料池、Prompt 构建、筛选与实验状态保存。 |
| `evaluator.py` | Builds or reuses tensor context, executes factor batches, and calculates research-period diagnostics. / 构建或复用张量上下文、执行因子批次并计算研究期诊断。 |
| `state_miner.py` | Generates, normalizes, expands, and evaluates market-state formulas. / 生成、规范化、展开并评估市场状态公式。 |
| `libraries/` | JSON-backed Foundation Features and Compound Operators: reusable formulas, research priors, and analogy examples for LLM innovation. / 基于 JSON 的 Foundation Features 与 Compound Operators：既是可复用公式和研究先验，也是启发 LLM 类比创新的示例。 |
| `manage_libraries.py` | Human-in-the-loop CLI for validating and registering library entries. / 用于校验并注册知识库条目的人工交互 CLI。 |
| `sample_features.py` | Curated examples with **2024 in-sample** diagnostics; examples are not OOS claims. / 带有 **2024 样本内**诊断的示例公式，不代表 OOS 结论。 |
| `parse_factors.py` | Auxiliary AST-subexpression lineage visualization; useful for building the graph/wiki view, not proof that recurring roots are predictive. / 辅助 AST 子表达式血缘可视化，用于形成图结构／Wiki 视角，但不能证明高频出现的根节点必然有效。 |
| `../tick_to_min/miner_agent.py` + `rag_vector_store.py` | Separate upstream RAG loop for discovering snapshot-aggregation `BaseTickFeature` classes. / 独立的上游 RAG 闭环，用于挖掘快照降频 `BaseTickFeature` 类。 |

## 7. Promotion Checklist / 候选晋级检查表

Before calling a candidate “research-ready,” verify: / 在把候选称为“可进入下一阶段”前，应确认：

- The candidate was defined through the framework's history-only feature interface: sequential snapshot classes or minute formulas composed from contemporaneous fields and backward-looking feature operators. Under this contract, causal calculation is guaranteed by construction; future-return operators remain in the separate label path. / 候选通过框架规定的历史信息特征接口定义：要么是按时间顺序更新的快照类，要么是由当期字段与后向特征算子组成的分钟公式。在这一约束内，因果计算由框架构造保证；未来收益算子只存在于独立的标签路径。
- Formula parsing, macro expansion, and operator registration all succeed. / 公式解析、宏展开与算子注册均成功。
- The mechanism remains understandable after full expansion. / 完整展开后仍能说明经济机制。
- Performance is not carried by one instrument, one quarter, or one narrow parameter. / 表现并非由单一品种、单一季度或单一窄参数支撑。
- Missingness, coverage, turnover, and tail behavior are acceptable. / 缺失率、覆盖率、换手与尾部行为可接受。
- The number of attempted variants is recorded. / 已记录尝试过的变体数量。
- All research decisions are frozen before later-period OOS evaluation. / 后续样本外评估前已冻结全部研究决策。
