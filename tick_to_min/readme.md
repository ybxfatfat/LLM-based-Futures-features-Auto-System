# High-Frequency Futures Feature Engine / 高频期货特征计算引擎

A hybrid C++/Python framework for quantitative trading research. It converts high-frequency Level-2 snapshots into minute-level aggregated features through a symbolic Directed Acyclic Graph (DAG) execution engine.
本项目是一套面向量化研究的 C++/Python 混合框架，通过符号化有向无环图（DAG）执行引擎，将高频 L2 快照聚合为分钟级特征宽表。

## 🎯 Core Paradigms / 核心架构范式

*   **Contiguous Buffer Handoff & C++ Native Execution (连续缓冲区传递与 C++ 原生计算)**:
    *   **[EN]** Python orchestrates the graph while NumPy stores input columns in contiguous arrays. PyBind11 exposes their buffers to the C++ engine, where the row-wise numerical loop runs without per-snapshot Python dispatch. This describes the binding boundary and does not imply that file loading and preprocessing are copy-free end to end.
    *   **[CN]** Python 负责组织计算图，输入列由 NumPy 连续数组承载，PyBind11 将其缓冲区暴露给 C++ 引擎；逐快照的数值循环在 C++ 中执行，避免每条数据都回到 Python 调度。这里描述的是绑定边界，并不表示文件读取与预处理端到端完全没有内存复制。
*   **Symbolic Tracing Graph (符号化计算图追踪)**: 
    *   **[EN]** In Python, you write equations that look like standard math, but they actually act as blueprints. Python "draws" the DAG, and the C++ engine dynamically allocates memory and processes the data pipeline based on this graph. 
    *   **[CN]** 在 Python 端编写的代码并不执行具体的数值计算，而是充当“图纸”。Python 负责“画图” (将公式解析并连线)，底层 C++ 引擎根据这张图纸动态分配算子并执行流水线运算。
*   **Stateful Time-Series Ops (状态化时序算子)**: Built-in $O(1)$ sliding-window moments, monotonic deques for extrema, and $O(\log W)$ red-black trees for quantile calculation. (内置 $O(1)$ 滑动窗口高阶矩、用于极值的单调队列，以及 $O(\log W)$ 红黑树分位数计算)。

## 📖 How It Works: Quote-Constrained Trade-Flow Attribution / 运作机制：盘口约束下的成交量方向归因

**[EN]** `BuyRatio` is a snapshot-based estimator rather than an exact aggressor-side label. It recovers interval traded volume and VWAP from cumulative-field differences, then attributes that volume by locating the VWAP relative to the previous and current best quotes. When the two quote intervals overlap, the common interval supports proportional attribution. When they do not overlap or merely touch, the estimator falls back to the **previous** spread so that post-trade quote movement does not reverse the classification of a trade executed at the old bid or ask.

**[CN]** `BuyRatio` 是基于快照的成交量方向估计，而不是真实逐笔主动方标签。它先通过累计字段差分恢复区间成交量与成交均价，再根据成交均价在前后两组最优报价中的位置进行比例归因。前后报价区间存在交集时，在共同区间内线性归因；区间不重叠或仅接触时，退回**上一快照**买卖价差，避免成交发生在旧买一或旧卖一、随后盘口移动时被当前报价反向分类。

The abbreviated class below demonstrates the three-step feature lifecycle while preserving the actual attribution branches used in `run.py`. / 下方精简类展示了特征的三阶段生命周期，并保留 `run.py` 中真实使用的各条归因分支：

```python
class HighFreqMinuteFeatures(BaseTickFeature):
    def __init__(self):
        super().__init__()
        # STEP 1: Node Allocation (节点注册)
        self.tick_vol_diff = self.register_state_op(ce.TsDelta(window=1, min_periods=1), auto_clear=False)
        self.tick_amt_diff = self.register_state_op(ce.TsDelta(window=1, min_periods=1), auto_clear=False)
        self.delay_ask1 = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)
        self.delay_bid1 = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)

        self.min_vol_sum = self.register_state_op(ce.TsSum(window=0, min_periods=1))
        self.min_vbuy_sum = self.register_state_op(ce.TsSum(window=0, min_periods=1))

    def update(self, snp):
        # STEP 2: Symbolic Wiring (符号追踪连线)
        c_0 = ce.ConstNode(0.0)
        c_1 = ce.ConstNode(1.0)
        c_2 = ce.ConstNode(2.0)
        c_eps = ce.ConstNode(1e-8)

        mid = (snp.ask_1_prc + snp.bid_1_prc) / c_2
        self.delay_ask1.set_input(snp.ask_1_prc)
        self.delay_bid1.set_input(snp.bid_1_prc)
        self.tick_vol_diff.set_input(snp.cum_volume)
        self.tick_amt_diff.set_input(snp.cum_amount)

        dv = ce.IfElse((self.tick_vol_diff > c_0), self.tick_vol_diff, c_0)
        dt = ce.IfElse((self.tick_amt_diff > c_0), self.tick_amt_diff, c_0)
        self.min_vol_sum.set_input(dv)

        tick_vwap = ce.IfElse((dv > c_0), dt / (dv * snp.contract_size), mid)

        max_ask = ce.IfElse((self.delay_ask1 > snp.ask_1_prc), self.delay_ask1, snp.ask_1_prc)
        min_bid = ce.IfElse((self.delay_bid1 < snp.bid_1_prc), self.delay_bid1, snp.bid_1_prc)
        overlap_ask = ce.IfElse((self.delay_ask1 < snp.ask_1_prc), self.delay_ask1, snp.ask_1_prc)
        overlap_bid = ce.IfElse((self.delay_bid1 > snp.bid_1_prc), self.delay_bid1, snp.bid_1_prc)

        overlap_width = overlap_ask - overlap_bid
        safe_overlap_width = ce.IfElse((overlap_width > c_0), overlap_width, c_eps)
        ratio_overlap = (tick_vwap - overlap_bid) / safe_overlap_width

        # No overlap or touch: classify against the pre-trade snapshot spread.
        # 无交集或仅接触：使用成交前的上一快照价差。
        previous_spread = self.delay_ask1 - self.delay_bid1
        safe_previous_spread = ce.IfElse((previous_spread > c_0), previous_spread, c_eps)
        ratio_fallback = (tick_vwap - self.delay_bid1) / safe_previous_spread

        buy_ratio_raw = ce.IfElse(
            (tick_vwap >= max_ask), c_1,
            ce.IfElse(
                (tick_vwap <= min_bid), c_0,
                ce.IfElse((overlap_bid < overlap_ask), ratio_overlap, ratio_fallback),
            ),
        )
        buy_ratio = ce.IfElse(
            (buy_ratio_raw > c_1), c_1,
            ce.IfElse((buy_ratio_raw < c_0), c_0, buy_ratio_raw),
        )
        self.min_vbuy_sum.set_input(dv * buy_ratio)

    def get_value(self) -> dict:
        # STEP 3: Harvesting (分钟边界收割)
        # Defines the output structure. Triggered by C++ at every minute boundary.
        c_0 = ce.ConstNode(0.0)
        
        # Zero-Division Protection Rule
        raw_buy_ratio = ce.IfElse((self.min_vol_sum > c_0), self.min_vbuy_sum / self.min_vol_sum, c_0)

        return {
            'BuyRatio': raw_buy_ratio,
        }
```

The fallback's temporal anchor is deliberate. For example, if the book moves from `100/101` to `101/102` while the interval VWAP is `101`, interpolation against the previous spread returns `1`; using the current spread would incorrectly return `0`. The symmetric downward move is handled in the same way. / 这一 fallback 的时间锚点是有意选择的。例如盘口从 `100/101` 上移至 `101/102`，区间成交均价为 `101` 时，按上一盘口插值得到 `1`；若使用当前盘口则会错误得到 `0`。盘口向下移动的情形同理对称处理。

### Anatomy of the Class (类结构解析):
1. **`__init__` (Allocate)**: Declares the persistent differencing/delay states and minute-reset aggregation states in C++. (声明跨分钟保留的差分、延迟状态，以及分钟边界自动清空的聚合状态)。
2. **`update` (Wire)**: Translates financial logic into a C++ execution path using operators like `ce.IfElse`. This is executed *once* during setup to compile the graph. (利用 `ce.IfElse` 和 `ce.ConstNode` 将微观结构逻辑画成计算图。此函数在系统启动时仅运行一次用于编译拓扑)。
3. **`get_value` (Harvest)**: Extracts the computed Pointers into a dictionary. At runtime, the C++ engine loops through millions of rows, natively updating the states, and pulls data from these pointers at exact minute boundaries. (定义提取字典。C++ 在底层跑完海量循环后，会在每分钟边界自动从这些指针中将算好的数据吐出，并清理需要 `auto_clear` 的变量)。

## 🏗️ Architecture & Modules / 架构说明

*   **`tick_engine_ops.hpp` & `tick_engine.cpp`**: 
    The high-performance C++ backend. Defines all atomic mathematical and stateful operators. (高性能 C++ 后端引擎)。
*   **`tick_framework.py`**: 
    The Python symbolic routing framework. Provides `BaseTickFeature` to construct PyTorch-like computational graphs. (Python 计算图符号路由框架)。
*   **`run.py`**: 
    The multiprocessing orchestrator. Scans local Tick files, filters valid trading sessions across different exchanges, and saves output as Parquet files. (多进程调度器与交易所洗膜模块)。
*   **`update_to_basepanel.py`**: 
    The alignment tool using PyArrow. Integrates high-frequency features securely into the structured minute-level `base_panel` using column patching strategies. (基于 PyArrow 的分钟频频宽表智能对齐与合并工具)。
*   **`prompts_config.py`**: 
    Defines the LLM's C++ DAG syntax constraints and microstructure-reasoning questions, such as identifying the proposed counterparty mechanism. (定义大模型使用的 C++ 计算图语法约束与微观结构推理问题，例如要求说明候选机制中的对手方)。
*   **`miner_agent.py` & `auto_generated_factors.py`**: 
    The conversational candidate generator. It translates human ideas into symbolic Python classes, performs static AST checks, and creates `auto_generated_factors.py` on first use to store accepted candidates. Candidates require human review and explicit registration before execution. (对话式候选生成器：把自然语言灵感翻译为符号化 Python 类，经 AST 静态校验后保存；`auto_generated_factors.py` 会在首次使用时创建，候选必须经过人工审查并显式注册后才会执行)。
*   **`rag_vector_store.py`**: 
    A dual-payload RAG database. It embeds the *economic hypothesis* for semantic similarity search while storing code and performance metrics as payload metadata, reducing syntax noise in retrieval. (双载荷 RAG 向量知识库：对经济学逻辑进行向量化，同时将代码与评测指标保存为元数据，以减少代码语法对检索的干扰)。
*   **`eval_and_reflect.py`**: 
    The Post-mortem Evaluator. Calculates Top vs Bottom decile spreads (Q10-Q1), prompts the LLM for deep reflection, and writes the experience back to the RAG KB. (高频验尸官与反思引擎，评测真实分组收益差并将经验打入知识库闭环)。

## 🤖 RAG-Assisted Candidate Generation / RAG 辅助特征候选生成

The optional agent uses a **Retrieval-Augmented Generation (RAG)** loop to translate natural-language microstructure hypotheses into reviewable symbolic-class candidates. It is an assisted research workflow, not an autonomous production deployment path.
可选的智能体通过 **RAG** 闭环把自然语言微观结构假设翻译为可审查的符号类候选。它是辅助研究流程，而不是自动部署到生产执行链路的机制。

1. **Dual-Payload Semantic Retrieval (双载荷语义检索)**: The agent embeds the economic mechanism rather than the code string, then retrieves prior hypotheses, metrics, and reflections as context. This can expose similar failed attempts but cannot guarantee perfect duplicate detection. (系统对经济机制而不是代码字符串做 Embedding，并检索过往假设、指标与反思作为上下文；它有助于发现相似失败尝试，但不能保证完整查重)。
2. **Constrained Symbolic Code Generation (受约束的符号化代码生成)**: The prompt specifies zero-division guards, constant wrapping (`ce.ConstNode`), and small-sample masks. `miner_agent.py` additionally parses the returned code with Python AST, requires one class whose name matches `feature_name`, rejects imports and top-level executable statements, and prevents duplicate class names. These checks catch structural errors but do not prove numerical or economic validity. (Prompt 规定防零除、常数包装与小样本遮罩；`miner_agent.py` 还会用 Python AST 解析返回代码，要求只含一个与 `feature_name` 同名的类，拒绝导入与顶层执行语句，并阻止重复类名。这些检查用于捕获结构错误，但不能证明数值或经济逻辑有效)。
3. **Human Review & Explicit Registration (人工审查与显式注册)**: Passing static validation only writes the class to `auto_generated_factors.py`. The class is not auto-loaded by `run.py`; a researcher must inspect it and deliberately add it to the feature registration list. (通过静态校验后只会把类写入 `auto_generated_factors.py`。`run.py` 不会自动加载它；研究员需要审查代码并明确加入特征注册列表)。
4. **Post-Evaluation Reflection (评测后反思)**: After a candidate has been reviewed, registered, computed, and evaluated, `eval_and_reflect.py` can summarize metrics such as Q10-Q1 and save the reflection back to the RAG database. (候选经审查、注册、计算与评测后，`eval_and_reflect.py` 可汇总 Q10-Q1 等指标，并把反思写回 RAG 经验库)。

## 🚀 Minimal Invocation / 最小运行方式

### 1. Compile the C++ Engine / 编译底座
```bash
bash build.sh
```

### 1.5 (Optional) Run the LLM Agent to Mine New Features / 启动大模型智能挖掘
Use natural language to generate a candidate class, then review and register it explicitly before execution.
```bash
python miner_agent.py
```

### 2. Run the Feature Generator / 启动特征流水线
Execute the multi-processing pipeline to process L2 snapshots into minute-frequency features. (执行多进程流水线计算)。
```bash
nohup python run.py > run_tick.log 2>&1 &
```

### 3. Update Base Panel / 更新基础宽表
Merge the newly generated features into your local unified base panel dataset. (将高频特征拼接入你的基础数据面板)。
```bash
python update_to_basepanel.py
```

---

## 🧭 Research Contract of the Snapshot Layer / 快照层的研究契约

**[EN]** This layer is not merely a faster resampler. It defines what one minute of microstructure information means. Choices about missing snapshots, cumulative-field differencing, quote timing, state persistence, and contract boundaries all change the economic object delivered to the minute-level research system.

**[CN]** 这一层不只是更快的重采样器，它实际上定义了“一分钟微观结构信息”是什么。缺失快照如何处理、累计字段如何差分、前后盘口怎样参与成交归因、状态是否跨休市保留、合约边界怎样定义，都会改变送入分钟频研究系统的经济对象。

The design goal is therefore: / 因此这一层的目标是：

1. Preserve the distinction between observed updates and inferred static states. / 区分真实观测更新与推断出的静态状态。
2. Make every attribution rule explicit enough to audit with hand-built examples. / 让每条归因规则都能用人工构造样例审查。
3. Keep contract-local state continuous where “market paused” is the intended interpretation. / 在研究假设把休市视为“市场冻结”时，保持合约内状态连续。
4. Reset only states whose mathematical meaning is minute-local. / 只重置数学含义确实局限于单分钟的状态。
5. Export stable sample keys so later feature batches can be merged without duplicating rows. / 输出稳定样本键，使后续特征批次可以横向合并而不复制样本。

## 1. Snapshot Data Are Not Trade Data / 快照数据不等于逐笔成交数据

**[EN]** A 500 ms snapshot reports the latest cumulative volume, cumulative turnover, last price, and visible book at the publication time. Multiple trades may have occurred since the previous snapshot, and the visible book may already have reacted to those trades. Consequently, exact aggressor-side reconstruction is impossible from snapshots alone. `BuyRatio` should be interpreted as a deterministic attribution estimator conditioned on the available quote states—not as exchange truth.

**[CN]** 500 ms 快照记录的是发布时点的累计成交量、累计成交额、最新价与可见盘口。两次快照之间可能发生多笔成交，而当前盘口也可能已经对这些成交完成了重定价。因此，仅凭快照无法精确还原真实主动方。`BuyRatio` 应被解释为“在可用盘口状态条件下的确定性成交归因估计”，而不是交易所真值。

### Recovering interval trades / 恢复快照间成交

Let `ΔV` be the positive difference in cumulative volume and `ΔT` the corresponding positive turnover difference. After applying the contract multiplier convention, their ratio estimates the average execution price of all trades between snapshots. This price is a volume-weighted summary; it cannot reveal the order or individual sides of those trades. / 记 `ΔV` 为累计成交量正差分、`ΔT` 为对应累计成交额正差分。按合约乘数口径换算后，两者之比估计快照间全部成交的平均执行价。这个价格是成交量加权摘要，无法恢复逐笔顺序与逐笔方向。

### Why both previous and current BBO matter / 为什么同时使用前后盘口？

The previous quote is the last visible state before the interval; the current quote is the visible state after the interval. Their overlap provides a conservative price region compatible with both states. When the quote intervals do not overlap or merely touch, the current book may already contain post-trade repricing, so the fallback deliberately anchors attribution to the previous spread. / 上一盘口是区间开始前最后可见状态，当前盘口是区间结束后的可见状态；两者交集提供了同时与前后状态相容的保守价格区间。当区间不重叠或仅接触时，当前盘口可能已经包含成交后的重定价，因此 fallback 有意回到上一盘口价差进行归因。

### Canonical one-tick example / 典型一跳案例

| Previous BBO | Current BBO | Interval VWAP | Previous-spread attribution | Interpretation |
|---|---|---:|---:|---|
| `100 / 101` | `101 / 102` | `101` | Buy share `1` | Trade is compatible with execution at the old ask followed by an upward quote move. / 成交可能发生在旧卖一，随后盘口上移。 |
| `100 / 101` | `99 / 100` | `100` | Buy share `0` | Symmetric case: execution at the old bid followed by a downward quote move. / 对称情形：成交可能发生在旧买一，随后盘口下移。 |

Using only the current spread would reverse both examples at precisely the point where the quote movement carries directional information. The fallback does not prove aggressor side; it avoids a systematic temporal inversion in this common snapshot pattern. / 如果只使用当前价差，这两个例子都会在最有方向信息的位置被反向分类。该 fallback 仍不能证明真实主动方，但能避免常见快照模式下的系统性时间错位。

## 2. Zero-Trade Updates and Run-Length Semantics / 零成交快照与连续游程语义

**[EN]** When `ΔV = 0`, there is no interval trade to attribute. The implementation gives both attributed buy volume and attributed sell volume zero contribution. At the update level, the fallback ratio may be represented numerically, but multiplying by zero volume ensures that the minute aggregates receive no buy or sell volume from that snapshot. The exported minute `BuyRatio` is `0` when the minute's total traded volume is zero under the current convention; the corresponding sell ratio should also be interpreted as `0`, not `1 - BuyRatio`.

**[CN]** 当 `ΔV = 0` 时，区间内没有成交可供归因。实现中该快照对买侧归因成交量与卖侧归因成交量的贡献都为零。更新级 fallback 比例可能仍有一个数值表示，但与零成交量相乘后不会进入分钟买卖成交汇总。按当前约定，如果整分钟总成交量为零，导出的分钟 `BuyRatio` 为 `0`；对应卖侧比例也应解释为 `0`，不能简单使用 `1 - BuyRatio` 得到 `1`。

**[EN]** Consecutive buy/sell runs are a separate state definition. In this research design, a no-trade snapshot represents “no new evidence,” while the contract's state may remain frozen across exchange breaks. Whether a zero-trade observation should pause, extend, or reset a run is therefore an explicit modeling choice. The current implementation should be read together with its actual reset conditions rather than inferred from the exported `BuyRatio` alone.

**[CN]** 连续买卖游程是另一套状态定义。在本研究设计中，无成交快照意味着“没有新增证据”，而交易所休市期间同一合约的状态可以被视为冻结。零成交观测究竟暂停、延长还是打断游程，属于明确的建模选择；理解当前实现时必须结合真实 reset 条件，不能只从导出的 `BuyRatio` 反推。

## 3. Missing Snapshot Policy / 缺失快照策略

**[EN]** There are two legitimate views of a missing update:

*   For reconstructing the last observable order book, forward-fill can be appropriate: no message often means no reported change.
*   For event-sample statistics, duplicating the last snapshot creates synthetic observations and changes counts, variances, ranks, and run lengths.

This engine chooses the second interpretation for aggregation: it preserves observed updates and tracks their count. This is not universally superior; it is aligned with features whose sample unit is an actual feed update. A time-weighted feature would need a different construction that explicitly weights state duration.

**[CN]** 对缺失推送存在两种都合理的解释：

*   如果目标是重建“当时最后可见盘口”，前向填充可能合理，因为没有消息通常表示没有报告变化；
*   如果目标是事件样本统计，复制上一快照会制造合成观测，并改变计数、方差、排序与游程。

本引擎在聚合时采用第二种解释：保留真实观测更新并记录数量。这并非普遍更优，而是与“真实 feed update 作为样本单位”的特征一致。若要研究时间加权状态，则应显式按照状态持续时间赋权，而不是借助重复行隐式实现。

### Sparse-minute protection / 稀疏分钟保护

The valid-snapshot threshold is a reliability mask, not an estimate of expected snapshot count. A threshold such as 20 prevents a distributional statistic from being reported on one or two auction or illiquid updates. Different fields may require different effective sample sizes; the threshold should be considered part of the feature definition and versioned accordingly. / 有效快照阈值是可靠性掩码，并不是理论快照数的估计。类似 20 的门槛用于避免在一两条集合竞价或低流动性更新上报告分布统计。不同字段可能需要不同有效样本量，因此阈值属于特征定义的一部分，应随特征版本记录。

## 4. State Lifetime Across Minutes and Sessions / 跨分钟与跨交易时段的状态生命周期

**[EN]** `auto_clear=True` represents a minute bucket: sums, maxima, counts, and other minute-local statistics reset after harvesting. `auto_clear=False` represents contract history: delay, difference, and selected run-state operators persist across minute boundaries.

**[CN]** `auto_clear=True` 表示分钟桶：分钟和、极值、计数等局部统计在收割后清空；`auto_clear=False` 表示合约历史：延迟、差分及选定游程状态跨分钟保留。

### Pauses as frozen market time / 把休市理解为冻结的市场时间

For selected features, lunch breaks, overnight gaps, and other no-data intervals are treated as pauses rather than synthetic price paths. No intermediate rows are generated, so the state immediately before the pause connects to the first real observation after it. This is intentional when the research object is “consecutive observed updates in the same contract.” It would be inappropriate for a feature defined in wall-clock time; such a feature must include elapsed time or session reset logic explicitly. / 对部分特征而言，午休、隔夜与其他无数据区间被理解为暂停，而不是合成价格路径。系统不会制造中间行，因此休市前状态会连接到恢复后的第一条真实观测。这适用于“同一合约连续真实更新”的研究对象；如果特征定义依赖自然时钟，则这种处理并不合适，必须显式加入时间间隔或交易时段重置逻辑。

### Daily cumulative fields / 日内累计字段

Cumulative volume and turnover normally reset at the trading-day boundary. Positive-clipped differences prevent a reset from being interpreted as a large negative trade. This protects aggregation but also means the first post-reset observation carries no reconstructed interval trade. Trading-date assignment for night sessions must therefore be consistent with the data vendor's convention. / 累计成交量与成交额通常在交易日边界重置。对差分取正可以避免把重置误认为巨大的负成交，但也意味着重置后的第一条观测无法恢复区间成交。夜盘交易日归属必须与数据供应商口径保持一致。

## 5. Stateful Operator Correctness / 状态算子的正确性要求

**[EN]** Stateful operators require more than formula-level unit tests. Their output depends on warm-up length, NaN transitions, circular-buffer position, resets, and partial windows. Important invariants include:

**[CN]** 状态算子不能只做公式级单元测试；其输出还依赖预热长度、NaN 转换、环形缓冲区位置、重置与未满窗口。重要不变量包括：

*   A NaN input may produce a NaN output for that update without permanently poisoning the internal EWMA state. / NaN 输入可以让当期输出为 NaN，但不应永久污染 EWMA 内部递推状态。
*   `TsFirst` on a partially filled window must return the earliest observed value, not a zero-initialized buffer slot. / `TsFirst` 在窗口未满时应返回最早真实观测，而不是零初始化缓冲区位置。
*   Delay/delta operators must define whether `window=1` means one observed update or one wall-clock interval. / Delay/Delta 必须明确 `window=1` 表示一个真实更新还是一个自然时间间隔。
*   A resettable operator must specify whether the reset observation contributes before or after reset. / Resettable 算子必须说明触发 reset 的当期观测是在重置前还是重置后参与计算。
*   Batch and streaming implementations should be compared on identical sequences containing warm-up, NaN, reset, and boundary cases. / 批量与流式实现应在包含预热、NaN、重置与边界情况的同一序列上做对照。

## 6. Symbolic Classes as Auditable Research Objects / 把符号类视为可审计研究对象

**[EN]** A `BaseTickFeature` subclass separates allocation, symbolic wiring, and harvesting. This makes the formula graph inspectable before data execution and allows the C++ engine to run the same topology for every row. The benefit is not only speed: the class is a versionable research artifact containing state lifetime, safeguards, and output definitions.

**[CN]** `BaseTickFeature` 子类把状态分配、符号连线与结果收割分开，使计算图在数据执行前即可审查，并让 C++ 引擎对每一行复用同一拓扑。其价值不只是性能：这个类本身就是可版本化研究产物，包含状态生命周期、防护条件与输出定义。

### Generated candidates require review / 生成候选必须人工审查

`miner_agent.py` retrieves related experience, requests a symbolic class, and performs structural AST checks. Passing those checks means only that the candidate has the expected class shape and contains no rejected top-level statements. It does not prove that every C++ operator combination is supported, that state semantics are correct, or that the economic hypothesis is credible. The generated class is therefore saved to the runtime-created candidate file `auto_generated_factors.py` and must be explicitly reviewed and registered; `run.py` does not auto-load it. / `miner_agent.py` 会检索相关经验、请求符号类，并执行结构性 AST 检查。通过检查只表示候选具有预期类结构且不包含被拒绝的顶层语句；它不能证明每种 C++ 算子组合都受支持，也不能证明状态语义正确或经济假设可信。因此生成类只会保存到运行时创建的候选文件 `auto_generated_factors.py`，必须经过人工审查与显式注册，`run.py` 不会自动加载。

## 7. Validation Strategy / 验证策略

### Hand-built microstructure cases / 人工微观结构案例

At minimum, the attribution and state machines should be tested on: unchanged quotes with trades, one-tick upward and downward moves, spread widening/narrowing, locked or invalid books, cumulative resets, zero-volume updates, NaN fields, sparse opening minutes, lunch breaks, night-session boundaries, and contract changes. Expected outputs should be written explicitly before running the engine. / 至少应测试：盘口不变但有成交、向上/向下一跳、价差扩大/收窄、锁盘或无效盘口、累计字段重置、零成交更新、NaN 字段、稀疏开盘分钟、午休、夜盘边界与合约切换。预期输出应在运行引擎前人工写明。

### Differential tests / 差分对照测试

A slow reference implementation in Python or vectorized NumPy is valuable even if it is not used in production. Random and adversarial sequences can be evaluated by both implementations, with tolerances defined per operator. Performance optimization should follow semantic equivalence, not substitute for it. / 即使生产环境不用，保留一个较慢的 Python 或 NumPy 参考实现仍很有价值。随机与对抗序列可以同时输入两套实现，并按算子定义误差容忍。性能优化应建立在语义等价之上，而不能替代语义验证。


