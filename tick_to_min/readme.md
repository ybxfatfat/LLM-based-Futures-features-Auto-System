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
    The alignment tool using PyArrow. Integrates high-frequency features securely into the structured daily `base_panel` using column patching strategies. (基于 PyArrow 的日频宽表智能对齐与合并工具)。
*   **`prompts_config.py`**: 
    Defines the LLM's C++ DAG syntax constraints and microstructure-reasoning questions, such as identifying the proposed counterparty mechanism. (定义大模型使用的 C++ 计算图语法约束与微观结构推理问题，例如要求说明候选机制中的对手方)。
*   **`miner_agent.py` & `auto_generated_factors.py`**: 
    The conversational candidate generator. It translates human ideas into symbolic Python classes, performs static AST checks, and stores accepted candidates in `auto_generated_factors.py`. Candidates require human review and explicit registration before execution. (对话式候选生成器：把自然语言灵感翻译为符号化 Python 类，经 AST 静态校验后写入 `auto_generated_factors.py`；候选必须经过人工审查并显式注册后才会执行)。
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

## 🚀 Quick Start / 快速启动

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
