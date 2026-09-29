# High-Frequency Futures Feature Engine / 高频期货特征极速计算引擎

A high-performance, hybrid C++/Python framework designed for quantitative trading research. It processes ultra-high-frequency Level-2 snapshot data (Tick data) into minute-level aggregated features utilizing a Zero-Copy Directed Acyclic Graph (DAG) computational engine.
本项目是一个专为量化交易研究设计的高性能 C++/Python 混合架构框架。它利用零拷贝的有向无环图 (DAG) 计算引擎，将超高频 L2 快照 (Tick 数据) 极速聚合并降频为分钟级特征宽表。

## 🎯 Core Paradigms / 核心架构范式

*   **Zero-Copy Memory & C++ Native Execution (零拷贝内存与 C++ 原生计算)**: 
    *   **[EN]** Python acts purely as the orchestrator, while C++ is the execution workhorse. Massive Tick data is loaded into contiguous memory via Python (NumPy). PyBind11 simply passes the raw memory pointers to the C++ engine. The heavy lifting—iterating over millions of ticks—happens entirely inside C++, bypassing the Python GIL and eliminating any serialization or data-copy overhead.
    *   **[CN]** Python 仅扮演“指挥官”角色，C++ 才是真正的“干活主力”。海量的 Tick 数据在 Python 端被加载为连续内存 (NumPy 数组) 后，PyBind11 仅将裸内存指针 (Raw Pointers) 传递给 C++ 底层。真正遍历数百万行 Tick 的高频计算完全在 C++ 内部闭环执行，彻底绕过 Python GIL 锁，真正实现了数据的**零拷贝 (Zero-Copy)** 和零传输损耗。
*   **Symbolic Tracing Graph (符号化计算图追踪)**: 
    *   **[EN]** In Python, you write equations that look like standard math, but they actually act as blueprints. Python "draws" the DAG, and the C++ engine dynamically allocates memory and processes the data pipeline based on this graph. 
    *   **[CN]** 在 Python 端编写的代码并不执行具体的数值计算，而是充当“图纸”。Python 负责“画图” (将公式解析并连线)，底层 C++ 引擎根据这张图纸动态分配算子并执行流水线运算。
*   **Stateful Time-Series Ops (状态化极速算子)**: Built-in $O(1)$ sliding window moments, Monotonic Deques for extremums, and $O(\log W)$ Red-Black Trees for quantiles calculation. (内置 $O(1)$ 滑动窗口高阶矩、用于极值的单调队列，以及 $O(\log W)$ 红黑树极速分位数计算)。

## 📖 How It Works: The Active Buy/Sell Decomposition Example / 运作机制：以主买主卖分解为例

To understand how the Python routing constructs the C++ execution graph, here is a core component from our production system: The **Active Buy/Sell Decomposition & Run-length Feature (主买主卖分解与游程特征)**.

It demonstrates the strict 3-step lifecycle of defining a feature:

```python
class HighFreqMinuteFeatures(BaseTickFeature):
    def __init__(self):
        super().__init__()
        # STEP 1: Node Allocation (节点注册)
        # Register C++ stateful operators. 
        # auto_clear=False keeps state across ticks; auto_clear=True resets at minute boundaries.
        self.tick_vol_diff = self.register_state_op(ce.TsDelta(window=1, min_periods=1), auto_clear=False)
        self.delay_mid = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)
        
        self.min_vol_sum = self.register_state_op(ce.TsSum(window=0, min_periods=1))
        self.min_vbuy_sum = self.register_state_op(ce.TsSum(window=0, min_periods=1))
        
        self.run_up_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_up_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)
        self.min_max_up_count = self.register_state_op(ce.TsMax(window=0, min_periods=1))

    def update(self, snp):
        # STEP 2: Symbolic Wiring (符号追踪连线)
        # NO ACTUAL COMPUTATION HAPPENS HERE. We are passing C++ pointers to build the DAG.
        c_0 = ce.ConstNode(0.0) # Strict Constant Wrapping Rule
        c_1 = ce.ConstNode(1.0)
        c_2 = ce.ConstNode(2.0)
        
        # Microstructure Math -> Generates AST nodes
        mid = (snp.ask_1_prc + snp.bid_1_prc) / c_2
        self.delay_mid.set_input(mid)
        
        self.tick_vol_diff.set_input(snp.cum_volume)
        dv = ce.IfElse((self.tick_vol_diff > c_0), self.tick_vol_diff, c_0)
        self.min_vol_sum.set_input(dv)
        
        # High-Frequency VWAP & Orderbook Imbalance Routing
        is_up = (mid > self.delay_mid)
        up_count_reset_cond = ce.IfElse(is_up, c_0, c_1)
        
        # Wiring the Resetable Wrapper for Consecutive Up-Ticks (游程连线)
        self.run_up_inner.set_input(c_1)
        self.run_up_wrapper.set_inputs(self.run_up_inner, up_count_reset_cond)
        self.min_max_up_count.set_input(self.run_up_wrapper)

    def get_value(self) -> dict:
        # STEP 3: Harvesting (分钟边界收割)
        # Defines the output structure. Triggered by C++ at every minute boundary.
        c_0 = ce.ConstNode(0.0)
        
        # Zero-Division Protection Rule
        raw_buy_ratio = ce.IfElse((self.min_vol_sum > c_0), self.min_vbuy_sum / self.min_vol_sum, c_0)

        return {
            'BuyRatio': raw_buy_ratio,
            'MaxConsecUpTicks': self.min_max_up_count
        }
```

### Anatomy of the Class (类结构解析):
1. **`__init__` (Allocate)**: Declares the required C++ memory nodes. For example, `ce.TsResetable()` initializes an isolated C++ memory block for capturing consecutive trends without writing slow `for-loops` in Python. (声明 C++ 底层状态机。例如申请一个支持条件打断的游程算子 `TsResetable`)。
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
    The "Soul" of the LLM. Contains strict C++ DAG syntactic rules and Micro-structural Economic Intuition constraints (e.g., "Who is the counterparty loser?"). (大模型系统指令，包含极严苛的 C++ 计算图语法铁律与微观博弈灵魂拷问)。
*   **`miner_agent.py` & `auto_generated_factors.py`**: 
    The conversational Agent entry point. Translates human ideas into C++ symbolic Python classes dynamically injected into `auto_generated_factors.py`. (全自动挖掘机，将大白话灵感翻译为底层算子并自动注入执行库)。
*   **`rag_vector_store.py`**: 
    A Dual-Payload RAG Database. It embeds the *economic hypothesis* for semantic similarity search, while storing the C++ code and performance metrics as payload metadata. (双载荷 RAG 向量知识库。仅对经济学逻辑进行向量化，完美规避代码语法干扰)。
*   **`eval_and_reflect.py`**: 
    The Post-mortem Evaluator. Calculates Top vs Bottom decile spreads (Q10-Q1), prompts the LLM for deep reflection, and writes the experience back to the RAG KB. (高频验尸官与反思引擎，评测真实分组收益差并将经验打入知识库闭环)。

## 🤖 Autonomous Agent System (LLM-RAG Feature Miner) / 自动化大模型特征挖掘系统

The engine is supercharged with a Multi-Agent system powered by a **Retrieval-Augmented Generation (RAG)** loop. It bridges the gap between natural language economic intuition and ultra-high-frequency C++ execution.
本项目搭载了基于 RAG 闭环的 Multi-Agent 智能体系统。它彻底打通了“自然语言描述经济学直觉”到“极高频 C++ 算子落地”的次元壁。

1. **Dual-Payload Semantic Retrieval (双载荷语义检索)**: When a new feature idea is proposed, the Agent queries the local RAG Vector Database. Instead of embedding code syntax (which causes noise), it embeds only the *economic mechanism*. If a similar idea failed previously due to specific microstructure noise, the LLM is warned before generating code. (系统仅对自然语言假设进行 Embedding。输入新灵感时，大模型会自动查阅历史相似因子库，精准避开由于微观结构噪音导致的“历史天坑”)。
2. **Strict Symbolic Code Generation (严苛的符号化代码生成)**: The `Coder Agent` strictly follows Zero-Division protections, Constant Wrapping (`ce.ConstNode`), and Small-Sample Masking rules to generate bug-free DAG logic. (代码生成器严格遵守防零除、常数包装与小样本遮罩铁律，确保生成的算子绝对不会导致底层 C++ 崩溃)。
3. **Dynamic Graph Injection (动态图注入与反射)**: `run.py` utilizes Python's `inspect` module to dynamically discover and mount any newly generated LLM factors into the C++ computation pipeline without manual intervention. (执行引擎利用 Python 动态反射机制，自动抓取大模型写出的新类并挂载入计算图，实现 0 代码复制粘贴修改)。
4. **Self-Reflection Loop (自动反思闭环)**: After computation, `eval_and_reflect.py` grades the feature based on Decile Spread (Q10-Q1) rather than generic IC. The LLM generates a diagnostic reflection, which is looped back into the RAG database. (验尸官脚本基于更具实盘意义的 Q10-Q1 分组收益差进行评判，大模型自动生成反思报告并打入知识库闭环，实现越挖越聪明的自我进化)。

## 🚀 Quick Start / 快速启动

### 1. Compile the C++ Engine / 编译底座
```bash
./build.sh
```

### 1.5 (Optional) Run the LLM Agent to Mine New Features / 启动大模型智能挖掘
Use natural language to instruct the agent to build new C++ L2 operators.
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