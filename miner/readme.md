# 🧬 LLM-GA Futures Alpha Miner / 大模型分层遗传因子挖掘系统

**[EN]** An end-to-end, ultra-high-frequency quantitative feature discovery pipeline. It synergizes the logical reasoning of Large Language Models (LLMs) with a PyTorch-based Directed Acyclic Graph (DAG) execution engine to automatically evolve, evaluate, and extract structural Alphas and Market Regimes from Level-2 order book data.

**[CN]** 本项目是一个端到端的高频量化特征挖掘流水线。它摒弃了传统大模型“盲目生成代码”的低效范式，将 LLM 的逻辑推理能力与底层高频 C++ DAG 计算图深度融合，构建了一套包含5层严格工序、具备“探索与利用(E&E)”博弈、以及带有量化铁律约束的智能因子进化树。

![LLM-GA 架构图](./figure.png)

---

## 🔥 Core Architecture & Philosophy / 核心架构哲学

### 1. The 5-Layer Genetic Pipeline / 5层定向基因突变架构
**[EN]** Instead of blind formula generation, the LLM progresses through a strictly scaffolded 5-step pipeline.
**[CN]** 系统禁止大模型一口气写完所有逻辑，将其拆解为 5 道不可逆的靶向流水线：
*   **[L1] Raw Ingredients (基础原料层)**: Injects external foundation features as templates, forcing the LLM to construct physically meaningful genes. / 注入外部高级特征作为灵感模板，强制大模型构造具备真实微观物理意义的初始基因。
*   **[L2] Relativization (截面相对化层)**: Forces the application of scaling operators to strip away absolute price/volume differences across diverse assets. / 强制施加规模剥离算子，消除跨品种间的绝对价格与流动性差异，保证截面可比性。
*   **[L3] Distribution Transformation (分布变换层)**: Simulates human visual diagnostics. Feeds 10-quantile return profiles to the LLM to mathematically correct U-shaped or fat-tailed anomalies. / 模拟人类“阅图诊断”。向 LLM 喂入特征的10等分未来收益率表现，使用算子强制修复非线性特征，重塑单调性。
*   **[L4] Temporal & Interaction (复杂时序交互层)**: The core deep-water zone. Injects complex compound operators (e.g., EWMAC) to inspire deep-nested momentum/reversal features. / 核心深水区。注入高阶复合算子启发 LLM 自创深度嵌套公式，自带最多 3 轮递归嵌套 (Nesting) 机制。
*   **[L5] Smoothing & Window Sweeping (降噪与网格爆破层)**: Enforces exponential smoothing and mandates the generation of 5-8 distinct time-window variants for grid sweeping. / 强制施加平滑去噪，并针对同一因子强行衍生 5-8 组完全不同数量级的时间窗口，探索 Alpha 存活边界。

### 2. Exploration vs. Exploitation / 蓄水池的 E&E 抽样博弈
**[EN]** The system dynamically balances deep exploitation of new genes and broad exploration of dormant genes across persistent pools.
**[CN]** 系统在层级间维护了持久化的基因池，在生成新一代 Prompt 时进行精妙的抽样平衡：
*   **Exploitation (深度利用)**: Extracts the freshly generated "new generation" genes to polish promising branches continuously. / 强制提取上一层刚刚生成的“新生代”基因，顺藤摸瓜将潜力分支一路打磨到底。
*   **Exploration (广度探索)**: Randomly revives dormant genes from massive historical pools to escape local optima. / 随机从历史沉淀池中唤醒“沉睡基因”重新变异，防止族群陷入局部最优。

### 3. LLM-As-A-Judge & Domain Axioms / 量化铁律与大模型裁判
**[EN]** Pre-loads financial engineering axioms to prevent data leakage, and uses the LLM as a strict judge to prune redundant factors based on robust metrics.
**[CN]** 预埋昂贵的实盘教训（量化铁律）作为防爆盾。当基因池溢出时，LLM 化身裁判，基于 `Period IC` 与 `Full_IC` 差异、非线性指标无情斩杀同质化噪声因子。

### 4. Symbolic Feature Discovery for DL / 面向深度学习的符号降维
**[EN]** Parses the AST lineage of the "Alpha Hall of Fame" to extract highly recursive sub-graphs (Roots) for use as high-SNR inputs in time-series Deep Learning models.
**[CN]** 利用 Python AST 语法树拆解“Alpha 名人堂”，提纯出高频共现的子图公式（Roots），直接作为 Transformer 或 LSTM 模型的极高信噪比输入特征。

---

## 🏗️ System Modules / 核心组件说明

*   **`evolutionary_miner.py`**
    **[EN]** The orchestrator of the 5-layer evolutionary network. Features dynamic AST constant folding to guarantee 100% execution safety on the GPU DAG engine.
    **[CN]** 流水线“大脑”。调度 5 层遗传网络，集成 AST 常数自动折叠防崩器，确保生成的公式在底层 GPU 引擎上 100% 语法安全。
*   **`evaluator.py`**
    **[EN]** The CPU/GPU Vectorized Evaluation Engine utilizing 2000-minute non-overlapping rolling windows (Period IC) and 10-quantile Monotonicity checks.
    **[CN]** 极速向量化评测引擎。摒弃粗糙的 Daily IC，采用 2000 分钟无重叠窗口 (Period IC)，结合 10 等分单调性进行最严苛的抗周期压测。
*   **`state_miner.py`**
    **[EN]** Market Regime Discovery. Reverses the Alpha paradigm by tasking the LLM to hypothesize micro-structure environments from actual winning/losing trade snapshots.
    **[CN]** 微观市场环境挖掘器。逆向抽出基准策略赚/亏钱的盘口快照，让大模型充当“侦探”生成刻画特定市场环境（如流动性真空）的状态张量。
*   **`parse_factors.py`**
    **[EN]** AST Lineage Parser. Extracts high-scoring Root components and generates interactive ECharts HTML graphs showing genetic diversity.
    **[CN]** AST 血缘解析器。拆解高分 Root 组件，生成带有强弱节点（基于 Max Descendant IR）的 ECharts 交互式关系网大屏。
*   **`manage_libraries.py`**
    **[EN]** Interactive CLI. Allows Quants to describe ideas in natural language, which the LLM translates into strictly validated JSON DAG operators.
    **[CN]** 人机交互 CLI。允许研究员用大白话描述灵感，底层调用 LLM 翻译成严格符合语法的 JSON 配置，直接注册进特征弹药库。

---

## 🚀 Quick Start / 快速开始

### 1. Configuration / 环境配置
**[EN]** Ensure your environment variables are set for the standard LLM client:
**[CN]** 确保环境中已配置原生的 LLM 接口凭据：
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="[https://api.openai.com/v1](https://api.openai.com/v1)" 
export LLM_MODEL_NAME="gpt-4o"