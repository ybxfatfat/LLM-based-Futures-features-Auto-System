# 🧬 LLM-GA Futures Alpha Miner / 大模型分层遗传因子挖掘系统

**[EN]** A research pipeline for minute-level feature discovery from Level-2-derived base panels. It combines LLM-guided hypothesis generation with a PyTorch-based Directed Acyclic Graph (DAG) execution engine to propose, evaluate, and organize symbolic Alphas and market-regime candidates.

**[CN]** 本项目是一套基于 L2 衍生基础面板的分钟频特征研究流水线。它将 LLM 引导的假设生成与 PyTorch DAG 计算图结合，通过 5 层分工、探索与利用（E&E）抽样及量化约束，提出、评估并组织符号化 Alpha 与市场状态候选。

![LLM-GA 架构图](./figure.png)

---

## 🔥 Core Architecture & Philosophy / 核心架构哲学

### 1. The 5-Layer Genetic Pipeline / 5层定向基因突变架构
**[EN]** Instead of blind formula generation, the LLM progresses through a strictly scaffolded 5-step pipeline.
**[CN]** 系统不让大模型一次性写完所有逻辑，而是将研究过程拆为 5 个定向阶段：
*   **[L1] Raw Ingredients (基础原料层)**: Supplies foundation features as interpretable starting materials and asks the LLM to state the proposed mechanism. / 将基础特征作为可解释的起始材料，并要求 LLM 说明候选机制。
*   **[L2] Relativization (截面相对化层)**: Applies justified scaling operations to reduce price- and volume-scale differences across assets. / 使用有经济解释的缩放算子，降低跨品种价格与流动性量级差异。
*   **[L3] Distribution Transformation (分布变换层)**: Feeds 10-quantile return profiles to the LLM to diagnose U-shaped, fat-tailed, or non-monotonic responses and propose transformations. / 向 LLM 提供特征的 10 分组收益表现，用于诊断 U 型、厚尾或非单调响应并提出变换。
*   **[L4] Temporal & Interaction (复杂时序交互层)**: Exposes compound operators such as EWMAC to support nested momentum, reversal, and interaction hypotheses, with recursion depth capped by the prompt. / 提供 EWMAC 等复合算子，用于构造嵌套的动量、反转与交互假设，并通过 Prompt 限制递归深度。
*   **[L5] Smoothing & Window Sweeping (降噪与窗口扫描层)**: Adds smoothing and generates 5-8 time-window variants so the same mechanism can be compared across different horizons. / 增加平滑处理并生成 5-8 组时间窗口变体，用于比较同一机制在不同周期下的表现。

### 2. Exploration vs. Exploitation / 蓄水池的 E&E 抽样博弈
**[EN]** The system dynamically balances deep exploitation of new genes and broad exploration of dormant genes across persistent pools.
**[CN]** 系统在层级间维护持久化基因池，在生成新一代 Prompt 时进行抽样平衡：
*   **Exploitation (深度利用)**: Extracts the freshly generated "new generation" genes to polish promising branches continuously. / 强制提取上一层刚刚生成的“新生代”基因，顺藤摸瓜将潜力分支一路打磨到底。
*   **Exploration (广度探索)**: Randomly revives dormant genes from historical pools to reduce repeated convergence on the same local branches. / 随机从历史沉淀池中唤醒“沉睡基因”重新变异，减少研究反复收敛到同一局部分支。

### 3. LLM-As-A-Reviewer & Domain Constraints / 领域约束与大模型审查
**[EN]** Encodes research constraints intended to reduce leakage risk and uses the LLM as one review signal when pruning redundant candidates from recorded metrics. The LLM is not treated as a proof of validity.
**[CN]** 将研究约束写入 Prompt 以降低泄漏风险，并在候选去冗余时把 LLM 判断作为基于已记录指标的一项审查信号；LLM 判断本身不等于有效性证明。

### 4. Symbolic Feature Discovery for DL / 面向深度学习的符号降维
**[EN]** The auxiliary `parse_factors.py` script analyzes formula lineage and recurring subexpressions as candidate roots for simpler downstream inputs. It is exploratory tooling; recurrence alone does not establish signal quality.
**[CN]** 辅助脚本 `parse_factors.py` 分析公式血缘与重复子表达式，将其作为更简洁下游输入的候选根节点。这属于探索性工具，重复出现本身并不能证明信号质量。

---

## 🏗️ System Modules / 核心组件说明

*   **`evolutionary_miner.py`**
    **[EN]** The orchestrator of the 5-layer evolutionary network. It normalizes formulas with AST-based rewrites and rejects candidates that fail parsing or execution checks; these checks reduce common failures but do not guarantee semantic correctness.
    **[CN]** 流水线调度器。它通过 AST 重写规范化公式，并拒绝未通过解析或执行检查的候选；这些检查能减少常见故障，但不保证公式在语义上正确。
*   **`evaluator.py`**
    **[EN]** The CPU/GPU Vectorized Evaluation Engine utilizing 2000-minute non-overlapping rolling windows (Period IC) and 10-quantile Monotonicity checks.
    **[CN]** 向量化评测引擎。采用 2000 分钟不重叠窗口（Period IC），并结合 10 分组结果检查单调性与阶段稳定性。
*   **`state_miner.py`**
    **[EN]** Market Regime Discovery. Reverses the Alpha paradigm by tasking the LLM to hypothesize micro-structure environments from actual winning/losing trade snapshots.
    **[CN]** 微观市场环境挖掘器。逆向抽出基准策略赚/亏钱的盘口快照，让大模型充当“侦探”生成刻画特定市场环境（如流动性真空）的状态张量。
*   **`parse_factors.py`**
    **[EN]** AST Lineage Parser. Extracts high-scoring Root components and generates interactive ECharts HTML graphs showing genetic diversity.
    **[CN]** AST 血缘解析器。拆解高分 Root 组件，生成带有强弱节点（基于 Max Descendant IR）的 ECharts 交互式关系网大屏。
*   **`manage_libraries.py`**
    **[EN]** Interactive CLI. Allows researchers to describe ideas in natural language, then validates the LLM-produced JSON structure before writing it to the configured library.
    **[CN]** 人机交互 CLI。允许研究员用自然语言描述灵感，并在写入配置库前校验 LLM 生成的 JSON 结构。

---

## 🚀 Quick Start / 快速开始

### 1. Configuration / 环境配置
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
