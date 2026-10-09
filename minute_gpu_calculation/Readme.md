# 🚀 High-Frequency Futures Alpha Engine: GPU-Accelerated DAG Compiler
### 期货高频因子图计算与 GPU 异步并发引擎

A GPU-based quantitative feature-engineering subsystem for minute-level futures research. It compiles textual alpha formulas into an execution DAG, reuses common subexpressions, releases eligible intermediates by graph lifetime, and keeps base tensors resident on the GPU across formula batches to reduce repeated transfers.

本系统是一套面向分钟频期货研究的 GPU 特征计算引擎。它将文本因子公式编译为有向无环图（DAG），复用公共子表达式，按图生命周期释放符合条件的中间张量，并让基础张量在公式批次间驻留 GPU，以减少重复数据搬运。

---

## 🔥 Core Architectures & Engineering Innovations (核心架构与工程设计)

### 1. AST-Driven DAG Compilation & CSE (基于抽象语法树的 DAG 编译与去重)
*   **[EN]** The engine uses a custom Python AST `NodeTransformer` to parse alpha formulas into mathematical trees instead of evaluating arbitrary formula strings with `eval()`. It canonicalizes supported commutative operations and applies **Common Subexpression Elimination (CSE)**. If 50 formulas contain the same canonical `TsMean(close, 20)` node, that node is represented once in the shared graph.
*   **[CN]** 引擎使用 Python `ast` 与自定义 `NodeTransformer` 将因子公式解析为数学语法树，而不是直接用 `eval()` 执行任意字符串。它会规范化受支持的交换运算并执行**公共子表达式消除（CSE）**；如果 50 个公式包含同一个规范化后的 `TsMean(close, 20)` 节点，共享图中只保留一份该节点。

### 2. Data-Stationary Formula Chunking (基于常驻显存的公式分块)
*   **[EN]** Rather than repeatedly chunking the time axis, the engine partitions the graph by formula width. Base-panel tensors are transferred to GPU memory during initialization and reused across formula batches, reducing repeated PCIe traffic during the compute phase. Initial loading, final export, and any explicit device conversion still incur I/O or copies.
*   **[CN]** 引擎不反复切分时间轴，而是按公式宽度划分计算图。基础面板张量在初始化阶段传入 GPU，并在多个公式批次间复用，从而减少计算阶段的重复 PCIe 传输；初始加载、最终导出及显式设备转换仍然会发生 I/O 或复制。

### 3. Dual-Layer Asynchronous Execution (双层异步流并发调度)
*   **[EN]** The `BatchExecutor` coordinates two dimensions of concurrency. It respects DAG-layer barriers while using `torch.cuda.Stream` to dispatch eligible kernels for different instruments. A cross-sectional alignment bridge maps local instrument time axes (`T_i`) to a shared global axis (`Global_T`) before cross-sectional operators run.
*   **[CN]** `BatchExecutor` 协调两个维度的并发：纵向遵守 DAG 层间屏障，横向使用 `torch.cuda.Stream` 调度不同品种中可并行的核函数。截面算子执行前，动态对齐桥会把各品种局部时间轴（`T_i`）映射到共享全局时间轴（`Global_T`）。

### 4. Topological Garbage Collection & Lifecycle Masking (拓扑级内存冲水与生命周期掩码)
*   **[EN]** Every intermediate DAG node tracks its `out_degree`. At a layer barrier, an intermediate whose downstream consumers are complete becomes eligible for deletion, which helps limit peak VRAM usage. A boolean **Lifecycle Mask Matrix** excludes observations outside each contract's listed-data interval from cross-sectional calculations.
*   **[CN]** 每个 DAG 中间节点都会跟踪其 `out_degree`。在层间屏障处，下游消费者均已完成的中间张量会进入可释放状态，以控制显存峰值。布尔型**生命周期掩码矩阵（Lifecycle Mask）**则用于排除合约有数据生命周期之外的观测，减少截面统计污染。

### 5. Strategy-Aligned Late Masking (实战导向的延迟掩码截断)
*   **[EN]** Instead of stitching a dominant-contract series before feature calculation, the engine computes each contract's tensors first and applies **Late Masking** during export using the configured main-contract routing. This preserves contract-local history during calculation while making the final panel follow the selected routing rule.
*   **[CN]** 引擎不会在特征计算前先拼接主力连续序列，而是先在各合约自身序列上计算张量，再在导出阶段依据既定主力合约路由执行**延迟掩码（Late Masking）**。这样既保留计算时的合约内历史，也使最终面板遵循所选主力路由规则。
