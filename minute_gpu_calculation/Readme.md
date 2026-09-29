# 🚀 High-Frequency Futures Alpha Engine: GPU-Accelerated DAG Compiler
### 期货高频因子图计算与 GPU 异步并发引擎

An ultra-low latency, highly scalable GPU-based quantitative feature engineering subsystem designed for high-frequency futures markets. This engine compiles textual LLM-generated alpha formulas into an optimized Execution DAG, aggressively manages VRAM via topological Garbage Collection, and eliminates PCIe bottlenecks through "Pin Memory" Data-Stationary architecture.

本系统是专为高频期货市场设计的极速 GPU 量化特征计算引擎。它能够将 LLM 生成的因子公式文本实时编译为优化的有向无环图 (DAG)，通过拓扑级垃圾回收机制极致榨取显存，并利用“数据不动指令动”的常驻显存架构彻底消灭 PCIe 传输瓶颈。

---

## 🔥 Core Architectures & Engineering Innovations (核心架构与工程设计)

### 1. AST-Driven DAG Compilation & CSE (基于抽象语法树的 DAG 编译与去重)
*   **[EN]** Instead of relying on slow `eval()` or procedural loops, the engine utilizes a custom Python AST `NodeTransformer` to parse alpha formulas into mathematical trees. It mathematically sorts commutative operations and enforces **Common Subexpression Elimination (CSE)**. If 50 formulas use `TsMean(close, 20)`, the engine computes it exactly once, drastically compressing the computational volume.
*   **[CN]** 摒弃了低效的 `eval()` 解析，底层利用 Python原生 `ast` 模块将大模型生成的因子公式编译为数学语法树。引擎支持交换律排序并实现了**公共子表达式消除 (CSE)**：如果 50 个复杂因子都用到了 `TsMean(close, 20)`，系统在全局只会计算一次，从编译层面对图体积进行了极度压缩。

### 2. Zero-IO "Formula Chunking" via Pin Memory (基于常驻显存的“列式分块”与零 IO 损耗)
*   **[EN]** Traditional time-series chunking suffers from massive CPU-to-GPU PCIe transfer overheads. This architecture implements a **Graph Width Partitioning (Formula Chunking)** approach. The entire year's Base Panel (high, low, close, order book depth) is loaded and pinned in the GPU VRAM at initialization. Formulas are then dispatched in batches, transforming the GPU into a pure execution state with absolutely zero I/O transfer latency during computation.
*   **[CN]** 传统的“按时间分块”会引发巨大的 CPU 到 GPU PCIe 总线拷贝开销。本系统创新性地采用了**图宽度切分 (按公式分批)** 架构。在系统启动时，将全市场整年的基础面板（量价、盘口深度等）一次性灌入并死死“钉”在 GPU 显存中 (Pin Memory)。后续计算仅在显存内部进行纯净的张量运算，彻底消灭了数据搬运损耗。

### 3. Dual-Layer Asynchronous Execution (双层异步流并发调度)
*   **[EN]** The `BatchExecutor` implements concurrency on two dimensions. Topologically, it respects the Execution Barrier of the DAG layers. Horizontally, it leverages `torch.cuda.Stream` to dispatch non-blocking operator kernels across different financial instruments simultaneously. A cross-sectional alignment bridge seamlessly merges local instrument time-series (`T_i`) into global macro time (`Global_T`) for zero-copy cross-sectional operations.
*   **[CN]** `BatchExecutor` 实现了二维并发调度：在纵向拓扑上，严格维持 DAG 执行层的同步屏障；在横向品种上，为每个期货品种分配独立的 `torch.cuda.Stream`，瞬间发射核函数 (Kernel) 实现全市场异步并行。对于截面算子，设计了动态对齐桥，将局部时间轴极速投影至全局时间，实现无缝的截面排序与标准化。

### 4. Topological Garbage Collection & Lifecycle Masking (拓扑级内存冲水与生命周期掩码)
*   **[EN]** VRAM is strictly protected. Every intermediate DAG node tracks its `out_degree`. The exact millisecond a temporary tensor is no longer needed by downstream operators, a layer-wise Garbage Collection barrier triggers `delete()` to free the memory. Furthermore, a boolean **Lifecycle Mask Matrix** is initialized at runtime to strictly eliminate "zombie prices" (unlisted/delisted contracts), preventing cross-sectional metric contamination.
*   **[CN]** 实现了极其严苛的显存保护机制。系统在编译期为每个节点计算出度 (`out_degree`)。在每一层计算结束的屏障处，出度归零的中间临时张量会被立刻安全释放 (GC 冲水)，极大降低了显存峰值。此外，底层构建了严格的**生命周期掩码矩阵 (Lifecycle Mask)**，物理级屏蔽了退市/未上市合约的僵尸数据，确保截面特征统计的绝对纯净。

### 5. Strategy-Aligned Late Masking (实战导向的延迟掩码截断)
*   **[EN]** Rather than aligning everything strictly to the dominant contract upfront—which destroys historical continuity—the engine computes all tensor features in a complete continuous universe. It applies **Late Masking** only at the final export phase, instantly slicing the active main contracts via highly optimized NumPy operations, ensuring perfect alignment with real-world trading logic.
*   **[CN]** 摒弃了在特征计算前就强行拼接主力合约的做法（这会导致时间序列指标跳空失真）。引擎在完整、平滑的全合约平行宇宙中计算所有指标，仅在最终落盘导出前，利用预先构建的路由字典执行**延迟掩码 (Late Masking)**，瞬间抽出所有对角线上的真实主力数据，完美契合实盘持仓逻辑。