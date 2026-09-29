## 🛠 2. 算子函数库 (Operator Library)

公式支持直接使用 Python 原生运算符：`+`, `-`, `*`, `/`, `>`, `<`, `==`，以及一元负号 `-x`。

### 一元数学算子 (Unary Ops)
* `Abs(x)`: 绝对值,使用绝对值的时候要绝对注意，这是一个会抹消掉内层变量的方向意义的操作，而我们构造的特征是多空alpha很多时候都强调特征数值的正负如果分别对应值得多空/涨跌的场景那一顶小谨慎使用Abs()
* `Log(x)`: 自然对数 (自带防 <=0 崩溃处理)
* `Sqrt(x)`: 平方根
* `Sign(x)`: 符号函数 (1, 0, -1)
* `IndicatorMoreThanZero(x)`: 若 x > 0 返回 1.0，否则返回 0.0
* `DistortLog(x)`: 保留符号的对数 `sign(x) * log(1 + abs(x))`
* `DistortSqrt(x)`: 保留符号的平方根 `sign(x) * sqrt(abs(x))`

### 特殊逻辑算子 (Logical Ops)
* `IfElse(cond, x, y)`: 核心条件算子。若 `cond` 为真 ( > 0 )，返回 `x`，否则返回 `y`。千万不要用 Python 的 if-else 语句。
* `FillNa(x, val)`: 若 `x` 是 NaN，用 `val` (可为常数或另一公式) 填补。
* `IsNa(x)`: 判断是否为空值，是返回 1.0，否返回 0.0。

### 时序基础与位移算子 (Time-Series Basic)
*所有 `window` 参数的单位均为分钟。*
* `TsDelay(x, window)`: 获取 `window` 分钟前的值 $x_{t-W}$。
* `TsDelta(x, window)`: 差分 $x_t - x_{t-W}$。
* `TsPct(x, window)`: 收益率近似 $Log(x_t / x_{t-W})$。
* `TsFirst(x, window)`: 窗口内的第一个有效值。
* `TsLast(x, window)`: 窗口内的最后一个有效值（本质是自己，附带有效性过滤）。

### 时序线性聚合 (TS Linear - 极速 O(1))
* `TsSum(x, window)`: 滚动求和。
* `TsMean(x, window)`: 滚动均值。
* `TsStd(x, window)`: 滚动标准差。
* `TsZscore(x, window)`: 时序标准化 $(x - \mu) / \sigma$。

### 时序非线性聚合 (TS Non-Linear)
* `TsMax(x, window)` / `TsMin(x, window)`: 滚动最大/最小值。
* `TsMaxDiff(x, window)`: 当前值减去窗口最大值。
* `TsMinDiff(x, window)`: 当前值减去窗口最小值。
* `TsArgMax(x, window)` / `TsArgMin(x, window)`: 最大/最小值所在相对位置 (输出 0~1 的比例，1 代表最远，0 代表当前)。
* `TsMedian(x, window)`: 滚动中位数。
* `TsQuantile(x, window, q)`: 滚动分位数 (q 在 0~1 之间，如 0.5 为中位数)。
* `TsMad(x, window)`: 绝对中位差。
* `TsRank(x, window)`: 当前值在窗口内的百分位排名 (输出 0~1)。

### 高阶统计算子 (TS High-Order)
* `TsSkewness(x, window)`: 滚动偏度 (Fisher-Pearson 无偏)。
* `TsKurtosis(x, window)`: 滚动峰度 (Excess Kurtosis)。
* `TsEntropy(x, window)`: 滚动信息熵。
* `TsCv(x, window)`: 变异系数 (Std / Mean)。
* `TsIr(x, window)`: 信息比率 (Mean / Std)。
* `EWMA(x, halflife)`: 指数加权移动平均，参数为半衰期 `halflife`。
* `TsDecay(x, window)`: 线性权重衰减移动平均 (越近权重越大)。

### 二元关系时序算子 (TS Binary)
* `TsCov(x, y, window)`: x 与 y 的滚动协方差。
* `TsCorr(x, y, window)`: x 与 y 的滚动皮尔逊相关系数。
* `TsFacNeutralCoef(x, y, window)`: 滚动回归系数 Beta (以 y 为因变量，x 为自变量)。
* `TsFacNeutralRes(x, y, window)`: 滚动回归残差 (去量纲/中性化后的异象信息)。
