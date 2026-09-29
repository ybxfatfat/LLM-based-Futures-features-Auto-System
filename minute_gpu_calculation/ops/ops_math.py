import torch
from ..ops_base import GPU_Operator

class StatelessGPUOperator(GPU_Operator):
    """
    无状态数学逻辑算子基类
    完美遵循【准则五：双模接口契约】
    由于没有时间依赖，流式 Update 可直接复用 Batch 模式的底层矩阵算术逻辑。
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._stream_val = None

    # =======================================================
    # 流式模式 (Streaming Mode) - 供实盘/C++引擎使用
    # =======================================================
    def stream_update(self, *input_vectors):
        """
        接收当前分钟的 1D 切面数据 (C_i,)。
        实盘时，这里的 inputs 可能是从 C++ 传过来的 numpy 数组或直接是 Tensor。
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        processed_args = []
        for arg in input_vectors:
            if isinstance(arg, torch.Tensor):
                processed_args.append(arg.to(torch.float32))
            elif isinstance(arg, (list, tuple)) or type(arg).__module__ == 'numpy':
                # 转为 Tensor 以便复用 PyTorch 极速内核
                processed_args.append(torch.tensor(arg, dtype=torch.float32, device=device))
            else:
                processed_args.append(float(arg))

        # 【契约复用】：直接调用子类的 _compute 核心逻辑！
        result = self._compute(*processed_args)
        
        # 遵循输出 float32 铁律
        if result.dtype == torch.bool:
            result = result.to(torch.float32)
            
        self._stream_val = result

    def stream_get_value(self):
        if self._stream_val is None:
            raise ValueError(f"[{self.__class__.__name__}] 尚未调用 stream_update 初始化！")
        return self._stream_val

    def stream_clear(self):
        self._stream_val = None


# ========================================================
# 1. 一元映射积木 (Unary Ops)
# ========================================================
class Abs(StatelessGPUOperator):
    def _compute(self, x):
        return torch.abs(x)

class Log(StatelessGPUOperator):
    def _compute(self, x):
        # 保护：防止 x <= 0 导致 log(0) 报错或产生 -inf
        return torch.log(torch.clamp(x, min=1e-8))

class Sqrt(StatelessGPUOperator):
    def _compute(self, x):
        # 保护：防止负数开根号产生 NaN
        return torch.sqrt(torch.clamp(x, min=0.0))

class Sign(StatelessGPUOperator):
    def _compute(self, x):
        return torch.sign(x)

class IndicatorMoreThanZero(StatelessGPUOperator):
    """变体：大于 0 返回 1.0，否则返回 0.0"""
    def _compute(self, x):
        return torch.where(x > 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))

class DistortLog(StatelessGPUOperator):
    """Signed-safe log transform (保留符号的对数变换)"""
    def _compute(self, x):
        # sign(x) * log(1 + abs(x))，平滑处理正负巨幅波动
        return torch.sign(x) * torch.log1p(torch.abs(x))

class DistortSqrt(StatelessGPUOperator):
    """Signed-safe sqrt transform"""
    def _compute(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))


# ========================================================
# 2. 二元对齐积木 (Binary Ops)
# ========================================================
class Add(StatelessGPUOperator):
    def _compute(self, x, y):
        # 【准则六-防墙3】：若确定 x 的引用计数为 1，可使用 x.add_(y) 做 in-place 提速。
        # 现阶段默认使用 torch.add 保证多分支 DAG 安全，由 PyTorch 缓存池自动优化显存。
        return torch.add(x, y)

class Minus(StatelessGPUOperator): # 为了与公式习惯对应，也可以直接叫 Sub
    def _compute(self, x, y):
        return torch.sub(x, y)

# 别名映射，方便 AST 解析
Sub = Minus 

class Prod(StatelessGPUOperator):
    def _compute(self, x, y):
        return torch.mul(x, y)

Mul = Prod

class Div(StatelessGPUOperator):
    def _compute(self, x, y):
        # SafeDiv：分母为 0 时返回 0.0，绝不崩溃
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.where(y != 0, x / y, zero)


# ========================================================
# 3. 特殊基础逻辑算子 (Logical Ops)
# ========================================================
class MoreThan(StatelessGPUOperator):
    def _compute(self, x, y):
        # 返回的是 bool tensor，基类会自动强转为 float32 (1.0 / 0.0)
        return torch.gt(x, y)

class LessThan(StatelessGPUOperator):
    def _compute(self, x, y):
        return torch.lt(x, y)

class Equal(StatelessGPUOperator):
    def _compute(self, x, y):
        # torch.isclose 极其严格，必须双方都是 Tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=y.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=x.device)
            
        return torch.isclose(x, y, atol=1e-5)

class IfElse(StatelessGPUOperator):
    """三元条件：condition, value_if_true, value_if_false"""
    def _compute(self, cond, x, y):
        # 由于布尔值已经被转成了 1.0 和 0.0，判断 cond > 0.5 即可
        return torch.where(cond > 0.0, x, y)

class FillNa(StatelessGPUOperator):
    """将缺失值填充为特定值 (常数或另一个张量)"""
    def _compute(self, x, fill_value):
        # 兼容 fill_value 是常量标量 或 张量矩阵 的情况
        if not isinstance(fill_value, torch.Tensor):
            fill_value = torch.tensor(fill_value, dtype=x.dtype, device=x.device)
        return torch.where(torch.isnan(x), fill_value, x)

class IsNa(StatelessGPUOperator):
    """判断是否为空值 (输出 1.0 或 0.0)"""
    def _compute(self, x):
        return torch.isnan(x)