import torch

class Expression:
    """
    所有特征节点的绝对基类，支持操作符重载。
    不仅支持加减乘除，还支持比较运算符，让 Python 对象直接变身 AST 节点。
    """
    def __add__(self, other): return Add(self, _ensure_expr(other))
    def __radd__(self, other): return Add(_ensure_expr(other), self)
    def __sub__(self, other): return Sub(self, _ensure_expr(other))
    def __rsub__(self, other): return Sub(_ensure_expr(other), self)
    def __mul__(self, other): return Mul(self, _ensure_expr(other))
    def __rmul__(self, other): return Mul(_ensure_expr(other), self)
    def __truediv__(self, other): return Div(self, _ensure_expr(other))
    def __rtruediv__(self, other): return Div(_ensure_expr(other), self)
    
    # 【新增】：支持负号 (一元减法) -> 自动转为乘 -1
    def __neg__(self): return Mul(Constant(-1.0), self)
    
    # 【新增】：支持逻辑判断符重载
    def __gt__(self, other): return MoreThan(self, _ensure_expr(other))
    def __lt__(self, other): return LessThan(self, _ensure_expr(other))
    def __ge__(self, other): return MoreThan(self, _ensure_expr(other)) # 暂用 MoreThan 近似替代，或新增 MoreEq
    def __le__(self, other): return LessThan(self, _ensure_expr(other))
    def __eq__(self, other): return Equal(self, _ensure_expr(other))


def _ensure_expr(val):
    if isinstance(val, Expression): return val
    elif isinstance(val, (int, float)): return Constant(val)
    elif isinstance(val, str): return Feature(val)
    else: raise TypeError(f"不支持的输入类型: {type(val)}")


class Feature(Expression):
    def __init__(self, name): self.name = name
    def __repr__(self): return f"${self.name}"

class Constant(Expression):
    def __init__(self, value): self.value = value
    def __repr__(self): return str(self.value)

class Operator(Expression):
    def __init__(self, *args):
        self.args = [_ensure_expr(arg) for arg in args]
    def __repr__(self):
        arg_strs = ", ".join(repr(a) for a in self.args)
        return f"{self.__class__.__name__}({arg_strs})"


class GPU_Operator(Operator):
    """
    [Architecture Design]: 双模算子接口契约
    强制所有子类必须同时提供回测 (Batch) 与实盘 (Stream, 逐行更新) 两套实现。
    """
    
    # ========================================================
    # 模式 A：批量回测接口 (Batch Mode)
    # ========================================================
    def batch_compute(self, *args):
        # 1. 提取张量，确保在 GPU 上
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if not tensors:
            raise ValueError(f"[{self.__class__.__name__}] 计算需至少包含一个张量！")
            
        device = tensors[0].device
        global_shape = tensors[0].shape
        
        # 2. 格式化参数：标量保持，张量强转 float32 并执行契约形状校验
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                assert arg.shape == global_shape, f"形状不匹配: {arg.shape} vs {global_shape}"
                processed_args.append(arg.to(dtype=torch.float32, device=device))
            else:
                processed_args.append(float(arg.value) if isinstance(arg, Constant) else float(arg))
                
        # 3. 核心计算 (非阻塞，异步压入 CUDA Stream)
        # [Memory Optimization]: 子类在 _compute 中应尽可能使用 in-place 操作 (如 x.add_(y)) 以降低显存峰值
        result = self._compute(*processed_args)
        
        # 4. 后置校验，确保吐出的绝对是 float32 (T_i, C_i) 形状的张量
        if not isinstance(result, torch.Tensor):
            result = torch.full(global_shape, result, dtype=torch.float32, device=device)
        if result.dtype == torch.bool:
            result = result.to(torch.float32)
            
        return result

    def _compute(self, *args):
        """留给子类的 GPU Tensor 并发实现逻辑"""
        raise NotImplementedError

    # ========================================================
    # 模式 B：流式实盘接口 (Streaming Mode)
    # ========================================================
    def stream_update(self, *input_vectors):
        """
        接收当前分钟的 1D 切面数据 (C_i,) 更新 Ring Buffer
        实盘时，C++ 引擎或 Python StreamExecutor 会调用此方法
        """
        raise NotImplementedError(f"[{self.__class__.__name__}] 未实现实盘防未来递推 stream_update()！")

    def stream_get_value(self):
        """O(1) 获取当前最新计算结果"""
        raise NotImplementedError(f"[{self.__class__.__name__}] 未实现实盘信号获取 stream_get_value()！")

    def stream_clear(self):
        """实盘重置"""
        raise NotImplementedError(f"[{self.__class__.__name__}] 未实现实盘重置 stream_clear()！")


# 解决循环引用，在此声明这些基础类 (具体逻辑将在 ops/ops_math.py 中实现)
class Add(GPU_Operator): pass
class Sub(GPU_Operator): pass
class Mul(GPU_Operator): pass
class Div(GPU_Operator): pass
class MoreThan(GPU_Operator): pass
class LessThan(GPU_Operator): pass
class Equal(GPU_Operator): pass