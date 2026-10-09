import torch
import math
from ..ops_base import GPU_Operator

# ======================================================================
# 核心状态机基类：实现【准则五：双模契约】
# ======================================================================
class StatefulGPUOperator(GPU_Operator):
    """
    带状态的时序算子基类
    为所有时序算子自动提供实盘流式 (Streaming) 的防未来递推能力。
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._buffers = None    
        self._ptr = 0           
        self._window = None     
        self._stream_val = None 

    def stream_update(self, *args):
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        scalars = [arg for arg in args if not isinstance(arg, torch.Tensor)]
        
        if not tensors:
            raise ValueError(f"[{self.__class__.__name__}] Stream 更新缺少张量！")
            
        if self._buffers is None:
            self._window = int(scalars[0]) if scalars else 2 
            self._buffers = [torch.full((self._window, t.shape[0]), float('nan'), 
                                      dtype=torch.float32, device=t.device) for t in tensors]

        for i, t in enumerate(tensors):
            self._buffers[i][self._ptr] = t.to(torch.float32)
            
        self._ptr = (self._ptr + 1) % self._window

        ordered_buffers = [torch.cat((b[self._ptr:], b[:self._ptr]), dim=0) for b in self._buffers]
        
        compute_args = ordered_buffers + scalars
        res_matrix = self._compute(*compute_args)
        self._stream_val = res_matrix[-1, :]

    def stream_get_value(self):
        if self._stream_val is None:
            raise ValueError("尚未调用 stream_update 初始化！")
        return self._stream_val

    def stream_clear(self):
        self._buffers = None
        self._ptr = 0
        self._stream_val = None




# ======================================================================
# 极致时间复杂度工具箱 (Time-Complexity Optimized Toolkits)
# ======================================================================
def _get_rolling_window_view(x, window):
    """
    非线性算子底层：O(1) 内存视图滑动
    【安全防线】：防止合约生命周期太短导致 T < W 时发生 pad/unfold 崩溃
    """
    T = x.shape[0]
    # 如果数据不足一个窗口，直接返回一个充满 NaN 的视图，避免后续计算越界
    if T < window:
        return torch.full((T, x.shape[1], window), float('nan'), dtype=torch.float32, device=x.device)
        
    pad_len = window - 1
    x_padded = torch.nn.functional.pad(x.unsqueeze(0), (0, 0, pad_len, 0), value=float('nan')).squeeze(0)
    return x_padded.unfold(dimension=0, size=window, step=1)

def _rolling_sum_count(x, window):
    """
    线性算子底层：O(1) 极速时间复杂度滚动求和。
    【完美处理 NaN】：遇到 NaN 不加和，仅有效计数器 valid_mask 增加。
    【安全防线】：拦截 T < W 导致的 torch.cat 拼接崩溃。
    """
    T, C = x.shape
    valid_mask = ~torch.isnan(x)
    x_zeroed = torch.where(valid_mask, x, torch.tensor(0.0, device=x.device)).to(torch.float64)
    
    cumsum_x = torch.cumsum(x_zeroed, dim=0)
    cumsum_v = torch.cumsum(valid_mask.float(), dim=0)
    
    # 越界保护：如果数据长度不够一个窗口，直接返回当前的累计和
    # (外部的 min_periods 判断会把它们拦住变回 NaN)
    if T <= window:
        return cumsum_x.to(torch.float32), cumsum_v
    
    pad_x = torch.cat([torch.zeros((window, C), dtype=torch.float64, device=x.device), cumsum_x[:-window]], dim=0)
    pad_v = torch.cat([torch.zeros((window, C), dtype=torch.float32, device=x.device), cumsum_v[:-window]], dim=0)
    
    roll_sum = (cumsum_x - pad_x).to(torch.float32)
    roll_cnt = cumsum_v - pad_v
    return roll_sum, roll_cnt

def _rolling_binary_sum_count(x, y, window):
    """
    二元关系的 O(1) 滑动：严格交集有效性
    【安全防线】：拦截 T < W 导致的 torch.cat 拼接崩溃。
    """
    T, C = x.shape
    valid_mask = (~torch.isnan(x)) & (~torch.isnan(y))
    x_zeroed = torch.where(valid_mask, x, torch.tensor(0.0, device=x.device)).to(torch.float64)
    y_zeroed = torch.where(valid_mask, y, torch.tensor(0.0, device=x.device)).to(torch.float64)
    xy_zeroed = x_zeroed * y_zeroed
    
    # 分别算累加
    cx = torch.cumsum(x_zeroed, dim=0)
    cy = torch.cumsum(y_zeroed, dim=0)
    cxy = torch.cumsum(xy_zeroed, dim=0)
    cv = torch.cumsum(valid_mask.float(), dim=0)
    
    # 越界保护：直接返回累计值
    if T <= window:
        return cx.to(torch.float32), cy.to(torch.float32), cxy.to(torch.float32), cv
    
    def _get_win(c, dtype=torch.float64):
        # 由于上面已经防范了 T <= window，这里的 c[:-window] 绝对安全
        pad = torch.cat([torch.zeros((window, C), dtype=dtype, device=c.device), c[:-window]], dim=0)
        return c - pad

    r_x, r_y, r_xy = _get_win(cx), _get_win(cy), _get_win(cxy)
    r_cnt = _get_win(cv, torch.float32)
    
    return r_x.to(torch.float32), r_y.to(torch.float32), r_xy.to(torch.float32), r_cnt


# ======================================================================
# 1. 基础位移算子 (Shifting & Pct)
# ======================================================================
class TsDelay(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=0):
        w = int(window)
        out = torch.full_like(x, float('nan'))
        if w < x.shape[0]: out[w:] = x[:-w]
        return out

class TsDelta(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=1):
        w = int(window)
        out = torch.full_like(x, float('nan'))
        if w < x.shape[0]: out[w:] = x[w:] - x[:-w]
        return out

class TsPct(StatefulGPUOperator):
    """Log-Diff 稳健收益率"""
    def _compute(self, x, window, min_periods=1):
        w = int(window)
        out = torch.full_like(x, float('nan'))
        if w < x.shape[0]:
            valid = (x[w:] > 0) & (x[:-w] > 0)
            out[w:] = torch.where(valid, torch.log(x[w:] / x[:-w]), torch.tensor(float('nan'), device=x.device))
        return out

class TsFirst(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=1):
        view = _get_rolling_window_view(x, int(window))
        # 提取窗口内的有效计数，确保生命周期
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        return torch.where(valid_cnt >= min_periods, view[:, :, 0], torch.tensor(float('nan'), device=x.device))

class TsLast(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=1):
        # 也就是自身，但需要判定过去一段时间的有效性
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        return torch.where(valid_cnt >= min_periods, x, torch.tensor(float('nan'), device=x.device))


# ======================================================================
# 2. 线性聚合算子 (O(1) 极速计算) - 时间复杂度优先
# ======================================================================
class TsSum(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        r_sum, r_cnt = _rolling_sum_count(x, int(window))
        return torch.where(r_cnt >= min_periods, r_sum, torch.tensor(float('nan'), device=x.device))

class TsMean(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        r_sum, r_cnt = _rolling_sum_count(x, int(window))
        return torch.where(r_cnt >= min_periods, r_sum / r_cnt, torch.tensor(float('nan'), device=x.device))

class TsStd(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        w = int(window)
        # 获取滑动视图 (T, C, W)
        view = _get_rolling_window_view(x, w)
        valid = ~torch.isnan(view)
        cnt = valid.sum(dim=-1).float()
        
        # 为了绝对的精度，向上转型
        view_f64 = view.to(torch.float64)
        mean = (torch.nansum(view_f64, dim=-1) / cnt).unsqueeze(-1)
        
        # 计算真实偏差的平方
        dev = torch.where(valid, view_f64 - mean, torch.tensor(0.0, dtype=torch.float64, device=x.device))
        var = torch.sum(dev**2, dim=-1) / cnt
        
        # 强转回 float32
        std = torch.sqrt(torch.clamp(var, min=0.0)).to(torch.float32)
        return torch.where(cnt >= min_periods, std, torch.tensor(float('nan'), device=x.device))

class TsZscore(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        w = int(window)
        view = _get_rolling_window_view(x, w)
        valid = ~torch.isnan(view)
        cnt = valid.sum(dim=-1).float()
        
        view_f64 = view.to(torch.float64)
        mean = (torch.nansum(view_f64, dim=-1) / cnt).unsqueeze(-1)
        dev = torch.where(valid, view_f64 - mean, torch.tensor(0.0, dtype=torch.float64, device=x.device))
        var = torch.sum(dev**2, dim=-1) / cnt
        
        mean_32 = mean.squeeze(-1).to(torch.float32)
        std_32 = torch.sqrt(torch.clamp(var, min=0.0)).to(torch.float32)
        
        zscore = torch.where(std_32 != 0, (x - mean_32) / std_32, torch.tensor(1.0, device=x.device))
        return torch.where(cnt >= min_periods, zscore, torch.tensor(float('nan'), device=x.device))


# ======================================================================
# 3. 非线性聚合算子 (基于 Unfold 视图) 
# ======================================================================
class TsMax(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        out, _ = torch.nan_to_num(view, nan=-float('inf')).max(dim=-1)
        return torch.where((valid_cnt >= min_periods) & ~torch.isinf(out), out, torch.tensor(float('nan'), device=x.device))

class TsMin(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        out, _ = torch.nan_to_num(view, nan=float('inf')).min(dim=-1)
        return torch.where((valid_cnt >= min_periods) & ~torch.isinf(out), out, torch.tensor(float('nan'), device=x.device))

class TsMaxDiff(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        out_max, _ = torch.nan_to_num(view, nan=-float('inf')).max(dim=-1)
        return torch.where((valid_cnt >= min_periods) & ~torch.isinf(out_max), x - out_max, torch.tensor(float('nan'), device=x.device))

class TsMinDiff(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        out_min, _ = torch.nan_to_num(view, nan=float('inf')).min(dim=-1)
        return torch.where((valid_cnt >= min_periods) & ~torch.isinf(out_min), x - out_min, torch.tensor(float('nan'), device=x.device))

class TsArgMax(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        w = int(window)
        view = _get_rolling_window_view(x, w)
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        idx = torch.nan_to_num(view, nan=-float('inf')).argmax(dim=-1).float()
        return torch.where(valid_cnt >= min_periods, (w - 1 - idx) / (w - 1), torch.tensor(float('nan'), device=x.device))

class TsArgMin(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        w = int(window)
        view = _get_rolling_window_view(x, w)
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        idx = torch.nan_to_num(view, nan=float('inf')).argmin(dim=-1).float()
        return torch.where(valid_cnt >= min_periods, (w - 1 - idx) / (w - 1), torch.tensor(float('nan'), device=x.device))

class TsMedian(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        # return torch.where(valid_cnt >= min_periods, torch.nanmedian(view, dim=-1).values, torch.tensor(float('nan'), device=x.device))
        res = torch.nanquantile(view, 0.5, dim=-1)
        return torch.where(valid_cnt >= min_periods, res, torch.tensor(float('nan'), device=x.device))


class TsQuantile(StatefulGPUOperator):
    def _compute(self, x, window, q, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        return torch.where(valid_cnt >= min_periods, torch.nanquantile(view, float(q), dim=-1), torch.tensor(float('nan'), device=x.device))

class TsMad(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        median = torch.nanmedian(view, dim=-1, keepdim=True).values
        abs_dev = torch.abs(view - median)
        return torch.where(valid_cnt >= min_periods, torch.nanmean(abs_dev, dim=-1), torch.tensor(float('nan'), device=x.device))

class TsRank(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1).float()
        curr_x = x.unsqueeze(-1)
        greater_count = torch.sum(curr_x > view, dim=-1, dtype=torch.float32)
        # 【修复】：将 valid_count 改为 valid_cnt
        return torch.where(
            (valid_cnt >= min_periods) & ~torch.isnan(x),
            greater_count / (valid_cnt - 1.0 + 1e-8),
            torch.tensor(float('nan'), device=x.device),
        )

# ======================================================================
# 4. 高阶统计与衰减算子
# ======================================================================

class TsSkewness(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=3):
        # 1. 获取滑动视图 (T, C, W)
        view = _get_rolling_window_view(x, int(window))
        valid = ~torch.isnan(view)
        cnt = valid.sum(dim=-1).float()
        
        # 2. 【精度防线】：向上转型为 float64 进行高阶计算
        view_f64 = view.to(torch.float64)
        
        # 3. 严格的窗口内均值
        mean = (torch.nansum(view_f64, dim=-1) / cnt).unsqueeze(-1)
        
        # 4. 窗口内真实偏差 (x_i - mean_window)
        dev = torch.where(valid, view_f64 - mean, torch.tensor(0.0, dtype=torch.float64, device=x.device))
        
        # 5. 计算中心矩
        m2 = torch.sum(dev**2, dim=-1) / cnt
        m3 = torch.sum(dev**3, dim=-1) / cnt
        
        # 6. 计算偏度 (加入 eps 防止除 0)
        skew = m3 / (torch.pow(m2, 1.5) + 1e-12)
        
        # 7. Pandas Fisher-Pearson 无偏调整
        adj = torch.sqrt(cnt * (cnt - 1)) / (cnt - 2 + 1e-12)
        unbiased_skew = skew * adj
        
        # 8. 强转回 float32 输出
        res = torch.where(cnt >= max(min_periods, 3), unbiased_skew.to(torch.float32), torch.tensor(float('nan'), device=x.device))
        return res


class TsKurtosis(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=4):
        view = _get_rolling_window_view(x, int(window))
        valid = ~torch.isnan(view)
        cnt = valid.sum(dim=-1).float()
        
        # 【精度防线】：向上转型为 float64 防止四次方溢出 (10000^4)
        view_f64 = view.to(torch.float64)
        mean = (torch.nansum(view_f64, dim=-1) / cnt).unsqueeze(-1)
        dev = torch.where(valid, view_f64 - mean, torch.tensor(0.0, dtype=torch.float64, device=x.device))
        
        m2 = torch.sum(dev**2, dim=-1) / cnt
        m4 = torch.sum(dev**4, dim=-1) / cnt
        
        # 计算基础峰度
        kurt = m4 / (m2**2 + 1e-12)
        
        # Pandas Fisher 无偏调整 (Excess Kurtosis)
        adj1 = (cnt - 1) / ((cnt - 2) * (cnt - 3) + 1e-12)
        unbiased_kurt = adj1 * ((cnt + 1) * kurt - 3 * (cnt - 1))
        
        res = torch.where(cnt >= max(min_periods, 4), unbiased_kurt.to(torch.float32), torch.tensor(float('nan'), device=x.device))
        return res

class TsEntropy(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        view = _get_rolling_window_view(x, int(window))
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        v_sum = torch.nansum(view, dim=-1, keepdim=True)
        prob = view / (v_sum + 1e-8)
        entropy = -torch.nansum(prob * torch.log(prob + 1e-8), dim=-1)
        return torch.where((valid_cnt >= min_periods) & (v_sum.squeeze(-1) > 0), entropy, torch.tensor(float('nan'), device=x.device))

class TsCv(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        w = int(window)
        view = _get_rolling_window_view(x, w)
        valid = ~torch.isnan(view)
        cnt = valid.sum(dim=-1).float()
        
        view_f64 = view.to(torch.float64)
        mean = (torch.nansum(view_f64, dim=-1) / cnt).unsqueeze(-1)
        dev = torch.where(valid, view_f64 - mean, torch.tensor(0.0, dtype=torch.float64, device=x.device))
        var = torch.sum(dev**2, dim=-1) / cnt
        
        mean_32 = mean.squeeze(-1).to(torch.float32)
        std_32 = torch.sqrt(torch.clamp(var, min=0.0)).to(torch.float32)
        
        cv = torch.where(mean_32 != 0, std_32 / mean_32, torch.tensor(0.0, device=x.device))
        return torch.where((cnt >= min_periods) & (mean_32 != 0), cv, torch.tensor(float('nan'), device=x.device))

class TsIr(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        w = int(window)
        view = _get_rolling_window_view(x, w)
        valid = ~torch.isnan(view)
        cnt = valid.sum(dim=-1).float()
        
        view_f64 = view.to(torch.float64)
        mean = (torch.nansum(view_f64, dim=-1) / cnt).unsqueeze(-1)
        dev = torch.where(valid, view_f64 - mean, torch.tensor(0.0, dtype=torch.float64, device=x.device))
        var = torch.sum(dev**2, dim=-1) / cnt
        
        mean_32 = mean.squeeze(-1).to(torch.float32)
        std_32 = torch.sqrt(torch.clamp(var, min=0.0)).to(torch.float32)
        
        ir = torch.where(std_32 != 0, mean_32 / std_32, torch.tensor(0.0, device=x.device))
        return torch.where((cnt >= min_periods) & (std_32 != 0), ir, torch.tensor(float('nan'), device=x.device))

import triton
import triton.language as tl

# ======================================================================
# Triton Kernel: 极速 O(1) 并发指数加权衰减
# 彻底终结 Python for 循环带来的 GPU 启动延迟地狱！
# ======================================================================
@triton.jit
def _ewma_forward_kernel(
    x_ptr, out_ptr, 
    T, C, alpha: tl.constexpr, 
    BLOCK_C: tl.constexpr
):
    """
    Triton EWMA 内核。
    按合约维度 (C) 并发。每个线程块沿着时间轴 (T) 极速扫描。
    完全在 GPU 高速 SRAM 内完成状态递推。
    """
    # 确定当前线程块负责的合约 ID 范围
    pid_c = tl.program_id(0)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c_offsets < C
    
    # 状态寄存器 (存放在极速的 SRAM 中)
    state = tl.zeros([BLOCK_C], dtype=tl.float32)
    has_valid_data = tl.zeros([BLOCK_C], dtype=tl.int32)
    
    # 沿着时间轴 T 进行 O(N) 单趟扫描
    for t in range(T):
        offset = t * C + c_offsets
        
        # 加载当前时点的数据
        x_val = tl.load(x_ptr + offset, mask=mask, other=float('nan'))
        
        # 判断是否为有效值 (非 NaN)
        is_valid = x_val == x_val
        
        # --- 核心状态递推 ---
        # 1. 遇到第一个有效值，直接赋给状态
        state = tl.where(is_valid & (has_valid_data == 0), x_val, state)
        
        # 2. 后续的有效值，应用 EWMA 公式
        state = tl.where(is_valid & (has_valid_data == 1), alpha * x_val + (1.0 - alpha) * state, state)
        
        # 3. 标记状态
        has_valid_data = tl.where(is_valid, 1, has_valid_data)
        
        # 输出：若数据有效则输出状态，若无效则直接输出 NaN (绝不跨空洞污染)
        out_val = tl.where(is_valid, state, float('nan'))
        tl.store(out_ptr + offset, out_val, mask=mask)


class EWMA(StatefulGPUOperator):
    def _compute(self, x, halflife, min_periods=0):
        if not x.is_cuda:
            raise ValueError("EWMA 的 Triton 实现要求输入必须在 GPU 显存上！")
            
        T, C = x.shape
        out = torch.empty_like(x)
        
        # 计算 EWMA 衰减系数
        alpha = 1.0 - math.exp(-math.log(2.0) / float(halflife))
        
        # 设定并发网格：32 个合约为一个处理块
        BLOCK_C = 32
        grid = (triton.cdiv(C, BLOCK_C), )
        
        # 瞬间发射 Triton 极速内核！
        _ewma_forward_kernel[grid](x, out, T, C, alpha, BLOCK_C=BLOCK_C)
        
        return out


class TsDecay(StatefulGPUOperator):
    def _compute(self, x, window, min_periods=5):
        w = int(window)
        view = _get_rolling_window_view(x, w)
        valid_cnt = torch.sum(~torch.isnan(view), dim=-1)
        
        # 【精度防线】：由于乘以 weights 后求和容易膨胀，在 sum 时转为 float64
        weights = torch.arange(1, w + 1, dtype=torch.float64, device=x.device)
        view_f64 = view.to(torch.float64)
        
        weighted_view = view_f64 * weights
        num = torch.nansum(weighted_view, dim=-1)
        
        valid_weights = (~torch.isnan(view_f64)) * weights
        den = torch.sum(valid_weights, dim=-1)
        
        res = torch.where((valid_cnt >= min_periods) & (den > 0), (num / den).to(torch.float32), torch.tensor(float('nan'), device=x.device))
        return res


# ======================================================================
# 5. 二元关系时序算子 (Binary Time-Series Ops)
# ======================================================================
class TsCov(StatefulGPUOperator):
    def _compute(self, x, y, window, min_periods=5):
        w = int(window)
        r_x, r_y, r_xy, r_cnt = _rolling_binary_sum_count(x, y, w)
        
        cov = (r_xy - (r_x * r_y) / (r_cnt + 1e-8)) / (r_cnt - 1.0 + 1e-8)
        return torch.where(r_cnt >= min_periods, cov, torch.tensor(float('nan'), device=x.device))

class TsCorr(StatefulGPUOperator):
    def _compute(self, x, y, window, min_periods=5):
        w = int(window)
        valid_mask = (~torch.isnan(x)) & (~torch.isnan(y))
        x_0 = torch.where(valid_mask, x, torch.tensor(0.0, device=x.device)).to(torch.float64)
        y_0 = torch.where(valid_mask, y, torch.tensor(0.0, device=x.device)).to(torch.float64)
        
        cx = torch.cumsum(x_0, dim=0)
        cy = torch.cumsum(y_0, dim=0)
        cxy = torch.cumsum(x_0 * y_0, dim=0)
        cx2 = torch.cumsum(x_0**2, dim=0)
        cy2 = torch.cumsum(y_0**2, dim=0)
        cv = torch.cumsum(valid_mask.float(), dim=0)
        
        def _get_window(c, dtype=torch.float64):
            # 【修复】：防止 window 大于数据长度导致的切片越界报错
            if w >= c.shape[0]: return c
            pad = torch.cat([torch.zeros((w, c.shape[1]), dtype=dtype, device=c.device), c[:-w]], dim=0)
            return c - pad

        r_x, r_y, r_xy = _get_window(cx), _get_window(cy), _get_window(cxy)
        r_x2, r_y2, r_cnt = _get_window(cx2), _get_window(cy2), _get_window(cv, torch.float32)
        
        cov = r_xy - (r_x * r_y) / (r_cnt + 1e-8)
        var_x = r_x2 - (r_x**2) / (r_cnt + 1e-8)
        var_y = r_y2 - (r_y**2) / (r_cnt + 1e-8)
        
        corr = cov / (torch.sqrt(torch.clamp(var_x * var_y, min=0.0)) + 1e-8)
        corr = torch.clamp(corr, -1.0, 1.0)
        return torch.where((r_cnt >= min_periods) & (var_x > 0) & (var_y > 0), corr.to(torch.float32), torch.tensor(float('nan'), device=x.device))
    
class TsFacNeutralCoef(StatefulGPUOperator):
    """回归系数 Beta: Cov(x, y) / Var(x)"""
    def _compute(self, x, y, window, min_periods=5):
        w = int(window)
        # 1. 严格交集有效性掩码
        valid_mask = (~torch.isnan(x)) & (~torch.isnan(y))
        
        # 2. 为 cumsum 准备的安全零填充张量 (必须用 float64 防溢出)
        x_0 = torch.where(valid_mask, x, torch.tensor(0.0, device=x.device)).to(torch.float64)
        y_0 = torch.where(valid_mask, y, torch.tensor(0.0, device=x.device)).to(torch.float64)
        
        # 3. 极速前缀和
        cx = torch.cumsum(x_0, dim=0)
        cy = torch.cumsum(y_0, dim=0)
        cxy = torch.cumsum(x_0 * y_0, dim=0)
        cx2 = torch.cumsum(x_0**2, dim=0)
        cv = torch.cumsum(valid_mask.float(), dim=0)
        
        def _get_window(c, dtype=torch.float64):
            if w >= c.shape[0]: return c
            pad = torch.cat([torch.zeros((w, c.shape[1]), dtype=dtype, device=c.device), c[:-w]], dim=0)
            return c - pad

        # 4. 提取窗口内滚动状态
        r_x, r_y, r_xy, r_x2 = _get_window(cx), _get_window(cy), _get_window(cxy), _get_window(cx2)
        r_cnt = _get_window(cv, torch.float32)
        
        # 5. 计算协方差与方差
        cov = r_xy - (r_x * r_y) / (r_cnt + 1e-8)
        var_x = r_x2 - (r_x**2) / (r_cnt + 1e-8)
        
        # 6. 计算 Beta (防零除)
        beta = cov / (var_x + 1e-8)
        return torch.where((r_cnt >= min_periods) & (var_x > 0), beta.to(torch.float32), torch.tensor(float('nan'), device=x.device))


class TsFacNeutralRes(StatefulGPUOperator):
    """回归残差: y - (Beta * x + Alpha)"""
    def _compute(self, x, y, window, min_periods=5):
        w = int(window)
        # 1. 严格交集有效性掩码
        valid_mask = (~torch.isnan(x)) & (~torch.isnan(y))
        
        # 2. 为 cumsum 准备的安全零填充张量 (float64 防溢出)
        x_0 = torch.where(valid_mask, x, torch.tensor(0.0, device=x.device)).to(torch.float64)
        y_0 = torch.where(valid_mask, y, torch.tensor(0.0, device=x.device)).to(torch.float64)
        
        cx = torch.cumsum(x_0, dim=0)
        cy = torch.cumsum(y_0, dim=0)
        cxy = torch.cumsum(x_0 * y_0, dim=0)
        cx2 = torch.cumsum(x_0**2, dim=0)
        cv = torch.cumsum(valid_mask.float(), dim=0)
        
        def _get_window(c, dtype=torch.float64):
            if w >= c.shape[0]: return c
            pad = torch.cat([torch.zeros((w, c.shape[1]), dtype=dtype, device=c.device), c[:-w]], dim=0)
            return c - pad

        # 3. 提取窗口内滚动状态
        r_x, r_y, r_xy, r_x2 = _get_window(cx), _get_window(cy), _get_window(cxy), _get_window(cx2)
        r_cnt = _get_window(cv, torch.float32)
        
        # 4. 计算滚动 Beta 与 Mean
        cov = r_xy - (r_x * r_y) / (r_cnt + 1e-8)
        var_x = r_x2 - (r_x**2) / (r_cnt + 1e-8)
        beta = cov / (var_x + 1e-8)
        
        mean_x = r_x / (r_cnt + 1e-8)
        mean_y = r_y / (r_cnt + 1e-8)
        
        # 5. 【极其关键的修复】：使用原始带有 NaN 的 x 和 y 计算残差！
        # 这样当当前时刻 x 或 y 为 NaN 时，残差天然变成 NaN，绝不会用 0.0 去算！
        res = (y - mean_y) - beta * (x - mean_x)
        
        return torch.where((r_cnt >= min_periods) & (var_x > 0), res.to(torch.float32), torch.tensor(float('nan'), device=x.device))



class TsFutureDelay(StatefulGPUOperator):
    """
    【危险算子：仅限 Label 构建时使用】
    向未来 (Forward) 偏移数据。这是彻头彻尾的未来函数。
    绝对禁止将其用于特征 (Feature) 的计算中，只能用于收益率标签 (Label)。
    
    参数 window: 向未来看多少根 Bar。
    例如 window=15，当前 t 的结果等于 t+15 时刻的值。
    """
    def _compute(self, x, window, min_periods=0):
        w = int(window)
        out = torch.full_like(x, float('nan'))
        if w < x.shape[0]:
            # 极速向量化：当前位置填入未来第 W 个位置的值
            out[:-w] = x[w:]
        return out

    def stream_update(self, *args):
        raise RuntimeError("【绝对安全拦截】: 你试图在实盘/流式引擎中调用带有未来函数的 Label 算子 (TsForwardReturn)！实盘中不存在未来的价格。该算子仅限回测 (Batch Mode) 提取标签使用。")

class TsForwardReturn(StatefulGPUOperator):
    """
    【标准工业级日内收益率标签 (Forward Return)】
    逻辑：在当前 Bar 产生信号，下一根 Bar (t+1) 进场，在第 t+1+w 根 Bar 离场。
    因此，收益 = Log( x[t+1+w] / x[t+1] )
    
    参数:
    x: 必须是真实交易价格 (建议传入 VWAP 或 CLOSE_TRD)
    window: 持仓的分钟数 (比如 15)
    """
    def _compute(self, x, window, min_periods=0):
        w = int(window)
        out = torch.full_like(x, float('nan'))
        
        # 至少需要往未来看 w+1 根 Bar
        if w + 1 < x.shape[0]:
            # 进场价 (Entry Price)：下一根 Bar (t+1)
            p_entry = x[1 : -w]
            
            # 离场价 (Exit Price)：第 t+1+w 根 Bar
            p_exit = x[w + 1 :]
            
            # 防御性过滤：如果进出场有任何一个价格 <= 0 (退市或停牌)，记为 NaN
            valid = (p_entry > 0) & (p_exit > 0)
            
            # 向量化计算收益率，直接填入当前时刻 t 的位置！
            out[:-w-1] = torch.where(
                valid, 
                torch.log(p_exit / p_entry), 
                torch.tensor(float('nan'), device=x.device)
            )
        return out

    def stream_update(self, *args):
        raise RuntimeError("【绝对安全拦截】: 你试图在实盘/流式引擎中调用带有未来函数的 Label 算子 (TsForwardReturn)！实盘中不存在未来的价格。该算子仅限回测 (Batch Mode) 提取标签使用。")
