import sys
import os
import glob
import time
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from tick_framework import BaseTickFeature, process_daily_ticks
import tick_engine as ce

# ======================================================================
# 1. 全市场标准品种白名单配置 (按交易所分类)
# ======================================================================
# 可以自由开关需要跑的交易所
RUN_EXCHANGES = [
    'cffex',  # 中金所 (股指)
    'shfe',   # 上期所
    'dce',    # 大商所
    'czce',   # 郑商所
    'ine',    # 能源中心
    'gfex'    # 广期所
]

# 纯净版全市场标准主力合约前缀
EXCHANGE_PREFIXES = {
    'cffex': ['IC', 'IF', 'IH', 'IM'],
    'shfe': ['AG', 'AL', 'AO', 'AU', 'BR', 'BU', 'CU', 'FU', 'HC', 'NI', 'PB', 'RB', 'RU', 'SN', 'SP', 'SS', 'WR', 'ZN'],
    'dce':  ['A', 'B', 'BB', 'C', 'CS', 'EB', 'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V', 'Y'],
    'czce': ['AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'PF', 'PK', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR'],
    'ine':  ['BC', 'EC', 'LU', 'NR', 'SC'],
    'gfex': ['LC']
}

# 动态组合当前 Worker 节点需要处理的终极前缀白名单
TARGET_PREFIXES = []
for exch in RUN_EXCHANGES:
    if exch in EXCHANGE_PREFIXES:
        TARGET_PREFIXES.extend(EXCHANGE_PREFIXES[exch])
        
print(f"🎯 当前配置将计算 {len(TARGET_PREFIXES)} 个品种的高频特征...")

# ======================================================================
# 2. 特征业务逻辑 (极其优雅的模块化定义)
# ======================================================================
class FeatureOBI(BaseTickFeature):
    def __init__(self):
        super().__init__()
        # 默认 auto_clear=True
        self.mean_obi = self.register_state_op(ce.TsMean())

    def update(self, snp):
        bid = snp.bid_1_qty
        ask = snp.ask_1_qty
        is_valid = (bid + ask) > 0
        obi = (bid - ask) / (bid + ask)
        self.mean_obi.set_input(ce.IfElse(is_valid, obi, ce.NAN))

    def get_value(self):
        return {'OBI_Minute': self.mean_obi}


class FeatureMinuteSnapshot(BaseTickFeature):
    def __init__(self):
        super().__init__()
        # ===================================================================
        # 注册状态算子：使用 TsLast 截取分钟切片末的最后一个有效值
        # 默认 auto_clear=True，在每分钟触发 get_value 后底层自动清空
        # ===================================================================
        self.last_ask_prc_op = self.register_state_op(ce.TsLast(window=0, min_periods=1))
        self.last_ask_qty_op = self.register_state_op(ce.TsLast(window=0, min_periods=1))
        self.last_bid_prc_op = self.register_state_op(ce.TsLast(window=0, min_periods=1))
        self.last_bid_qty_op = self.register_state_op(ce.TsLast(window=0, min_periods=1))
        
        # ⚠️ 新增：持仓量算子
        self.oi_op = self.register_state_op(ce.TsLast(window=0, min_periods=1))
        
        # ⚠️ 基础算子：获取本分钟最后一个快照的 Mid Price (计算Label必备)
        self.last_mid_op = self.register_state_op(ce.TsLast(window=0, min_periods=1))

    def update(self, snp):
        """
        Symbolic Tracing Update - 构建 C++ DAG 图连线
        """
        # -----------------------------------------------------------
        # 💥【常数包装铁律】严格使用 ce.ConstNode
        # -----------------------------------------------------------
        zero_node = ce.ConstNode(0.0)
        two_node = ce.ConstNode(2.0)
        
        # -----------------------------------------------------------
        # 💥【逻辑括号铁律】校验盘口买一卖一是否有效
        # -----------------------------------------------------------
        is_valid_quote = (snp.ask_1_prc > zero_node) & (snp.bid_1_prc > zero_node)
        
        # 1. 记录最优买卖挂单量、价 (防脏数据注入)
        valid_ask_prc = ce.IfElse(is_valid_quote, snp.ask_1_prc, ce.NAN)
        valid_ask_qty = ce.IfElse(is_valid_quote, snp.ask_1_qty, ce.NAN)
        valid_bid_prc = ce.IfElse(is_valid_quote, snp.bid_1_prc, ce.NAN)
        valid_bid_qty = ce.IfElse(is_valid_quote, snp.bid_1_qty, ce.NAN)
        
        self.last_ask_prc_op.set_input(valid_ask_prc)
        self.last_ask_qty_op.set_input(valid_ask_qty)
        self.last_bid_prc_op.set_input(valid_bid_prc)
        self.last_bid_qty_op.set_input(valid_bid_qty)
        
        # 2. 记录持仓量 (open_interest) 校验非负
        is_valid_oi = (snp.open_interest >= zero_node)
        valid_oi = ce.IfElse(is_valid_oi, snp.open_interest, ce.NAN)
        self.oi_op.set_input(valid_oi)
        
        # 3. 计算当笔快照的 Mid Price
        current_mid = (snp.ask_1_prc + snp.bid_1_prc) / two_node
        self.last_mid_op.set_input(ce.IfElse(is_valid_quote, current_mid, ce.NAN))

    def get_value(self) -> dict:
        """
        分钟边界触发提取。提取完成后自动 clear()。
        """
        return {
            'ask1_qty': self.last_ask_qty_op,
            'bid1_qty': self.last_bid_qty_op,
            'open_interest': self.oi_op,             # 新增的持仓量
            'End_Mid_Price': self.last_mid_op        # 千万别忘了这个！
        }

class FeatureSpecificMicrostructure(BaseTickFeature):
    def __init__(self):
        super().__init__()
        
        # =====================================================================
        # [层级 1] 跨分钟保留状态的算子 (求 Tick 间的差分/延迟，auto_clear=False)
        # =====================================================================
        # 1. 计算 DownTick 需要的延迟 Mid
        self.delay_mid_op = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)
        
        # 2. 计算 Stealth Ask Retreat 需要的三个延迟状态
        self.delay_ask1_prc_op = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)
        self.delay_ask1_ratio_op = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)
        self.delay_ask5_vol_op = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)

        # =====================================================================
        # [层级 2] 本分钟内累加 (分钟边界清空，默认 auto_clear=True)
        # =====================================================================
        self.min_down_tick_sum = self.register_state_op(ce.TsSum())
        self.min_valid_tick_count = self.register_state_op(ce.TsSum())
        self.min_stealth_retreat_sum = self.register_state_op(ce.TsSum())
        
        # ⚠️ 必带的基础算子：获取本分钟最后一个快照的 Mid Price
        self.last_mid_op = self.register_state_op(ce.TsLast(window=0, min_periods=1))

    def update(self, snp):
        """
        Symbolic Tracing Update - 构建 C++ DAG 图连线
        """
        # ==========================================================
        # 1. 基础物理量计算与 Mid 提取
        # ==========================================================
        # 💥 铁律：常数必须包装，逻辑必须加括号
        is_valid_quote = (snp.ask_1_prc > ce.ConstNode(0.0)) & (snp.bid_1_prc > ce.ConstNode(0.0))
        mid = (snp.ask_1_prc + snp.bid_1_prc) / ce.ConstNode(2.0)
        
        self.last_mid_op.set_input(ce.IfElse(is_valid_quote, mid, ce.NAN))
        
        # 将有效 Tick 计入分母累加器
        is_valid_tick_val = ce.IfElse(is_valid_quote, ce.ConstNode(1.0), ce.ConstNode(0.0))
        self.min_valid_tick_count.set_input(is_valid_tick_val)

        # ==========================================================
        # 2. Down Tick Ratio 计算逻辑
        # ==========================================================
        cond_down = (mid < self.delay_mid_op)
        is_down_tick_val = ce.IfElse(cond_down, ce.ConstNode(1.0), ce.ConstNode(0.0))
        self.min_down_tick_sum.set_input(is_down_tick_val)
        
        # 状态递推 (为下一次计算做准备)
        self.delay_mid_op.set_input(mid)

        # ==========================================================
        # 3. Stealth Ask Retreat (卖方隐性退守) 计算逻辑
        # 含义：卖一价未变 + 卖一占5档比例下降 + 5档总量未崩盘 时，累计退守强度
        # ==========================================================
        # 算子：当前 5 档总量与占比
        ask5_vol_sum = snp.ask_1_qty + snp.ask_2_qty + snp.ask_3_qty + snp.ask_4_qty + snp.ask_5_qty
        
        # 💥 铁律：除法防零保护
        cond_ask5_valid = (ask5_vol_sum > ce.ConstNode(0.0))
        ask1_ratio = ce.IfElse(cond_ask5_valid, snp.ask_1_qty / ask5_vol_sum, ce.ConstNode(0.0))
        
        # 条件 1：卖一价未变
        cond1_prc_stable = (snp.ask_1_prc == self.delay_ask1_prc_op)
        # 条件 2：卖一占比下降 (说明挂单偷偷往后撤了)
        cond2_ratio_drop = (ask1_ratio < self.delay_ask1_ratio_op)
        # 条件 3：5档总量未崩盘 (这里定义为当前总量不低于上一快照总量的 90%)
        cond3_vol_stable = (ask5_vol_sum >= (self.delay_ask5_vol_op * ce.ConstNode(0.9)))
        
        # 💥 铁律：多重逻辑全括号包裹
        is_stealth_retreat = ((cond1_prc_stable) & (cond2_ratio_drop)) & (cond3_vol_stable)
        
        # 计算退守强度 = 下降的比例差值
        retreat_strength = ce.IfElse(
            is_stealth_retreat, 
            self.delay_ask1_ratio_op - ask1_ratio, 
            ce.ConstNode(0.0)
        )
        self.min_stealth_retreat_sum.set_input(retreat_strength)
        
        # 状态递推 (为下一次计算做准备)
        self.delay_ask1_prc_op.set_input(snp.ask_1_prc)
        self.delay_ask1_ratio_op.set_input(ask1_ratio)
        self.delay_ask5_vol_op.set_input(ask5_vol_sum)

    def get_value(self) -> dict:
        """
        分钟边界触发提取。输出本分钟的汇总特征。
        """
        c_0 = ce.ConstNode(0.0)
        c_20 = ce.ConstNode(20.0)
        
        # 💥 铁律：特征除法同样需要严格的防零保护 (图计算模式下所有分支都会生成计算节点，即使被外部覆盖也需内部防爆)
        cond_has_ticks = (self.min_valid_tick_count > c_0)
        down_tick_ratio = ce.IfElse(
            cond_has_ticks, 
            self.min_down_tick_sum / self.min_valid_tick_count, 
            c_0
        )
        
        # 💥 小样本防噪铁律拦截：判断本分钟内的有效快照数是否大于 20
        is_valid_bar = (self.min_valid_tick_count > c_20)
        
        # 强制遮罩保护
        final_down_tick_ratio = ce.IfElse(is_valid_bar, down_tick_ratio, ce.NAN)
        final_stealth_ask_retreat = ce.IfElse(is_valid_bar, self.min_stealth_retreat_sum, ce.NAN)
        
        return {
            'down_tick_ratio_bar': final_down_tick_ratio,
            'stealth_ask_retreat': final_stealth_ask_retreat
        }

class FeatureLastDeepOrderBook(BaseTickFeature):
    def __init__(self):
        super().__init__()
        # 1. 统计有效快照数量 (防噪基建)
        self.min_valid_tick_count = self.register_state_op(ce.TsSum())

        # 2. 批量注册 5 档买卖价量的 TsLast 算子
        # window=0 表示分钟桶模式，auto_clear=True(默认) 确保跨分钟时自动清空状态
        self.ask_prc_ops = [self.register_state_op(ce.TsLast(window=0, min_periods=1)) for _ in range(5)]
        self.ask_qty_ops = [self.register_state_op(ce.TsLast(window=0, min_periods=1)) for _ in range(5)]
        self.bid_prc_ops = [self.register_state_op(ce.TsLast(window=0, min_periods=1)) for _ in range(5)]
        self.bid_qty_ops = [self.register_state_op(ce.TsLast(window=0, min_periods=1)) for _ in range(5)]

    def update(self, snp):
        c_0 = ce.ConstNode(0.0)
        c_1 = ce.ConstNode(1.0)

        # 记录有效快照数 (买卖一价均存在视为有效 Tick)
        is_valid_quote = (snp.ask_1_prc > c_0) & (snp.bid_1_prc > c_0)
        self.min_valid_tick_count.set_input(ce.IfElse(is_valid_quote, c_1, c_0))

        # 遍历 5 档盘口，更新到 TsLast 算子中
        # ⚠️ 注意：Python 循环在这里仅仅是“画图”工具，最终生成的是扁平极速的 C++ 计算节点
        for i in range(5):
            lvl = i + 1
            self.ask_prc_ops[i].set_input(getattr(snp, f'ask_{lvl}_prc'))
            self.ask_qty_ops[i].set_input(getattr(snp, f'ask_{lvl}_qty'))
            self.bid_prc_ops[i].set_input(getattr(snp, f'bid_{lvl}_prc'))
            self.bid_qty_ops[i].set_input(getattr(snp, f'bid_{lvl}_qty'))

    def get_value(self) -> dict:
        c_20 = ce.ConstNode(20.0)
        
        # 小样本防噪铁律拦截
        is_valid_bar = (self.min_valid_tick_count > c_20)

        result = {}
        for i in range(1,5):
            lvl = i + 1
            # 严格按照要求的字段名：ask1, ask1_qty, bid1, bid1_qty 等
            result[f'ask{lvl}'] = ce.IfElse(is_valid_bar, self.ask_prc_ops[i], ce.NAN)
            result[f'ask{lvl}_qty'] = ce.IfElse(is_valid_bar, self.ask_qty_ops[i], ce.NAN)
            result[f'bid{lvl}'] = ce.IfElse(is_valid_bar, self.bid_prc_ops[i], ce.NAN)
            result[f'bid{lvl}_qty'] = ce.IfElse(is_valid_bar, self.bid_qty_ops[i], ce.NAN)

        return result

class HighFreqMinuteFeatures(BaseTickFeature):
    def __init__(self):
        super().__init__()
        
        # =====================================================================
        # [层级 1] 跨分钟基础状态算子 (auto_clear=False)
        # =====================================================================
        self.tick_vol_diff = self.register_state_op(ce.TsDelta(window=1, min_periods=1), auto_clear=False)
        self.tick_amt_diff = self.register_state_op(ce.TsDelta(window=1, min_periods=1), auto_clear=False)
        
        self.delay_mid = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)
        self.delay_ask1 = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)
        self.delay_bid1 = self.register_state_op(ce.TsDelay(window=1, min_periods=1), auto_clear=False)

        # =====================================================================
        # [层级 2] 连续游程的核心组件 (auto_clear=False)
        # =====================================================================
        # --- 原始：连续上涨 ---
        self.run_up_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_up_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)
        self.run_amp_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_amp_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)
        
        # 🟢 [新增] 对称：连续下跌
        self.run_down_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_down_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)
        self.run_down_amp_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_down_amp_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)

        # --- 原始：连续主卖 ---
        self.run_sell_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_sell_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)
        # 🟢 [新增] 连续主卖下的累计 Volume
        self.run_sell_vol_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_sell_vol_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)

        # 🟢 [新增] 对称：连续主买 及 累计 Volume
        self.run_buy_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_buy_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)
        self.run_buy_vol_inner = self.register_state_op(ce.TsSum(window=0, min_periods=1), auto_clear=False)
        self.run_buy_vol_wrapper = self.register_state_op(ce.TsResetable(), auto_clear=False)

        # =====================================================================
        # [层级 3] 本分钟聚合桶 (auto_clear=True)
        # =====================================================================
        self.min_valid_tick_count = self.register_state_op(ce.TsSum(window=0, min_periods=1))
        
        self.min_vol_sum = self.register_state_op(ce.TsSum(window=0, min_periods=1))
        self.min_vbuy_sum = self.register_state_op(ce.TsSum(window=0, min_periods=1))
        
        self.min_max_up_count = self.register_state_op(ce.TsMax(window=0, min_periods=1))
        self.min_max_up_amp = self.register_state_op(ce.TsMax(window=0, min_periods=1))
        
        # 🟢 [新增] 聚合：连续下跌极值
        self.min_max_down_count = self.register_state_op(ce.TsMax(window=0, min_periods=1))
        self.min_max_down_amp = self.register_state_op(ce.TsMax(window=0, min_periods=1))
        
        self.min_max_sell_count = self.register_state_op(ce.TsMax(window=0, min_periods=1))
        # 🟢 [新增] 聚合：主卖及主买的相关极值
        self.min_max_sell_vol = self.register_state_op(ce.TsMax(window=0, min_periods=1))
        self.min_max_buy_count = self.register_state_op(ce.TsMax(window=0, min_periods=1))
        self.min_max_buy_vol = self.register_state_op(ce.TsMax(window=0, min_periods=1))

        self.min_basis_x_dv = self.register_state_op(ce.TsSum(window=0, min_periods=1))
        self.min_spread_std = self.register_state_op(ce.TsStd(window=0, min_periods=1))
        self.min_spread_mean = self.register_state_op(ce.TsMean(window=0, min_periods=1)) 
        self.min_dv_std = self.register_state_op(ce.TsStd(window=0, min_periods=1))
        self.min_dv_mean = self.register_state_op(ce.TsMean(window=0, min_periods=1))
        
        self.min_info_toxicity = self.register_state_op(ce.TsCorr(window=0, min_periods=3))
        self.min_obi_mean = self.register_state_op(ce.TsMean(window=0, min_periods=1))

    def update(self, snp):
        c_0 = ce.ConstNode(0.0)
        c_1 = ce.ConstNode(1.0)
        c_2 = ce.ConstNode(2.0)
        c_0_5 = ce.ConstNode(0.5)
        c_1e8 = ce.ConstNode(1e-8)
        
        # 1. 基础状态解析
        mid = (snp.ask_1_prc + snp.bid_1_prc) / c_2
        
        is_valid_quote = (snp.ask_1_prc > c_0) & (snp.bid_1_prc > c_0)
        tick_increment = ce.IfElse(is_valid_quote, c_1, c_0)
        self.min_valid_tick_count.set_input(tick_increment)
        
        self.delay_mid.set_input(mid)
        self.delay_ask1.set_input(snp.ask_1_prc)
        self.delay_bid1.set_input(snp.bid_1_prc)
        
        self.tick_vol_diff.set_input(snp.cum_volume)
        self.tick_amt_diff.set_input(snp.cum_amount)
        dv = ce.IfElse((self.tick_vol_diff > c_0), self.tick_vol_diff, c_0)
        dt = ce.IfElse((self.tick_amt_diff > c_0), self.tick_amt_diff, c_0)
        self.min_vol_sum.set_input(dv)
        
        # 2. 高频成交量方向归因
        tick_vwap = ce.IfElse((dv > c_0), dt / (dv * snp.contract_size), mid)
        
        max_ask = ce.IfElse((self.delay_ask1 > snp.ask_1_prc), self.delay_ask1, snp.ask_1_prc)
        min_bid = ce.IfElse((self.delay_bid1 < snp.bid_1_prc), self.delay_bid1, snp.bid_1_prc)
        A_active = ce.IfElse((self.delay_ask1 < snp.ask_1_prc), self.delay_ask1, snp.ask_1_prc)
        B_active = ce.IfElse((self.delay_bid1 > snp.bid_1_prc), self.delay_bid1, snp.bid_1_prc)
        
        denom_overlap = A_active - B_active
        safe_denom_overlap = ce.IfElse((denom_overlap > c_0), denom_overlap, c_1e8)
        ratio_overlap = (tick_vwap - B_active) / safe_denom_overlap
        
        # 前后报价区间不重叠或仅接触时，使用成交发生前的上一快照盘口。
        # 当前盘口可能已经包含该段成交造成的重定价，直接用它插值可能反转成交方向。
        denom_fallback = self.delay_ask1 - self.delay_bid1
        safe_denom_fallback = ce.IfElse((denom_fallback > c_0), denom_fallback, c_1e8)
        ratio_fallback = (tick_vwap - self.delay_bid1) / safe_denom_fallback

        cond_pure_buy = (tick_vwap >= max_ask)
        cond_pure_sell = (tick_vwap <= min_bid)
        cond_overlap = (B_active < A_active)

        buy_ratio_raw = ce.IfElse(
            cond_pure_buy, c_1,                           
            ce.IfElse(
                cond_pure_sell, c_0,                       
                ce.IfElse(cond_overlap, ratio_overlap, ratio_fallback)
            )
        )
        
        cond_gt_1 = (buy_ratio_raw > c_1)
        cond_lt_0 = (buy_ratio_raw < c_0)
        buy_ratio = ce.IfElse(cond_gt_1, c_1, ce.IfElse(cond_lt_0, c_0, buy_ratio_raw))
        v_buy = dv * buy_ratio
        self.min_vbuy_sum.set_input(v_buy)

        # =====================================================================
        # 3. 核心游程特征 (Up/Down, Buy/Sell 完全对称实现)
        # =====================================================================
        cond_delay_mid_valid = (self.delay_mid > c_0)
        
        # --- [A] 连续上涨 (原有) ---
        is_up = (mid > self.delay_mid)
        up_count_reset_cond = ce.IfElse(is_up, c_0, c_1)
        
        self.run_up_inner.set_input(c_1)
        self.run_up_wrapper.set_inputs(self.run_up_inner, up_count_reset_cond)
        self.min_max_up_count.set_input(self.run_up_wrapper)
        
        log_ret_up = ce.IfElse(cond_delay_mid_valid, ce.Log(mid / self.delay_mid), c_0)
        self.run_amp_inner.set_input(log_ret_up)
        self.run_amp_wrapper.set_inputs(self.run_amp_inner, up_count_reset_cond)
        self.min_max_up_amp.set_input(self.run_amp_wrapper)

        # 🟢 [B] 连续下跌 (对称新增) ---
        is_down = (mid < self.delay_mid)
        down_count_reset_cond = ce.IfElse(is_down, c_0, c_1)
        
        self.run_down_inner.set_input(c_1)
        self.run_down_wrapper.set_inputs(self.run_down_inner, down_count_reset_cond)
        self.min_max_down_count.set_input(self.run_down_wrapper)
        
        # ⚠️ 防零除铁律：C++ 计算图两端都会求值，为防止中位价为0触发极小概率的Log(除0)异常，增加 safe_mid 保护
        safe_mid_for_log = ce.IfElse((mid > c_0), mid, c_1)
        # 使用 Log(delay_mid / mid) 保证连续下跌幅度为正数，方便对称衡量
        log_ret_down = ce.IfElse(cond_delay_mid_valid & (mid > c_0), ce.Log(self.delay_mid / safe_mid_for_log), c_0)
        
        self.run_down_amp_inner.set_input(log_ret_down)
        self.run_down_amp_wrapper.set_inputs(self.run_down_amp_inner, down_count_reset_cond)
        self.min_max_down_amp.set_input(self.run_down_amp_wrapper)

        # --- [C] 连续主卖 (包含 Volume) ---
        is_active_sell = (buy_ratio < c_0_5)
        sell_count_reset_cond = ce.IfElse(is_active_sell, c_0, c_1)
        
        self.run_sell_inner.set_input(c_1)
        self.run_sell_wrapper.set_inputs(self.run_sell_inner, sell_count_reset_cond)
        self.min_max_sell_count.set_input(self.run_sell_wrapper)
        
        self.run_sell_vol_inner.set_input(dv) # 将本笔量送入游程求和
        self.run_sell_vol_wrapper.set_inputs(self.run_sell_vol_inner, sell_count_reset_cond)
        self.min_max_sell_vol.set_input(self.run_sell_vol_wrapper)

        # 🟢 [D] 连续主买 (包含 Volume) ---
        is_active_buy = (buy_ratio > c_0_5)
        buy_count_reset_cond = ce.IfElse(is_active_buy, c_0, c_1)
        
        self.run_buy_inner.set_input(c_1)
        self.run_buy_wrapper.set_inputs(self.run_buy_inner, buy_count_reset_cond)
        self.min_max_buy_count.set_input(self.run_buy_wrapper)
        
        self.run_buy_vol_inner.set_input(dv) # 将本笔量送入游程求和
        self.run_buy_vol_wrapper.set_inputs(self.run_buy_vol_inner, buy_count_reset_cond)
        self.min_max_buy_vol.set_input(self.run_buy_vol_wrapper)

        # =====================================================================
        # 4. 常规截面与聚流特征 (原有)
        # =====================================================================
        ask_bid_vol = snp.ask_1_qty + snp.bid_1_qty
        micro_prc = ce.IfElse((ask_bid_vol > c_0), 
                             (snp.bid_1_prc * snp.ask_1_qty + snp.ask_1_prc * snp.bid_1_qty) / ask_bid_vol, 
                             mid)
        self.min_basis_x_dv.set_input((tick_vwap - micro_prc) * dv)
        
        spread = snp.ask_1_prc - snp.bid_1_prc
        self.min_spread_std.set_input(spread)
        self.min_spread_mean.set_input(spread)
        
        self.min_dv_std.set_input(dv)
        self.min_dv_mean.set_input(dv)
        
        dMid = mid - self.delay_mid
        self.min_info_toxicity.set_inputs(dv, dMid)
        
        w1 = ce.ConstNode(1.0)
        w2 = ce.ConstNode(0.606)
        w3 = ce.ConstNode(0.367)
        w4 = ce.ConstNode(0.223)
        w5 = ce.ConstNode(0.135)
        weighted_bid = snp.bid_1_qty*w1 + snp.bid_2_qty*w2 + snp.bid_3_qty*w3 + snp.bid_4_qty*w4 + snp.bid_5_qty*w5
        weighted_ask = snp.ask_1_qty*w1 + snp.ask_2_qty*w2 + snp.ask_3_qty*w3 + snp.ask_4_qty*w4 + snp.ask_5_qty*w5
        obi_denom = weighted_bid + weighted_ask
        obi = ce.IfElse((obi_denom > c_0), (weighted_bid - weighted_ask) / obi_denom, c_0)
        self.min_obi_mean.set_input(obi)

    def get_value(self) -> dict:
        c_0 = ce.ConstNode(0.0)
        c_20 = ce.ConstNode(20.0)
        
        is_valid_bar = (self.min_valid_tick_count > c_20)
        
        def safe_div(num, denom):
            return ce.IfElse((denom > c_0), num / denom, c_0)
            
        def mask_nan(val):
            return ce.IfElse(is_valid_bar, val, ce.NAN)
        
        raw_buy_ratio = safe_div(self.min_vbuy_sum, self.min_vol_sum)
        raw_basis = safe_div(self.min_basis_x_dv, self.min_vol_sum)
        raw_spread_cv = safe_div(self.min_spread_std, self.min_spread_mean)
        raw_vol_cv = safe_div(self.min_dv_std, self.min_dv_mean)
        raw_toxicity = ce.FillNa(self.min_info_toxicity, c_0)

        return {
            'BuyRatio': mask_nan(raw_buy_ratio),
            'MicroPriceBasis': mask_nan(raw_basis),
            'SpreadCV': mask_nan(raw_spread_cv),
            'VolIncrementCV': mask_nan(raw_vol_cv),
            'InfoToxicity': mask_nan(raw_toxicity),
            'ExpOBI': mask_nan(self.min_obi_mean),
            
            # --- 游程特征部分 ---
            'MaxConsecUpTicks': mask_nan(self.min_max_up_count),
            'MaxConsecUpAmp': mask_nan(self.min_max_up_amp),
            
            # 🟢 新增：下跌游程
            'MaxConsecDownTicks': mask_nan(self.min_max_down_count),
            'MaxConsecDownAmp': mask_nan(self.min_max_down_amp),
            
            # 🟢 新增：主买游程及成交量
            'MaxConsecBuyCount': mask_nan(self.min_max_buy_count),
            'MaxConsecBuyVolume': mask_nan(self.min_max_buy_vol),
            
            # 🟢 新增：主卖游程及成交量
            'MaxConsecSellCount': mask_nan(self.min_max_sell_count),
            'MaxConsecSellVolume': mask_nan(self.min_max_sell_vol),
        }



# ======================================================================
# 2. 全市场智能交易时段洗膜 (自动识别股指与商品，对症下药)
# ======================================================================
def filter_global_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    print(f"执行全市场交易时段清洗前: {len(df)} 行")
    time_str = df['index_time'].dt.strftime('%H:%M:%S')
    
    # ------------------ 分支 A: 股指期货 (CFFEX) ------------------
    is_cffex = df['instrument'].isin(['IC', 'IF', 'IH', 'IM'])
    mask_cffex = (
        ((time_str >= '09:30:00') & (time_str <= '11:30:00')) |
        ((time_str >= '13:00:00') & (time_str <= '15:00:00'))
    )
    
    # ------------------ 分支 B: 商品期货 ------------------
    is_commodity = ~is_cffex
    mask_comm_day = (
        ((time_str >= '09:00:00') & (time_str <= '10:15:00')) |
        ((time_str >= '10:30:00') & (time_str <= '11:30:00')) |
        ((time_str >= '13:30:00') & (time_str <= '15:00:00'))
    )
    
    group_0230 = {'AU', 'AG', 'SC'}
    group_0100 = {'CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'SS', 'BC'}
    group_no_night = {'WR', 'BB', 'FB', 'JD', 'LH', 'RR', 'AP', 'CJ', 'PK', 'RS', 'SM', 'SF', 'UR', 'EC', 'LC'}
    
    mask_night_2300 = (time_str >= '21:00:00') & (time_str <= '23:00:00')
    mask_night_0100 = (time_str >= '21:00:00') | (time_str <= '01:00:00')
    mask_night_0230 = (time_str >= '21:00:00') | (time_str <= '02:30:00')

    is_0230 = df['instrument'].isin(group_0230)
    is_0100 = df['instrument'].isin(group_0100)
    is_no_night = df['instrument'].isin(group_no_night)
    is_2300 = ~(is_0230 | is_0100 | is_no_night | is_cffex)

    mask_comm = mask_comm_day | (
        (is_0230 & mask_night_0230) |
        (is_0100 & mask_night_0100) |
        (is_2300 & mask_night_2300)
    )
    
    # ------------------ 终极合并矩阵 ------------------
    # 股指走股指的规则，商品走商品的规则，井水不犯河水！
    valid_mask = (is_cffex & mask_cffex) | (is_commodity & mask_comm)

    df_clean = df[valid_mask].copy()
    df_clean = df_clean.sort_values(['index_time', 'instrument', 'contract']).reset_index(drop=True)
    
    print(f"✅ 执行全市场清洗后: {len(df_clean)} 行")
    return df_clean


import traceback

import gc  # 引入 Python 的垃圾回收模块

def process_single_task(task_dict: dict):
    """Worker 节点运行的原子任务，包含极端的内存防爆处理"""
    try:
        def load_files(file_list):
            temp = []
            for f in file_list:
                if os.path.exists(f):
                    df_raw = pd.read_csv(f)
                    
                    # ---- 新增：quot_time 年份自动修复 ----
                    qt_str = df_raw['quot_time'].astype(str)
                    
                    # 提取年份前4位
                    qt_year = qt_str.str[:4]
                    
                    # 尝试按标准格式解析，标记失败的
                    try:
                        test_parse = pd.to_datetime(qt_str, format='%Y%m%d%H%M%S%f', errors='coerce')
                        bad_mask = test_parse.isna() & qt_str.str.match(r'^\d{17}$')
                    except Exception:
                        # 如果全部炸了（OutOfBounds），直接判定：年份不在 [2000, 2099] 的都需要修
                        bad_mask = ~qt_year.astype(int).between(2000, 2099) & qt_str.str.match(r'^\d{17}$')
                    
                    if bad_mask.any():
                        # 用文件路径里的年份作为正确年份来修复
                        file_year = re.search(r'_(\d{8})\.csv', f).group(1)[:4]
                        df_raw.loc[bad_mask, 'quot_time'] = (
                            file_year + qt_str[bad_mask].str[4:]
                        ).astype(int)
                        print(f"  ⚠️ 修复 {bad_mask.sum()} 行年份异常数据 (→ {file_year})")
                    # -------------------------------------------------
                    
                    mask = df_raw['secu_cd'].str.extract(r'^([a-zA-Z]+)')[0].str.upper().isin(TARGET_PREFIXES)
                    df_filtered = df_raw[mask].copy()
                    temp.append(df_filtered)
                    
                    del df_raw
                    del mask
            
            if temp:
                res_df = pd.concat(temp, ignore_index=True)
                # 💥 拼接完立刻清理列表
                del temp
                res_df = res_df.sort_values(['quot_time', 'secu_cd']).reset_index(drop=True)
                return res_df
            
            return pd.DataFrame()

        dfs = []
        max_warmup_time_str = "" 
        
        # 1. 预热数据
        df_warmup = load_files(task_dict.get('warmup_files', []))
        if not df_warmup.empty:
            max_warmup_time_str = str(df_warmup['quot_time'].max())
            dfs.append(df_warmup)
            del df_warmup  # 💥 移入列表后立刻销毁变量引用
            
        # 2. 目标数据
        df_target = load_files(task_dict.get('target_files', []))
        if not df_target.empty:
            dfs.append(df_target)
            del df_target  # 💥 移入列表后立刻销毁变量引用
            
        if not dfs:
            return pd.DataFrame()
            
        # 3. 拼接总表
        df_combined = pd.concat(dfs, ignore_index=True)
        del dfs  # 💥 拼接后立刻销毁列表
        df_combined = df_combined.sort_values(['quot_time', 'secu_cd']).reset_index(drop=True)
        
        # 调用特征引擎
        my_factors = [HighFreqMinuteFeatures, FeatureSpecificMicrostructure, FeatureMinuteSnapshot, FeatureLastDeepOrderBook]
        df_res_combined = process_daily_ticks(my_factors, df_combined)
        
        # 💥 引擎算完后，给底层喂的数据可以直接杀了
        del df_combined 
        del my_factors
        
        if df_res_combined.empty:
            gc.collect() # 退出前强制回收
            return pd.DataFrame()
            
        # 4. 截断预热期数据
        if max_warmup_time_str:
            warmup_dt = pd.to_datetime(max_warmup_time_str[:17], format="%Y%m%d%H%M%S%f")
            final_target_res = df_res_combined[df_res_combined['index_time'] > warmup_dt].copy()
        else:
            final_target_res = df_res_combined.copy()
            
        # 💥 彻底斩杀最后的原始结果表
        del df_res_combined
        gc.collect()

        # ========== 【修改点：清洗洗膜后再存盘】 ==========
        if not final_target_res.empty:
            
            # --- 🚀 新插入的清洗逻辑 开始 ---
            final_target_res['contract'] = final_target_res['contract'].astype(str).str.upper()
            extracted = final_target_res['contract'].str.extract(r'^([A-Z]+)(\d+)$')
            final_target_res['raw_prefix'] = extracted[0]
            final_target_res['raw_suffix'] = extracted[1]
            
            # 直接使用标准的交易所代码
            final_target_res['instrument'] = final_target_res['raw_prefix']

            # 郑商所交割月补齐
            final_target_res['raw_suffix'] = final_target_res['raw_suffix'].apply(
                lambda x: '2' + str(x) if isinstance(x, str) and len(str(x)) == 3 else str(x)
            )
            final_target_res['contract'] = '20' + final_target_res['raw_suffix']

            final_target_res = final_target_res.dropna(subset=['instrument'])
            final_target_res = final_target_res.drop_duplicates(subset=['index_time', 'instrument', 'contract'], keep='last')
            
            if not final_target_res.empty:
                feature_cols = [c for c in final_target_res.columns if c not in ['index_time', 'contract', 'raw_prefix', 'raw_suffix', 'instrument']]
                final_cols = ['index_time', 'instrument', 'contract'] + feature_cols
                final_target_res = final_target_res[final_cols]
                final_target_res = final_target_res.sort_values(['index_time', 'instrument', 'contract']).reset_index(drop=True)
                final_target_res = filter_global_trading_hours(final_target_res)
            # --- 🚀 新插入的清洗逻辑 结束 ---
        
        if not final_target_res.empty:
            # 建立一个通用相对输出目录
            out_dir = "./data/features_output/2024/"
            os.makedirs(out_dir, exist_ok=True)
            
            date_str = task_dict.get('target_date')
            # 生成类似 "feat_20251128_xxx.parquet" 的唯一文件名
            # 这里可以用一个 UUID 或者 任务日期_进程ID 命名
            file_name = f"feat_day_{date_str}_{os.getpid()}.parquet"
            save_path = os.path.join(out_dir, file_name)
            
            # 直接落盘！
            final_target_res.to_parquet(save_path)
            
            # 清理内存
            del final_target_res
            gc.collect()
            
            return save_path  # 只返回文件路径字符串，不返回数据本身！
        
    except Exception as e:
        print(f"\n❌ [Worker 报错] 处理 {task_dict.get('target_date')} 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return None
        # raise e


def get_date_from_filename(filename: str) -> str:
    match = re.search(r'_day_(\d{8})\.csv', filename)
    return match.group(1) if match else None


def run_local_multiprocessing(tasks: list, max_workers: int = 4) -> list[str]:
    print(f"🚀 启动本地多进程模式 (核心数: {max_workers})...")
    output_paths: list[str] = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, output_path in enumerate(executor.map(process_single_task, tasks)):
            # Worker 已经将结果落盘，父进程只收集有效文件路径，
            # 避免将整年 DataFrame 重新汇总到主进程。
            if output_path is not None and os.path.isfile(output_path):
                output_paths.append(output_path)
                
            if (i + 1) % 10 == 0:
                print(f"   已完成 {i+1}/{len(tasks)} 个交易日...")
                
    if not output_paths:
        print("\n⚠️ 警告：所有交易日的结果全为空！请检查上面的 Worker 报错日志！")
        return []

    return output_paths


# ======================================================================
# 4. 引擎主控 & 最终格式化洗膜
# ======================================================================
if __name__ == "__main__":
    print("🔍 正在扫描 2025 年全市场期货原始数据文件...")
    base_dir = "./data/market_snapshot/2024"
    
    tasks = []
    
    # 根据开关遍历对应的交易所
    for exch in RUN_EXCHANGES:
        search_pattern = os.path.join(base_dir, "**", f"*.q.{exch}.futr", f"L2_new_FUTR_{exch.upper()}_*.csv")
        all_exch_files = glob.glob(search_pattern, recursive=True)
        
        date_to_files = {}
        for f in all_exch_files:
            match = re.search(r'_(?:day|night)_(\d{8})\.csv', f)
            if match:
                date_str = match.group(1)
                if date_str not in date_to_files:
                    date_to_files[date_str] = []
                date_to_files[date_str].append(f)
                
        sorted_dates = sorted(list(date_to_files.keys()))
    
        for i in range(len(sorted_dates)):
            target_date = sorted_dates[i]
            target_files = date_to_files[target_date]
            warmup_files = date_to_files[sorted_dates[i-1]] if i > 0 else []
            
            tasks.append({
                'warmup_files': warmup_files,
                'target_files': target_files,
                'target_date': target_date
            })

    print(f"✅ 找到并构建了 {len(tasks)} 个跨市场交易日任务！")

    MODE = 'local' 
    t_start = time.time()
    
    if MODE == 'local':
        valid_paths = run_local_multiprocessing(tasks, max_workers=4)

    t_end = time.time()
    
    # ---------------------------------------------------------
    # 主节点大瘦身：只要统计一下文件就行了
    # ---------------------------------------------------------
    # 过滤掉失败返回 None 的任务
    if valid_paths is not None:
        valid_paths = [p for p in valid_paths if p is not None]
    else:
        valid_paths = []

    print("\n" + "="*60)
    if valid_paths:
        print(f"🏆 2024 年期货 (全量交割月) 计算完成！总耗时: {t_end - t_start:.2f} 秒")
        print(f"✅ 成功落盘了 {len(valid_paths)} 个特征切片文件！")
        print("📁 存储路径: ./data/features_output/2024/")
        
        # 如果你想看个样本，可以随便读一个刚刚生成的文件打印一下
        try:
            sample_df = pd.read_parquet(valid_paths[0])
            print("\n输出样本 (完美符合 Base Panel 格式):")
            print(sample_df.head(3))
        except Exception:
            pass
    else:
        print("⚠️ 计算结果为空或所有任务均失败！")
    print("="*60)
