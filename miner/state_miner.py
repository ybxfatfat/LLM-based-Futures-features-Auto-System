#!/usr/bin/env python3
"""
state_miner.py - Automated Data-Driven Market State Miner (Pool-Specific & Cross-Factor Generalized).

Architectural flow:
1. Data-Driven Clues: Extracts actual winning/losing trades from a baseline Alpha position.
2. LLM Hypothesis: LLM analyzes the micro-structure snapshot to propose one-shot State variables (Macro vs Micro dichotomy).
3. GPU Execution: Compiles and executes the state formulas concurrently.
4. Regime Asymmetry Evaluation: Tests if the state variable successfully partitions multiple Alphas' 
   trades into "Golden Regimes" or "Toxic Regimes", using Median IR Spread across a feature pool to prevent overfitting.

Created for High-Frequency Quantitative Research.
"""

from __future__ import annotations

import os
import sys
import json
import re
import random
import logging
import pandas as pd
import numpy as np
import torch
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Literal
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

MIN_MEDIAN_SPREAD = 0.5  # 组间 IR 极差的合格线
MIN_MEDIAN_STD = 0.1     # 组间 IR 标准差的合格线

# ======================================================================
# ⚙️ 1. 环境变量与【全局宇宙配置】
# ======================================================================
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path: 
    sys.path.insert(0, str(project_root))

FOCUS_UNIVERSE = "Index"  # "Commodity" 或 "Index"
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

# 💥 脱敏修复：使用白名单直接动态拉取宇宙品种，消灭私有 CSV 与内部版块代号硬编码！
def get_universe_instruments(universe: str) -> list[str]:
    CFFEX_INSTRUMENTS = ['IC', 'IF', 'IH', 'IM']
    COMMODITY_INSTRUMENTS = [
        'AG', 'AL', 'AO', 'AU', 'BR', 'BU', 'CU', 'FU', 'HC', 'NI', 'PB', 'RB', 'RU', 'SN', 'SP', 'SS', 'WR', 'ZN',
        'A', 'B', 'BB', 'C', 'CS', 'EB', 'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V', 'Y',
        'AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'PF', 'PK', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR',
        'BC', 'EC', 'LU', 'NR', 'SC', 'LC'
    ]
    EXCLUDED_INSTRUMENTS = {'LC', 'EC', 'AO', 'BB', 'FB', 'WR', 'RS', 'BC', 'LU'}
    
    if universe == "Index":
        all_targets = set(CFFEX_INSTRUMENTS)
    elif universe == "Commodity":
        all_targets = set(COMMODITY_INSTRUMENTS)
    else:
        all_targets = set(CFFEX_INSTRUMENTS + COMMODITY_INSTRUMENTS)
        
    return [inst for inst in all_targets if inst not in EXCLUDED_INSTRUMENTS]

TARGET_INSTRUMENTS = get_universe_instruments(FOCUS_UNIVERSE)
TARGET_POOL_NAME = FOCUS_UNIVERSE

POS_COL_NAME = 'w2000_pos_30m_intra_z'
RET_COL_NAME = 'Ret_30m_Fwd'
# 脱敏修复：使用通用的相对路径
POSITIONS_DIR = project_root / "data" / "feature_positions"
BASE_PANEL_DIR = project_root / "data" / "intraday_parquet" / "base_panel"
BEST_FEATURES_FILE = project_root / "data" / f"best_features_{FOCUS_UNIVERSE.lower()}.json"
STATE_REGISTRY_FILE = project_root / "data" / f"market_states_registry_{FOCUS_UNIVERSE.lower()}.json"

# 互信息：必须至少有非零的非线性依存关系
MI_MIN = 0.2

# KS 统计量：赢亏两组的 state 分布必须有可检测的差异
KS_MIN = 0.1

# Uplift：相对基准 IR 的提升幅度（可选，作为辅助）
UPLIFT_MIN = 0.30  # 30%

logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [%(levelname)s] 🧠 [StateMiner-{FOCUS_UNIVERSE}]: %(message)s")
logger = logging.getLogger("StateMiner")

# ======================================================================
# 📦 导入依赖 & LLM / GPU 接口 (完全脱敏为 OpenAI 原生调用)
# ======================================================================
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        
    def to_dict(self):
        return {"role": self.role, "content": self.content}

class Chat:
    """轻量级 LLM 包装器，支持 Pydantic 结构化输出"""
    def __init__(self, model: str, messages: list[Message], temperature: float = 0.7, response_format=None):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.response_format = response_format

    def create(self):
        dict_messages = [m.to_dict() for m in self.messages]
        if self.response_format and issubclass(self.response_format, BaseModel):
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=dict_messages,
                temperature=self.temperature,
                response_format=self.response_format
            )
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=dict_messages,
                temperature=self.temperature
            )
        return response

from minute_gpu_calculation.tensor_context import GlobalTensorContext
from minute_gpu_calculation.dag_scheduler import DAGManager
from minute_gpu_calculation.batch_executor import BatchExecutor
from miner.evaluator import get_or_create_global_context, clean_dynamic_vram_tensors
from miner.evolutionary_miner import macro_expand_formula
from miner.libraries import (
    load_compound_operators,
    load_extra_base_fields as load_extra_base_fields_registry,
)

# ======================================================================
# 📝 1. LLM Pydantic 约束
# ======================================================================
class StateCandidate(BaseModel):
    name: str = Field(..., description="Unique snake_case name for the state (e.g. liquidity_drain_ratio, trend_smoothness_index)")
    category: Literal["Volatility", "Trend", "Liquidity", "MicroStructure"] = Field(..., description="Dimension category of the state.")
    idea_rationale: str = Field(..., description="The logic derived from the trade snapshots and market micro-structure theory.")
    formula: str = Field(..., description="Complete AST formula string, heavily favoring TsMean or EWMA wrapping for smoothness.")

class StateBatch(BaseModel):
    candidates: List[StateCandidate] = Field(..., min_items=5, max_items=10)


def simplify_and_safeguard_formula(formula_str: str) -> str:
    """
    Simplifies and safeguards a formula string to be executed on the PyTorch execution engine.
    Uses a robust Python AST (Abstract Syntax Tree) NodeTransformer to mathematically guarantee 
    Constant Folding (e.g., Sqrt(Max(9, 4)) -> 3) and reciprocal division rewriting across ALL operators.
    """
    import math
    import re
    import ast

    # 0. Auto-capitalize common names to match PyTorch/DAG Engine expectations
    operator_caps = {
        "abs": "Abs", "sign": "Sign", "log": "Log", "sqrt": "Sqrt",
        "indicatormorethanzero": "IndicatorMoreThanZero", "distortlog": "DistortLog", "distortsqrt": "DistortSqrt",
        "ifelse": "IfElse", "fillna": "FillNa", "isna": "IsNa",
        "morethan": "MoreThan", "lessthan": "LessThan", "equal": "Equal",
        "sub": "Sub", "mul": "Mul", "div": "Div", "add": "Add", "prod": "Prod", "minus": "Minus",
        "max": "Max", "min": "Min",
        "ewma": "EWMA", "tsmean": "TsMean", "tsstd": "TsStd", "tsrank": "TsRank",
        "tszscore": "TsZscore", "tscorr": "TsCorr", "tsdelta": "TsDelta",
        "tspct": "TsPct", "tssum": "TsSum", "tsdecay": "TsDecay", "tscov": "TsCov",
        "tsmax": "TsMax", "tsmin": "TsMin", "tsmedian": "TsMedian", "tsmad": "TsMad"
    }
    for op_lower, op_cap in operator_caps.items():
        pattern = r'\b' + re.escape(op_lower) + r'\s*\('
        formula_str = re.sub(pattern, f"{op_cap}(", formula_str, flags=re.IGNORECASE)

    # ==========================================
    # 1. 真正的工业级 AST 常量折叠器
    # ==========================================
    class FormulaConstantFolder(ast.NodeTransformer):
        
        def get_val(self, n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant): return -n.operand.value
            return None

        def format_res(self, res):
            if isinstance(res, float) and res.is_integer(): res = int(res)
            return ast.Constant(value=res)

        def visit_Call(self, node):
            self.generic_visit(node)
            if not isinstance(node.func, ast.Name): return node
            arg_vals = [self.get_val(arg) for arg in node.args]
            if None in arg_vals or len(arg_vals) != len(node.args): return node
                
            func_name = node.func.id.lower()
            res = None
            try:
                if len(arg_vals) == 1:
                    v = arg_vals[0]
                    if func_name == 'sqrt': res = math.sqrt(max(v, 0.0))
                    elif func_name == 'log': res = math.log(v) if v > 0 else 0.0
                    elif func_name == 'abs': res = abs(v)
                    elif func_name == 'sign': res = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
                    elif func_name == 'indicatormorethanzero': res = 1.0 if v > 0 else 0.0
                    elif func_name == 'distortlog': res = (1.0 if v > 0 else (-1.0 if v < 0 else 0.0)) * math.log(1 + abs(v))
                    elif func_name == 'distortsqrt': res = (1.0 if v > 0 else (-1.0 if v < 0 else 0.0)) * math.sqrt(abs(v))
                    elif func_name == 'isna': res = 0.0
                
                elif len(arg_vals) == 2:
                    v1, v2 = arg_vals
                    if func_name in ('mul', 'prod'): res = v1 * v2
                    elif func_name == 'add': res = v1 + v2
                    elif func_name in ('sub', 'minus'): res = v1 - v2
                    elif func_name == 'div': res = v1 / v2 if v2 != 0 else 0.0
                    elif func_name == 'max': res = max(v1, v2)
                    elif func_name == 'min': res = min(v1, v2)
                    elif func_name == 'morethan': res = 1.0 if v1 > v2 else 0.0
                    elif func_name == 'lessthan': res = 1.0 if v1 < v2 else 0.0
                    elif func_name == 'equal': res = 1.0 if v1 == v2 else 0.0
                    elif func_name == 'fillna': res = v1
                
                elif len(arg_vals) == 3:
                    v1, v2, v3 = arg_vals
                    if func_name == 'ifelse': res = v2 if v1 > 0 else v3
                    
                if res is not None: return self.format_res(res)
            except Exception: pass
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and node.value < 0:
                safeguard_expr = f"(close * 0.0 - {abs(node.value)})"
                return ast.parse(safeguard_expr, mode='eval').body
            return node
            
        def visit_UnaryOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                val = node.operand.value
                if isinstance(val, (int, float)):
                    safeguard_expr = f"(close * 0.0 - {abs(val)})"
                    return ast.parse(safeguard_expr, mode='eval').body
            return node
            
        def visit_BinOp(self, node):
            self.generic_visit(node)
            v_left = self.get_val(node.left)
            v_right = self.get_val(node.right)
            
            if v_left is not None and v_right is not None:
                try:
                    res = None
                    if isinstance(node.op, ast.Add): res = v_left + v_right
                    elif isinstance(node.op, ast.Sub): res = v_left - v_right
                    elif isinstance(node.op, ast.Mult): res = v_left * v_right
                    elif isinstance(node.op, ast.Div): res = v_left / v_right if v_right != 0 else 0.0
                    if res is not None: return self.format_res(res)
                except Exception: pass
            
            if isinstance(node.op, ast.Div) and v_right is not None:
                if v_right == 0: return self.format_res(0.0)
                recip = 1.0 / v_right
                return ast.BinOp(left=node.left, op=ast.Mult(), right=self.format_res(recip))
            return node

    try:
        tree = ast.parse(formula_str, mode='eval')
        folder = FormulaConstantFolder()
        tree = folder.visit(tree)
        ast.fix_missing_locations(tree)
        final_str = ast.unparse(tree)
        
        import re
        def format_float(m): return f"{float(m.group(0)):.15f}".rstrip('0')
        final_str = re.sub(r'\b\d+(?:\.\d+)?e[-+]?\d+\b', format_float, final_str, flags=re.IGNORECASE)
        return final_str
    except Exception as e:
        logger.warning(f"AST Parsing fallback activated for formula: {formula_str}. Reason: {e}")
        return formula_str

# ======================================================================
# 🔍 2. 案发现场快照提取
# ======================================================================
def extract_empirical_clues_for_llm(pos_files: list, n_samples: int = 3) -> str:
    """从优质池的特定因子中，提取赚/亏钱时的微观盘口快照。"""
    if not pos_files:
        return "池内无有效仓位文件，请直接依靠常识和理论推理。"

    chosen_file = random.choice(pos_files)
    factor_name = Path(chosen_file).stem
    logger.info(f"🕵️ 从池内独家特征 '{factor_name}' 的实盘仓位中提取交易线索...")
    
    table = pq.read_table(chosen_file)
    df_pos = table.to_pandas()
    
    df_pos = df_pos.reset_index()

    if 'instrument' not in df_pos.columns:
        return f"仓位文件缺少 instrument 字段，无法继续。当前所有列为: {list(df_pos.columns)}"

    df_pos = df_pos[df_pos['instrument'].isin(TARGET_INSTRUMENTS)]

    df_active = df_pos[df_pos[POS_COL_NAME].abs() > 0.9].copy()
    if len(df_active) < n_samples * 2:
        return "该品种池内开仓样本过少，不足以提取线索。"

    df_active['trade_profit'] = df_active[POS_COL_NAME] * df_active[RET_COL_NAME]

    q90 = df_active['trade_profit'].quantile(0.90)
    q10 = df_active['trade_profit'].quantile(0.10)

    wins = df_active[df_active['trade_profit'] >= q90].sample(n=min(n_samples, len(df_active)))
    losses = df_active[df_active['trade_profit'] <= q10].sample(n=min(n_samples, len(df_active)))

    def get_snapshot(row):
        inst = row['instrument']
        t_end = row['index_time']
        t_start = t_end - pd.Timedelta(minutes=30)
        
        panel_path = BASE_PANEL_DIR / inst / f"base_panel_{inst}_parquet.parquet"
        if not panel_path.exists(): return None
        
        try:
            table = pq.read_table(panel_path, filters=[('index_time', '>=', t_start), ('index_time', '<=', t_end)])
            df_raw = table.to_pandas().reset_index()
            keep = ['index_time', 'close', 'volume', 'bid1', 'ask1', 'tick_count']
            df_raw = df_raw[[c for c in keep if c in df_raw.columns]]
            if len(df_raw) < 10: return None
            
            price_change = (df_raw['close'].iloc[-1] / df_raw['close'].iloc[0] - 1) * 10000 
            vol_trend = df_raw['volume'].iloc[-5:].mean() / (df_raw['volume'].iloc[:5].mean() + 1e-8)
            spread_mean = ((df_raw['ask1'] - df_raw['bid1']) / df_raw['close']).mean() * 10000
            
            path_length = df_raw['close'].diff().abs().sum()
            net_move = abs(df_raw['close'].iloc[-1] - df_raw['close'].iloc[0])
            efficiency = net_move / (path_length + 1e-8)
            
            return (
                f"  - 品种: {inst} | 时刻: {t_end.strftime('%Y-%m-%d %H:%M')}\n"
                f"    开仓前30m涨跌幅: {price_change:.1f} bps\n"
                f"    盘面趋势效率(1单边/0震荡): {efficiency:.2f}\n"
                f"    平均买卖价差: {spread_mean:.2f} bps\n"
                f"    近期量能突变比(近5/前5分): {vol_trend:.2f} 倍\n"
            )
        except:
            return None

    clues = []
    clues.append(f"【背景】这是基于基准因子 `{factor_name}` 在 {TARGET_POOL_NAME} 品种上的真实交易切片。")
    clues.append(f"【黄金环境：该策略开仓后暴赚 (Top 10%) 的盘口特征】")
    for _, row in wins.iterrows():
        snap = get_snapshot(row)
        if snap: clues.append(snap)

    clues.append(f"\n【有毒环境：该策略开仓后暴亏 (Bottom 10%) 的盘口特征】")
    for _, row in losses.iterrows():
        snap = get_snapshot(row)
        if snap: clues.append(snap)

    return "\n".join(clues)


# ======================================================================
# ⚖️ 3. 状态非对称收益评估
# ======================================================================
def evaluate_single_regime(df_merged: pd.DataFrame, n_bins: int = 5) -> dict:
    try:
        df_merged['state_bin'] = pd.qcut(df_merged['state_value'], q=n_bins, labels=False, duplicates='drop')
    except Exception as e:
        return {"success": False, "error_msg": f"qcut分箱失败"}
        
    actual_bins = df_merged['state_bin'].nunique()
    if actual_bins < 3: return {"success": False, "error_msg": f"有效分箱数({actual_bins})小于3"}

    df_active = df_merged[df_merged[POS_COL_NAME].abs() >= 0.9].copy()
    
    if len(df_active) < 100: 
        return {"success": False, "error_msg": f"强开仓信号样本过少: {len(df_active)} 次"}

    df_active['pnl'] = df_active[POS_COL_NAME] * df_active[RET_COL_NAME]
    base_ir = df_active['pnl'].mean() / (df_active['pnl'].std() + 1e-8) * np.sqrt(252 * 48)

    bin_irs = {}
    for b in range(actual_bins):
        group = df_active[df_active['state_bin'] == b]
        if len(group) >= 20: 
            bin_irs[b] = group['pnl'].mean() / (group['pnl'].std() + 1e-8) * np.sqrt(252 * 48)
        else:
            bin_irs[b] = 0.0

    ir_values = list(bin_irs.values())
    max_ir = max(ir_values)
    min_ir = min(ir_values)
    mean_ir = np.mean(ir_values)
    
    abs_base = abs(base_ir) + 1e-8
    rel_uplift = (max_ir - base_ir) / abs_base
    rel_spread = (max_ir - min_ir) / abs_base
    cv = np.std(ir_values) / (abs(mean_ir) + 1e-8)

    mutual_info = 0.0
    ks_stat = 0.0
    
    valid_xy = df_active[['state_value', 'pnl']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_xy) > 50:
        try:
            from sklearn.feature_selection import mutual_info_regression
            x = valid_xy['state_value'].values
            y = valid_xy['pnl'].values
            if len(x) > 10000:
                idx = np.random.choice(len(x), 10000, replace=False)
                x_sub, y_sub = x[idx].reshape(-1, 1), y[idx]
            else:
                x_sub, y_sub = x.reshape(-1, 1), y
            mutual_info = mutual_info_regression(x_sub, y_sub, n_neighbors=3, random_state=42)[0]
        except Exception:
            pass 
            
        try:
            from scipy.stats import ks_2samp
            state_win = valid_xy[valid_xy['pnl'] > 0]['state_value'].values
            state_lose = valid_xy[valid_xy['pnl'] < 0]['state_value'].values
            if len(state_win) > 10 and len(state_lose) > 10:
                ks_result = ks_2samp(state_win, state_lose)
                ks_stat = ks_result.statistic
        except Exception:
            pass

    return {
        "success": True,
        "base_ir": base_ir,
        "max_ir": max_ir,         
        "min_ir": min_ir,
        "rel_uplift": rel_uplift,
        "rel_spread": rel_spread,
        "cv": cv,
        "mutual_info": mutual_info,
        "ks_stat": ks_stat
    }

def evaluate_regime_across_pool(df_state: pd.DataFrame, state_col: str, pos_files_sample: list, n_bins: int = 5) -> dict:
    all_details = []
    error_reasons = []
    
    if 'instrument' not in df_state.columns and 'instrument' in df_state.index.names:
        df_state = df_state.reset_index()
    if 'instrument' not in df_state.columns:
        return {"passed": False, "error_msg": "GPU张量缺列 instrument"}
        
    df_state_filtered = df_state[df_state['instrument'].isin(TARGET_INSTRUMENTS)][['index_time', 'instrument', state_col]].copy()
    df_state_filtered.rename(columns={state_col: 'state_value'}, inplace=True)

    for pos_file in pos_files_sample:
        try:
            table = pq.read_table(pos_file)
            df_pos = table.to_pandas()
            if 'instrument' not in df_pos.columns and 'instrument' in df_pos.index.names:
                df_pos = df_pos.reset_index()
            if 'instrument' not in df_pos.columns: continue

            df_pos = df_pos[df_pos['instrument'].isin(TARGET_INSTRUMENTS)]
            df_merged = pd.merge(df_pos, df_state_filtered, on=['index_time', 'instrument'], how='inner').dropna()
            
            res = evaluate_single_regime(df_merged, n_bins)
            if res and res.get("success", False):
                res['factor_name'] = Path(pos_file).stem
                all_details.append(res)
            else:
                error_reasons.append(res.get("error_msg", "评估失败"))
        except Exception as e:
            continue
            
    valid_evals = len(all_details)
    if valid_evals < len(pos_files_sample) * 0.5:
        from collections import Counter
        top_err = Counter(error_reasons).most_common(1)[0][0] if error_reasons else "未知错误"
        return {"passed": False, "error_msg": f"成功率低({valid_evals}/{len(pos_files_sample)})。主要死因: {top_err}"}

    def get_med(key): return float(np.median([d[key] for d in all_details]))

    med_base_ir = get_med('base_ir')
    med_max_ir = get_med('max_ir')
    med_rel_uplift = get_med('rel_uplift')
    med_cv = get_med('cv')
    med_mi = get_med('mutual_info')
    med_ks = get_med('ks_stat')

    passed = (
        (med_mi >= MI_MIN) and 
        (med_ks >= KS_MIN or med_rel_uplift >= UPLIFT_MIN)
    ) 

    detail_log = (
        f"\n      [组间离散] 基准IR: {med_base_ir:.2f} ➡️ Top IR: {med_max_ir:.2f} (提升 {med_rel_uplift*100:+.1f}%) | 变异系数(CV): {med_cv:.3f}"
        f"\n      [独立性检验] 互信息(MI): {med_mi:.5f} | KS-Stat(赢亏分布差异): {med_ks:.4f}"
    )
    
    return {
        "passed": passed,
        "valid_evals": valid_evals,
        "detail_log": detail_log,
        "metrics": {
            "base_ir": med_base_ir, "max_ir": med_max_ir,
            "rel_uplift": med_rel_uplift, "cv": med_cv,
            "mutual_info": med_mi, "ks_stat": med_ks
        }
    }


# ======================================================================
# 🚀 4. 大模型提词生成
# ======================================================================
def load_extra_base_fields() -> str:
    try:
        data = load_extra_base_fields_registry()
        if not data:
            return ""
        return f"\n【可用的扩展基础变量 (Extra Base Fields)】: {', '.join(list(data.keys()))}\n(可直接在公式中使用这些名称)"
    except Exception as e:
        logger.warning(f"读取 extra_base_fields 失败: {e}")
        return ""

def load_compound_operators_desc(sample_size: int = 5) -> str:
    try:
        data = load_compound_operators()
        if data:
            keys = random.sample(list(data.keys()), min(sample_size, len(data)))
            lines = [f'- `{k}`: {data[k].get("description")}' for k in keys]
            return "\n【🚀 复合高级算子 (可直接调用)】\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"读取 compound_operators 失败: {e}")
    return ""

def generate_states_via_llm(existing_names: list, empirical_clues: str) -> List[dict]:
    extra_fields_text = load_extra_base_fields()
    compound_ops_text = load_compound_operators_desc(sample_size=6)
    
    prompt = f"""# Role: 高频量价因子与微观市场状态 (Micro-Structure Regime) 挖掘专家 & 表达式工程师

你现在是一个顶尖的量化对冲基金 Meta-Model（元模型）研究员。你的任务是挖掘【{TARGET_POOL_NAME} 宇宙专属的微观市场状态变量 Market States】。
好的状态变量能在底层因子池的泛化测试中，将收益率客观切分出云泥之别！

为了帮助你构建，我提取了某极其赚钱的真实基准策略，在 {TARGET_POOL_NAME} 品种上的开仓切片（开仓前30分钟盘口快照）：
{empirical_clues}

## 🎯 任务要求
生成 5 到 10 个微观状态变量算子公式。
⚠️ 本轮挖掘**绝对禁止**构造跨日或长周期的宏观特征！请将所有的创造力集中在日内的微观盘口博弈上（例如：瞬时流动性枯竭、买卖压极度失衡、量价异常背离、高频波动率脉冲、盘面震荡熵增等）。

---

## 📚 1. 基础数据字段池 (Base Fields)
我们的底层数据频率为 **1 分钟 (1-Minute Bar)**。所有的时间窗口参数 `window` 的单位均为 **“分钟”**。

### 💥 宇宙法则：纯微观盘口宇宙
* **时间窗口 (window)**：必须严格限制在 **5 到 240 分钟** 之间！绝对不允许出现大于 240 的 window 参数！
* **可用数据**：只能使用未复权原始数据：`close`, `open`, `high`, `low`, `volume`, `vwap`, `bid1`, `ask1`, `tick_count`, `open_interest` 以及带 `_trd` 后缀的真实成交价。
* **禁用数据**：绝对禁止使用任何带 `_adj` 后缀的复权数据！
{extra_fields_text}

---

## 🛠 2. 算子函数库 (Operator Library)
公式支持直接使用 Python 原生运算符：`+`, `-`, `*`, `/`, `>`, `<`, `==`，以及一元负号 `-x`。

⚠️ 【致命警告】：只能使用以下列出的函数！绝对禁止自行发明如 `Return`, `Pow`, `Subtract` 等函数！
收益率请用 `TsPct`，减法请用 `-` 或 `Minus`，平方请用乘法 `*` 或 `Abs`！

### 一元数学算子 (Unary Ops)
* `Abs(x)`: 绝对值
* `Log(x)`: 自然对数 (自带防 <=0 崩溃处理)
* `Sqrt(x)`: 平方根
* `Sign(x)`: 符号函数 (1, 0, -1)
* `IndicatorMoreThanZero(x)`: 若 x > 0 返回 1.0，否则返回 0.0
* `DistortLog(x)`: 保留符号的对数 `sign(x) * log(1 + abs(x))`
* `DistortSqrt(x)`: 保留符号的平方根 `sign(x) * sqrt(abs(x))`

### 特殊逻辑算子 (Logical Ops)
* `IfElse(cond, x, y)`: 核心条件算子。若 `cond` > 0 返回 `x`，否则返回 `y`。千万不要用 Python 的 if-else 语句。
* `FillNa(x, val)`: 若 `x` 是 NaN，用 `val` (可为常数或另一公式) 填补。
* `IsNa(x)`: 判断是否为空值，是返回 1.0，否返回 0.0。

### 时序基础与位移算子 (Time-Series Basic)
* `TsDelay(x, window)`: 获取 `window` 分钟前的值。
* `TsDelta(x, window)`: 差分。
* `TsPct(x, window)`: 收益率近似。
* `TsFirst(x, window)` / `TsLast(x, window)`。

### 时序线性聚合 (TS Linear)
* `TsSum(x, window)`: 滚动求和。
* `TsMean(x, window)`: 滚动均值。
* `TsStd(x, window)`: 滚动标准差。
* `TsZscore(x, window)`: 时序标准化。
* `EWMA(x, halflife)`: 指数加权移动平均。
* `TsDecay(x, window)`: 线性权重衰减移动平均。

### 时序非线性聚合 (TS Non-Linear - 刻画微观极端分位极佳)
* `TsMax(x, window)` / `TsMin(x, window)`: 滚动最大/最小值。
* `TsMaxDiff(x, window)` / `TsMinDiff(x, window)`: 当前值减去最大/最小值。
* `TsArgMax(x, window)` / `TsArgMin(x, window)`: 最大/最小值所在相对位置 (输出 0~1 的比例，1 代表最远，0 代表当前)。
* `TsMedian(x, window)`: 滚动中位数。
* `TsQuantile(x, window, q)`: 滚动分位数 (q 在 0~1 之间，如 0.5 为中位数)。
* `TsMad(x, window)`: 绝对中位差。
* `TsRank(x, window)`: 当前值在窗口内的百分位排名 (输出 0~1)。

### 高阶统计算子 (TS High-Order)
* `TsSkewness(x, window)`: 滚动偏度。
* `TsKurtosis(x, window)`: 滚动峰度。
* `TsEntropy(x, window)`: 滚动信息熵 (完美刻画震荡程度)。
* `TsCv(x, window)`: 变异系数 (Std / Mean)。
* `TsIr(x, window)`: 信息比率 (Mean / Std)。

### 二元关系时序算子 (TS Binary)
* `TsCov(x, y, window)`: 滚动协方差。
* `TsCorr(x, y, window)`: 滚动皮尔逊相关系数。
* `TsFacNeutralCoef(x, y, window)`: 滚动回归系数 Beta。
* `TsFacNeutralRes(x, y, window)`: 滚动回归残差 (去量纲/中性化后的异象信息)。
{compound_ops_text}

---
【输出要求】
1. 状态变量为了保持稳定性，公式最外层强烈建议包裹 `TsMean` 或 `EWMA`。
2. 请确保你设计的变量名称唯一，勿与已有名称重复: {existing_names[:15]}...
3. 必须严格输出对应的 JSON 格式 (StateBatch 结构)。
"""
    logger.info("🤖 请求大模型进行数据驱动的状态变量生成 (搭载全量语法引擎与纯微观宇宙隔离)...")
    try:
        chat_obj = Chat(
            model=LLM_MODEL_NAME,
            messages=[Message(role="user", content=prompt)],
            temperature=0.7,
            response_format=StateBatch
        )
        response = chat_obj.create()
        # 脱敏修复：统一使用原生结构解析
        raw_content = response.choices[0].message.content
        clean_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content.strip(), re.DOTALL)
        target_str = clean_json.group(1) if clean_json else raw_content
        
        candidates = json.loads(target_str).get("candidates", [])
        logger.info(f"✨ 大模型成功构建了 {len(candidates)} 个高逻辑微观状态变量！")
        return candidates
    except Exception as e:
        logger.error(f"大模型调用或解析失败: {e}")
        return []

def start_state_miner(rounds=3):
    logger.info("=========================================================")
    logger.info(f"🌟🎯 启动 [数据驱动·非对称市场状态挖掘系统] ({FOCUS_UNIVERSE} 版) 🎯🌟")
    STATE_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    registry = {}
    if STATE_REGISTRY_FILE.exists():
        try:
            with open(STATE_REGISTRY_FILE, "r") as f: registry = json.load(f)
        except: pass

    if not BEST_FEATURES_FILE.exists():
        logger.error(f"❌ 找不到池子白名单 {BEST_FEATURES_FILE}")
        return
        
    with open(BEST_FEATURES_FILE, "r") as f:
        best_features = json.load(f)

    pos_files = [str(f) for f in POSITIONS_DIR.glob("*.parquet") if f.stem in best_features]
    if not pos_files:
        return

    try:
        logger.info(f"⏳ 正在预热加载 {FOCUS_UNIVERSE} 宇宙的全局 GPU 张量上下文 (一次加载，永久长驻)...")
        global_ctx = get_or_create_global_context(focus_universe=FOCUS_UNIVERSE)
    except Exception as e:
        logger.error(f"❌ 初始化 GPU 上下文失败: {e}")
        return

    for r in range(rounds):
        logger.info(f"\n========== 🔄 第 {r+1}/{rounds} 轮环境发掘 ==========")
        
        benchmark_pos_file = random.choice(pos_files)
        clues = extract_empirical_clues_for_llm([benchmark_pos_file], n_samples=3)
        
        candidates = generate_states_via_llm(list(registry.keys()), clues)
        if not candidates: continue
            
        dag = DAGManager()
        target_nodes = {}
        compiled_formulas = {}
        for c in candidates:
            try:
                expanded_formula = macro_expand_formula(c["formula"])
                compiled_formula = simplify_and_safeguard_formula(expanded_formula)
                target_nodes[c["name"]] = dag.add_formula(compiled_formula)
                compiled_formulas[c["name"]] = compiled_formula
            except Exception as exc:
                logger.warning(f"⚠️ 状态公式 {c.get('name', '<unnamed>')} 展开或编译失败: {exc}")
                
        if not target_nodes: continue

        df_final = None
        executor = BatchExecutor(dag, global_ctx, max_concurrent_insts=4)
        for nid in target_nodes.values(): 
            if nid in executor.out_degrees: executor.out_degrees[nid] += 999999
        
        executor.run()
        df_final = executor.extract_multiple_signals(target_nodes_dict=target_nodes, mode="main_only", target_start="2024-01-08", target_end="2024-12-31")

        # 💥 脱敏/防呆修复：彻底清理了多余的重复回收逻辑
        clean_dynamic_vram_tensors(global_ctx, [])
        if 'executor' in locals(): del executor
        if 'dag' in locals(): del dag
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        if df_final is None or df_final.empty:
            logger.warning("⚠️ 本轮未能成功生成合法的状态张量，直接跳过评估阶段！")
            continue

        # 5. 跨池子泛化评估 (Median IR Spread)
        eval_sample_size = min(len(pos_files), 100)
        pos_files_sample = random.sample(pos_files, eval_sample_size)
        logger.info(f"📊 计算状态变量在 {eval_sample_size} 个顶级特征上的 泛化区分度 (Median Regime Spread)...")
        
        results = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(evaluate_regime_across_pool, df_final, name, pos_files_sample, 5): name for name in target_nodes}
            for fut in futures:
                name = futures[fut]
                results[name] = fut.result()

        del df_final
        gc.collect()

        # 6. 沉淀与淘汰 (强制白盒打印)
        for cand in candidates:
            name = cand["name"]
            res = results.get(name, {})
            
            # 计分板
            if "metrics" in res:
                m = res["metrics"]
                logger.info(f"  🔸 {name}: Base IR={m['base_ir']:.2f} | Max IR={m['max_ir']:.2f} | "
                            f"Uplift={m['rel_uplift']*100:+.1f}% | CV={m['cv']:.3f} | "
                            f"MI={m['mutual_info']:.5f} | KS={m['ks_stat']:.4f}")
            else:
                logger.warning(f"  ⚠️ {name}: 评估完全失败 -> {res.get('error_msg', '计算崩溃')}")

            # 获奖日志
            if res.get("passed", False) and "metrics" in res:
                m = res["metrics"]
                compiled_formula = compiled_formulas[name]
                logger.info(f"\n🏆 [绝佳泛化环境入选] : {name} | 维度: {cand['category']}")
                logger.info(f"   => 逻辑: {cand['idea_rationale']}")
                logger.info(f"   => 展开后公式: {compiled_formula}")
                logger.info(f"   => Base IR: {m['base_ir']:.2f} | Uplift: {m['rel_uplift']*100:+.1f}% | "
                            f"MI: {m['mutual_info']:.5f} | KS: {m['ks_stat']:.4f}")

                registry[name] = {
                                    "category": cand["category"],
                                    "formula": compiled_formula,
                                    "rationale": cand["idea_rationale"],
                                    "eval_pool_size": res.get('valid_evals', 0),
                                    "stats": res,
                                    "timestamp": datetime.now().isoformat()
                                }
                
        # 持久化
        with open(STATE_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

            
    logger.info(f"\n🎉 状态发掘阶段结束！优质条件特征库总量: {len(registry)}，已完整保存至 {STATE_REGISTRY_FILE}。")

if __name__ == "__main__":
    start_state_miner(rounds=300)
