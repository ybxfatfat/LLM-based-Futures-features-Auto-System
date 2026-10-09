"""
evolutionary_miner.py - Multi-Layer Evolutionary Futures Alpha Mining Sandbox.

Enforces a layered genetic GP-like progression with LLM:
1. Base Candidate Layer (Raw Ingredients)
2. Scale-Relativization Layer (Cross-Sectional Comparability)
3. Distribution Transformation Layer (Outlier/Skewness Reduction)
4. Advanced Temporal & Multi-Variable Interaction Layer (Predictive Patterns)
5. Smoothing & Noise Filtering Layer (Turnover/Stability Control)

Implements Natural Selection: when the pool size exceeds the limit, the LLM reads
formulas, rationales, and full metrics (linear, non-linear, and monotonicity)
and elects exactly half to keep.
"""

from __future__ import annotations

import os
import sys
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any

# Keep research-state files under miner/, while importing project modules from
# the repository root through stable package-qualified paths.
project_root = Path(__file__).resolve().parent
repository_root = project_root.parent
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

# Import evaluator tools
from miner.evaluator import evaluate_factor_batch, get_or_create_global_context
from miner.libraries import (
    load_compound_operators,
    load_extra_base_fields,
    load_foundation_features,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evolutionary_miner")

def load_compound_operators_desc(sample_size: int = 10) -> str:
    """
    Loads compound operators from libraries/compound_operators.json and randomly 
    samples them to prevent LLM attention bias and force genetic diversity (Operator Dropout).
    """
    try:
        data = load_compound_operators()
        if data:
            import random
            keys = list(data.keys())
            # 💥 核心逻辑：随机抽取 N 个复合算子 (防止 Prompt 冗长和注意力集中)
            sampled_keys = random.sample(keys, min(sample_size, len(keys)))
            
            lines = []
            for k in sampled_keys:
                info = data[k]
                lines.append(f'- `{k}` (参数: {", ".join(info.get("parameters", []))}): {info.get("description")}')
            
            # 加一句提示，告诉 LLM 这些是本轮的高级特供武器
            header = f"（本轮随机解锁了 {len(sampled_keys)} 个高阶算子，强烈建议尝试使用它们）\n"
            return header + "\n".join(lines)
    except Exception as e_op:
        pass
    return "- `EWMAC`: 指数加权移动平均交叉\n- `Path_Efficiency`: 滚动路径效率"

# ======================================================================
# LLM Client Wrapper (OpenAI / DeepSeek Compatible)
# ======================================================================
from pydantic import BaseModel, Field
from openai import OpenAI
import os

# 实例化标准 OpenAI 客户端
# 你可以在系统环境变量中配置 OPENAI_API_KEY。
# 如果想用 DeepSeek，只需配置 OPENAI_BASE_URL="https://api.deepseek.com/v1"
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
    """
    轻量级 LLM 包装器，支持 Pydantic 结构化输出 (Structured Outputs)
    """
    def __init__(self, model: str, messages: list[Message], temperature: float = 0.7, response_format=None):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.response_format = response_format

    def create(self):
        dict_messages = [m.to_dict() for m in self.messages]
        
        # 利用官方的 beta.parse 支持强类型 Pydantic Schema 输出
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
        

# ======================================================================
# GLOBAL CONFIGURATION
# ======================================================================
# Target Asset Universe: "Commodity" or "Index"
# - "Commodity": Focus entirely on Commodity Futures (ME, AG, IN, EN, TR). Only load and run on these.
# - "Index"    : Focus entirely on Stock Index Futures (SI). Only load and run on these.
FOCUS_UNIVERSE = "Commodity"

# Global holding horizon / lookahead period in minutes (duration for future label returns)
HOLDING_HORIZON = 30

LLM_MODEL_NAME = "gemini-3-1-pro"

# Define Local State File
STATE_FILE = project_root / f"evolution_state_{FOCUS_UNIVERSE.lower()}.json"
POOL_MAX_SIZE = 100  # Strict upper limit of candidates per layer

# ======================================================================
# 1. Pydantic Action and State Schemas per Layer
# ======================================================================

# Layer 1: Base Candidate (Raw Ingredients)
class L1Candidate(BaseModel):
    name: str = Field(..., description="Unique descriptive snake_case name for this raw factor. Do NOT use generic names like L1_F1. Instead combine rationales, e.g., oir_volume_friction or spread_range_deviation.")
    idea_rationale: str = Field(..., description="Economic hypothesis and micro-structural driver behind this ingredient.")
    formula: str = Field(..., description="Raw formula combining fields or foundation features, e.g. 'OIR' or '(close_trd - vwap) / (vwap + 1e-8)'")

class L1Batch(BaseModel):
    candidates: List[L1Candidate] = Field(..., min_items=5, max_items=16)

# Layer 2: Relativization Action
class L2Action(BaseModel):
    name: str = Field(..., description="Unique descriptive snake_case name, e.g. oir_vol_friction_normalized or price_dev_zscore. Do NOT use generic L2_F1.")
    source_factor: str = Field(..., description="Name of the L1 candidate to wrap as input, e.g. oir_volume_friction")
    idea_rationale: str = Field(..., description="Reasoning for applying the chosen relativization logic.")
    formula: str = Field(..., description="Complete mathematical formula. You MUST use the exact ID string from `source_factor` as the variable name in your formula. E.g., if source_factor is 'oir_ratio', formula should be 'oir_ratio / (TsMean(oir_ratio, 60) + 1e-8)'.")

class L2Batch(BaseModel):
    candidates: List[L2Action] = Field(..., min_items=5, max_items=16)

# Layer 3: Distribution Transformation Action
class L3Action(BaseModel):
    name: str = Field(..., description="Unique descriptive snake_case name, e.g. oir_vol_friction_log or price_dev_log. Do NOT use generic L3_F1.")
    source_factor: str = Field(..., description="Name of L2 candidate to wrap as input, e.g. oir_vol_friction_normalized")
    idea_rationale: str = Field(..., description="Economic reasoning for reshaping the distribution.")
    formula: str = Field(..., description="Complete mathematical formula. You MUST use the exact ID string from `source_factor` as the variable name. E.g., if source_factor is 'vol_trend', formula could be 'DistortLog(vol_trend)' or just 'vol_trend' for NoOp.")

class L3Batch(BaseModel):
    candidates: List[L3Action] = Field(..., min_items=5, max_items=16)

# Layer 4: Advanced Temporal & Multi-Variable Interaction Action
class L4Action(BaseModel):
    name: str = Field(..., description="Unique descriptive snake_case name, e.g. oir_vol_friction_trend or price_dev_neutral_res. Do NOT use generic L4_F1.")
    source_factor: str = Field(..., description="Name of L3 candidate to wrap as input, e.g. oir_vol_friction_log")
    idea_rationale: str = Field(..., description="Reasoning for temporal pattern extraction or correlation.")
    formula: str = Field(..., description="Complete mathematical formula. You MUST use the exact ID string from `source_factor` as the variable name. You can freely combine it with other base fields like volume. E.g., if source_factor is 'price_dev', formula could be 'TsCorr(price_dev, volume, 60)'.")
    pattern_op: Optional[str] = Field("NoOp", description="Set to 'NoOp' if you want to skip adding any temporal operators and keep the parent formula unchanged (Early Exit). Otherwise, write the active operator name.")

class L4Batch(BaseModel):
    candidates: List[L4Action] = Field(..., min_items=5, max_items=16)

# Layer 5: Smoothing & Noise Filtering Action (with Window Variants)
class L5WindowVariant(BaseModel):
    window_label: str = Field(..., description="Short descriptive label for this variant, e.g. 'ultra_fast_5min', 'slow_120min', 'adaptive_60min'. Must be unique within this factor's variants.")
    formula: str = Field(..., description="Complete self-contained variant formula with adjusted window parameters. You MUST use the exact ID string from `source_factor` as the variable name. Must be a fully executable expression, NOT just a number.")
    window_rationale: str = Field(..., description="1-sentence economic reasoning explaining why this specific window combination is worth testing, relative to the factor's logic.")

class L5Action(BaseModel):
    name: str = Field(..., description="Unique descriptive snake_case name, e.g. oir_vol_friction_smoothed or price_dev_trend_filtered. Do NOT use generic L5_F1.")
    source_factor: str = Field(..., description="Name of L4 candidate to wrap as input, e.g. oir_vol_friction_trend")
    idea_rationale: str = Field(..., description="Reasoning for applying smoothing/filtering.")
    formula: str = Field(..., description="Base smoothed formula. You MUST use the exact ID string from `source_factor` as the variable name. E.g., 'EWMA(my_factor, 10)' or 'my_factor' if NoOp.")
    window_variants: List[L5WindowVariant] = Field(default_factory=list, description="MANDATORY: 5-8 alternative window parameter variants for this factor. Each variant MUST have a meaningfully different window value (NOT just ±1 from original!) and a complete executable formula.")

class L5Batch(BaseModel):
    candidates: List[L5Action] = Field(..., min_items=5, max_items=18)

# ======================================================================
# Natural Selection Decision Schema (打分淘汰制)
# ======================================================================
class FactorEvaluation(BaseModel):
    name: str = Field(..., description="候选因子的唯一名称")
    score: int = Field(..., description="对因子的保留价值打分 (0-10分)。10分代表逻辑新颖、非线性指标(MI/ANOVA)优异，应优先保留；0-3分代表与其他因子严重同质化冗余、或者属于纯噪声。")
    reason: str = Field(..., description="一句简短的打分理由，例如'高MI且逻辑独特'或'与xxx因子公式高度冗余且IC更低'")

class PoolScoringDecision(BaseModel):
    evaluations: List[FactorEvaluation] = Field(..., description="对提供的所有候选因子的打分评估列表")

# ======================================================================
# 2. State Loading & Persistence
# ======================================================================

def load_evolution_state() -> dict:
    """Loads evolutionary gene pool state or initializes the starter configuration."""
    state = {}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read evolutionary state: {e}. Resetting.")

    # Load Failure Knowledge Bank from research_log if available
    research_lessons = []
    log_path = project_root / f"research_log_{FOCUS_UNIVERSE.lower()}.json"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                r_log = json.load(f)
                research_lessons = r_log.get("kb_lessons", [])
        except Exception:
            pass

    # Ensure backward compatibility and set default structured fields
    state.setdefault("active_round", 1)
    state.setdefault("current_layer", 1)
    state.setdefault("pools", {})
    state["pools"].setdefault("L2_pool", [])
    state["pools"].setdefault("L3_pool", [])
    state["pools"].setdefault("L4_pool", [])
    state["pools"].setdefault("L5_pool", [])
    state["focus_universe"] = FOCUS_UNIVERSE
    state.setdefault("active_pool", [])
    state.setdefault("history", [])
    state.setdefault("kb_lessons", research_lessons or [])
    return state

def save_evolution_state(state: dict):
    """Persists current state back to disk."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

# ======================================================================
# 3. Dynamic Template-wrapping Formula Compiler
# ======================================================================

def split_nested_arguments(arg_str: str) -> list[str]:
    """Splits a comma-separated function argument string at depth 0, ignoring commas in nested parentheses."""
    args = []
    current_arg = []
    depth = 0
    for char in arg_str:
        if char == "(":
            depth += 1
            current_arg.append(char)
        elif char == ")":
            depth -= 1
            current_arg.append(char)
        elif char == "," and depth == 0:
            args.append("".join(current_arg).strip())
            current_arg = []
        else:
            current_arg.append(char)
    if current_arg:
        args.append("".join(current_arg).strip())
    return args


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
            """安全提取节点中的常数值（包含负数）"""
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
                return -n.operand.value
            return None

        def format_res(self, res):
            """将计算结果包装回 AST Constant 节点"""
            if isinstance(res, float) and res.is_integer():
                res = int(res)
            return ast.Constant(value=res)

        def visit_Call(self, node):
            # 自底向上遍历，先折叠内部嵌套（如 Sqrt(IfElse(...))）
            self.generic_visit(node)
            
            if not isinstance(node.func, ast.Name):
                return node
                
            arg_vals = [self.get_val(arg) for arg in node.args]
            # 如果参数里有任何一个不是纯常数（比如包含了 volume），则放弃折叠，交给 GPU
            if None in arg_vals or len(arg_vals) != len(node.args):
                return node
                
            func_name = node.func.id.lower()
            res = None
            try:
                # 动态映射所有函数求值
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
                    
                if res is not None:
                    return self.format_res(res)
            except Exception:
                pass
                
            return node

        # 💥拦截直接写在 AST 常量节点里的负数
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and node.value < 0:
                # 将负数 -X 转化为语法树节点: (close * 0.0 - abs(X))
                safeguard_expr = f"(close * 0.0 - {abs(node.value)})"
                return ast.parse(safeguard_expr, mode='eval').body
            return node
            
        # 💥拦截通过负号 (UnaryOp) 生成的负数
        def visit_UnaryOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                val = node.operand.value
                if isinstance(val, (int, float)):
                    # 将 -X 转化为语法树节点: (close * 0.0 - abs(X))
                    safeguard_expr = f"(close * 0.0 - {abs(val)})"
                    return ast.parse(safeguard_expr, mode='eval').body
            return node
            
        def visit_BinOp(self, node):
            # 处理原生的 Python 符号 (+, -, *, /)
            self.generic_visit(node)
            
            v_left = self.get_val(node.left)
            v_right = self.get_val(node.right)
            
            # 1. 二元纯数字符号计算
            if v_left is not None and v_right is not None:
                try:
                    res = None
                    if isinstance(node.op, ast.Add): res = v_left + v_right
                    elif isinstance(node.op, ast.Sub): res = v_left - v_right
                    elif isinstance(node.op, ast.Mult): res = v_left * v_right
                    elif isinstance(node.op, ast.Div): res = v_left / v_right if v_right != 0 else 0.0
                    
                    if res is not None:
                        return self.format_res(res)
                except Exception:
                    pass
            
            # 2. 常数倒数替换防御 (A / 2 -> A * 0.5)
            if isinstance(node.op, ast.Div) and v_right is not None:
                if v_right == 0:
                    return self.format_res(0.0)
                recip = 1.0 / v_right
                return ast.BinOp(left=node.left, op=ast.Mult(), right=self.format_res(recip))
                
            return node

    # ==========================================
    # 2. 执行 AST 转换
    # ==========================================
    try:
        # 将字符串解析为真正的 Python 语法树
        tree = ast.parse(formula_str, mode='eval')
        # 应用遍历与折叠
        folder = FormulaConstantFolder()
        tree = folder.visit(tree)
        # 修复位置信息并转换回字符串
        ast.fix_missing_locations(tree)
        final_str = ast.unparse(tree)
        
        # 💥 终极拔除隐患：防止 ast.unparse 自动生成科学计数法 1e-08，导致底层 DAG 误解析为 1 * e - 8 (引发标量 Prod 崩溃)
        import re
        def format_float(m):
            return f"{float(m.group(0)):.15f}".rstrip('0')
            
        final_str = re.sub(r'\b\d+(?:\.\d+)?e[-+]?\d+\b', format_float, final_str, flags=re.IGNORECASE)
        
        return final_str
        
    except Exception as e:
        logger.warning(f"AST Parsing fallback activated for formula: {formula_str}. Reason: {e}")
        return formula_str


def macro_expand_formula(formula_str: str) -> str:
    """
    Recursively expands all registered custom foundation features and multi-step compound operators in a formula string.
    Ensures that the output formula contains ONLY primitive execution tensors understood by the GPU backtest engine.
    """
    custom_feats = load_foundation_features()
    custom_ops = load_compound_operators()

    expanded = formula_str

    # 1. Expand Custom Foundation Features recursively using word boundaries
    found_feat_to_expand = True
    while found_feat_to_expand:
        found_feat_to_expand = False
        for feat_name, feat_info in custom_feats.items():
            pattern = r'\b' + re.escape(feat_name) + r'\b'
            if re.search(pattern, expanded):
                # Expand with enclosing parentheses to safeguard operator precedence
                expanded = re.sub(pattern, f"({feat_info['formula']})", expanded)
                found_feat_to_expand = True

    # 2. Expand Custom Compound Operators recursively
    found_op_to_expand = True
    while found_op_to_expand:
        found_op_to_expand = False
        for op_name, op_info in custom_ops.items():
            start_pattern = op_name + r'\('
            match = re.search(start_pattern, expanded)
            if match:
                found_op_to_expand = True
                start_idx = match.start()
                # Find matching closing parenthesis for this call
                paren_count = 1
                curr_idx = match.end()
                while paren_count > 0 and curr_idx < len(expanded):
                    if expanded[curr_idx] == '(':
                        paren_count += 1
                    elif expanded[curr_idx] == ')':
                        paren_count -= 1
                    curr_idx += 1

                if paren_count == 0:
                    # Found matching closing parenthesis
                    full_call = expanded[start_idx:curr_idx]
                    arg_str = expanded[match.end():curr_idx-1]

                    # Split arguments at depth 0
                    args = split_nested_arguments(arg_str)

                    # Map positional arguments to operator parameters
                    params = op_info["parameters"] # e.g. ["input", "window"]
                    template = op_info["formula_template"] # e.g. "Minus(EWMA({input}, {window}), ...)"

                    # Substitute parameters
                    compiled_op = template
                    for i, param in enumerate(params):
                        val = args[i] if i < len(args) else ""
                        compiled_op = compiled_op.replace(f"{{{param}}}", val)

                    # Substitute full_call with compiled_op in the formula
                    expanded = expanded[:start_idx] + compiled_op + expanded[curr_idx:]
                else:
                    logger.warning(f"Unbalanced parentheses found when scanning compound operator {op_name} in: {expanded}")
                    found_op_to_expand = False

    # Apply dynamic simplification and safeguarding to prevent division-by-scalar and scalar-only operator call bugs on PyTorch
    expanded = simplify_and_safeguard_formula(expanded)

    return expanded


def compile_evolution_formula(action_data: dict, previous_formulas: dict[str, str]) -> str:
    """Recursively wraps previous formulas by substituting the parent's formula string inside the child's formula expression."""
    if "formula" not in action_data:
        raise ValueError("Missing 'formula' field in candidate action.")

    formula_str = action_data["formula"]
    parent_name = action_data.get("source_factor")

    if parent_name:
        if parent_name not in previous_formulas:
            raise ValueError(f"Parent factor source '{parent_name}' is missing in the previous gene pool.")
        parent_formula = previous_formulas[parent_name]
        # Recursively substitute parent_name inside child formula string as a standalone token
        pattern = r'\b' + re.escape(parent_name) + r'\b'
        formula_str = re.sub(pattern, f"({parent_formula})", formula_str)

    return formula_str

# ======================================================================
# 4. GPU/CPU Integration & Evaluation Execution
# ======================================================================

def evaluate_evolutionary_pool(pool: List[dict], previous_formulas: dict[str, str], focus_universe: str = "Commodity") -> List[dict]:
    """Compiles, recursively macro-expands, and executes real GPU+CPU evaluations."""
    # Find factors that need evaluation
    unevaluated = [f for f in pool if f.get("metrics") is None]
    if not unevaluated:
        logger.info("Active gene pool already has complete up-to-date metrics.")
        return pool

    # Build and recursively macro-expand formulas
    formulas_to_run = {}
    for f in unevaluated:
        try:
            compiled = compile_evolution_formula(f["action"], previous_formulas)
            # Apply dynamic recursive macro-expansion to convert compound ops and foundation features into execution primitives
            compiled_expanded = macro_expand_formula(compiled)
            f["formula"] = compiled_expanded
            formulas_to_run[f["name"]] = compiled_expanded
            logger.info(f"🧬  Compiled: {f['name']} -> Expanded: '{compiled_expanded}'")
        except Exception as ce:
            logger.error(f"Compilation crash for {f['name']}: {ce}")
            f["metrics"] = {
                "passed": False,
                "error_msg": f"Formula compilation crash: {ce}",
                "ic": 0.0, "full_ic": 0.0, "ir": 0.0, "win_rate": 0.0, "q_spread": 0.0,
                "full_mi": 0.0, "mean_mi": 0.0, "full_anova": 0.0, "mean_anova": 0.0,
                "group_spearman": 0.0, "feedback_log": f"AST 包装转换崩溃: {ce}",
                "tail_low_adj_ret": 0.0, "tail_high_adj_ret": 0.0,
                "tail_low_glob_adj_ret": 0.0, "tail_high_glob_adj_ret": 0.0
            }

    # Extract valid formulas to execute on GPU
    valid_formulas = {name: formula for name, formula in formulas_to_run.items() if name not in [f["name"] for f in unevaluated if f.get("metrics") is not None]}

    if valid_formulas:
        try:
            logger.info(f"⏳ Dispatching {len(valid_formulas)} evolutionary formulas to GPU executor and CPU AlphaProfiler...")
            batch_metrics = evaluate_factor_batch(valid_formulas, focus_universe=focus_universe, holding_horizon=HOLDING_HORIZON)
        except Exception as ex:
            logger.error(f"Evaluation crashed at outer boundary: {ex}")
            batch_metrics = {}

        # Save metrics back to factors
        for f in unevaluated:
            if f["name"] in batch_metrics:
                f["metrics"] = batch_metrics[f["name"]]
                logger.info(f"  √ Evaluated: {f['name']} -> Mean Period IC: {f['metrics'].get('ic', 0.0):.4f}, Full IC: {f['metrics'].get('full_ic', 0.0):.4f}, IR: {f['metrics'].get('ir', 0.0):.3f}")
            elif f.get("metrics") is None:
                # Execution failed
                f["metrics"] = {
                    "passed": False,
                    "error_msg": "Evaluation engine failed to return metrics",
                    "ic": 0.0, "full_ic": 0.0, "ir": 0.0, "win_rate": 0.0, "q_spread": 0.0,
                    "full_mi": 0.0, "mean_mi": 0.0, "full_anova": 0.0, "mean_anova": 0.0,
                    "group_spearman": 0.0, "feedback_log": "计算引擎返回空，请检查公式变量是否溢出或不匹配。",
                    "tail_low_adj_ret": 0.0, "tail_high_adj_ret": 0.0,
                    "tail_low_glob_adj_ret": 0.0, "tail_high_glob_adj_ret": 0.0
                }
    return pool

# ======================================================================
# 5. Persistent Material Pool Natural Selection & Name Uniqueness
# ======================================================================

def make_name_unique(name: str, existing_names: set[str]) -> str:
    """Ensures a factor name is unique by appending a numeric suffix if needed."""
    if name not in existing_names:
        return name

    base_name = name
    suffix = 1
    match = re.search(r'_(\d+)$', name)
    if match:
        suffix = int(match.group(1)) + 1
        base_name = name[:match.start()]

    while f"{base_name}_{suffix}" in existing_names:
        suffix += 1

    return f"{base_name}_{suffix}"

def prune_pool_via_llm_loop(pool: List[dict], layer_num: int, target_size: int = 25, max_attempts: int = 10) -> List[dict]:
    """
    Calls LLM to score all factors in the pool from 0-10. 
    Sorts factors based on LLM scores (and absolute full_ic as tie-breaker) to retain exactly target_size.
    """
    logger.info(f"⚡ [Pruning Pool L{layer_num+1}] Pool size ({len(pool)}) exceeds limit. Triggering LLM Scoring Pruning...")

    # 1. 精简输入给 LLM 的指标，避免注意力分散和 Token 浪费
    candidates_details = []
    for f in pool:
        m = f.get("metrics", {})
        det = (
            f"- Name/ID: {f['name']}\n"
            f"  公式: {f['formula']}\n"
            f"  机理: {f['rationale']}\n"
            f"  主要绩效指标:\n"
            f"    - Mean Period IC (指定长度窗口 IC 均值) = {m.get('ic', 0.0):.5f}\n"
            f"    - Full Year IC (首要核心指标) = {m.get('full_ic', 0.0):.5f}\n"
            f"    - Annualized IR = {m.get('ir', 0.0):.3f}\n"
            f"    - Win Rate = {m.get('win_rate', 0.0)*100:.1f}%\n"
            f"    - Q_Spread = {m.get('q_spread', 0.0):.6f}\n"
            f"    - Full Year MI = {m.get('full_mi', 0.0):.5f}\n"
            f"    - Full Year ANOVA R2 = {m.get('full_anova', 0.0):.5f}\n"
            f"    - 10组 Monotonicity Spearman IC = {m.get('group_spearman', 0.0):.5f}\n"
            f"    - Tail Q05 Adj Return (5%极低分位自适应调整未来收益均值) = {m.get('tail_low_adj_ret', 0.0):.5f}\n"
            f"    - Tail Q95 Adj Return (95%极高分位自适应调整未来收益均值) = {m.get('tail_high_adj_ret', 0.0):.5f}\n"
            f"    - Tail Q05 Global Adj Return (5%极低分位全样本调整未来收益均值) = {m.get('tail_low_glob_adj_ret', 0.0):.5f}\n"
            f"    - Tail Q95 Global Adj Return (95%极高分位全样本调整未来收益均值) = {m.get('tail_high_glob_adj_ret', 0.0):.5f}\n"
        )
        candidates_details.append(det)

    candidates_text = "\n".join(candidates_details)

    # 2. 修改 Prompt 为评分模式
    prompt = f"""你是一个极其严格且极具大局观的量化因子基因筛选专家。
当前我们要对【Layer {layer_num+1} 原料池】进行优胜劣汰。原料池目前的候选因子共有 {len(pool)} 个，我们需要精简到 {target_size} 个。

【待评估的候选因子列表】
{candidates_text}

【你的打分评估标准 (0 - 10分)】
1. 深度科学机理与多样性 (首要标准)：严禁保留公式结构、逻辑高度相似的因子！对于同义反复的公式，只保留 Full_IC 绝对值最高的那一个(7-10分)，其余打极低分(0-3分)淘汰。
2. 绝对的单调性与时序稳健的线性预测力 (核心追求)：本系统的唯一目标是挖掘【方向性强、单调性好且每日表现稳健】的预测因子。
   - 首先，审查 Full Year IC 的绝对值和 MonoIC (10组单调性)。如果 Full Year IC 极其接近于 0，说明它是无方向的“状态变量”，必须打低分(0-4分)坚决淘汰！
   - 【极其关键的稳健性检查】：必须对比 Full Year IC 和 Mean Period IC (指定长度窗口 IC 均值)！那些 Full Year IC 极高，但与 Mean Period IC 符号相反 或 差异巨大的因子，往往是由极少数几天的极端异常值驱动的假信号，实盘极度危险，必须大幅降分。只有 Full Year IC 和 Mean Period IC 符号一致且数值较为接近的因子，才是真正抗周期的稳健 Alpha，应给予最高分！3. 尾部风险调整收益保障：如果两个特征的 IC 差不多，优先给那些【5%或95%服务端自适应调整收益率】绝对值更大的特征打高分，这代表其在极端情况下的实盘做多/做空确定性更强。

请对以上列表中的【每一个】因子进行独立打分（0-10分），并附带一句简短的一句话理由。

请严格按照提供的 JSON 格式输出。
"""
    
    score_map = {}
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        logger.info(f"  -> Scoring Attempt {attempts}/{max_attempts}...")
        try:
            chat_obj = Chat(
                model=LLM_MODEL_NAME,
                messages=[Message(role="user", content=prompt)],
                temperature=0.3, # 低温度保证评分的一致性
                response_format=PoolScoringDecision
            )
            response = chat_obj.create()
            
            if hasattr(Message, "from_chat_response"):
                message_obj = Message.from_chat_response(response)
                raw_response = message_obj.content
            else:
                raw_response = response.choices[0].message.content

            clean_json = raw_response.strip()
            if clean_json.startswith("```"):
                clean_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_json, re.DOTALL).group(1)

            decision = json.loads(clean_json)
            
            # 解析打分结果
            for eval_item in decision.get("evaluations", []):
                score_map[eval_item.get("name")] = eval_item.get("score", 0)
                
            logger.info(f"  √ Successfully parsed scores for {len(score_map)}/{len(pool)} factors.")
            break # 成功解析，跳出重试循环

        except Exception as e:
            logger.error(f"  Scoring attempt {attempts} failed: {e}")

    # 3. 如果 LLM 完全失败或漏掉了因子，制定强大的 Fallback (按 Full_IC 绝对值兜底)
    # 对因子进行排序，第一主键是 LLM 分数(降序)，第二主键是 Full_IC绝对值(降序，作为平局打破者)
    def sort_key(f):
        # 如果 LLM 忘了打分，默认给个负分，让其垫底
        llm_score = score_map.get(f["name"], -1)
        abs_ic = abs(f.get("metrics", {}).get("full_ic", 0.0))
        return (llm_score, abs_ic)

    pool_sorted = sorted(pool, key=sort_key, reverse=True)
    
    survivors = pool_sorted[:target_size]
    
    logger.info(f"🏆 Pool L{layer_num+1} pruned down from {len(pool)} to {len(survivors)} factors.")
    logger.info("Top 3 retained factors (by LLM score & IC):")
    for idx, f in enumerate(survivors[:3]):
        logger.info(f"  #{idx+1}: {f['name']} (Score: {score_map.get(f['name'], 'N/A')}, IC: {f.get('metrics', {}).get('full_ic', 0.0):.4f})")

    return survivors

# ======================================================================
# 6. Theme and Operator Library dynamic prompt compilation
# ======================================================================

def format_quantile_returns_10(metrics: dict) -> str:
    """将底层的10等分收益率直接转为易读的 bps 字符串"""
    if not metrics:
        return "暂无数据"
        
    # 根据你 evaluator 的实际输出字典名获取 (假设你改成了 Group_Returns_10)
    # 如果它包在 pooled_metrics 里面，尝试提取
    ret_10 = metrics.get("Group_Returns_10")
    if not ret_10 and "pooled_metrics" in metrics and len(metrics["pooled_metrics"]) > 0:
        ret_10 = metrics["pooled_metrics"][0].get("Group_Returns_10")
        
    if not ret_10 or len(ret_10) != 10:
        return "无10分组数据"
        
    # 直接全部乘以 10000 变成 bps，保留一位小数并带上正负号
    formatted_vals = [f"{v * 10000:+.1f}" for v in ret_10]
    return "[" + ", ".join(formatted_vals) + "]"

def build_evolutionary_prompt(layer_num: int, state: dict, available_inputs_text: str, nesting_step: int | None = None) -> str:
    """
    Constructs detailed thematic prompts for each evolutionary layer, 
    injecting global rules, layer-specific goals, and the full operator library.
    """
    # 1. 动态加载外挂库和经验教训
    operator_lib_text = ""
    try:
        lib_md_path = project_root / "libraries" / "operator_library.md"
        if lib_md_path.exists():
            with open(lib_md_path, "r", encoding="utf-8") as f_md:
                operator_lib_text = f_md.read()
    except Exception:
        pass
    kb_lessons_text = "\n".join(f"- {l}" for l in state.get("kb_lessons", []))
    compound_ops_desc = load_compound_operators_desc()
    
    # 获取外部衍生特征
    extra_fields = load_extra_base_fields()
    extra_fields_text = ""
    if extra_fields:
        extra_fields_text = "\n【🎁 外部框架注入的高级衍生特征】\n" + "\n".join(f"- {k}: {info.get('description')}" for k, info in extra_fields.items())

    # 2. 【核心铁律】：宇宙A与宇宙B的基础数据规则
    base_rules_text = """
# Role: 高频量价因子挖掘专家 & 表达式工程师 (Alpha Factor Engineer)
你现在是一个顶尖的量化对冲基金研究员。你的任务是根据不同 Layer 的目标，发挥极致的创造力，设计出符合我方计算引擎语法的【因子表达式】。

## 📚 基础数据字段池 (Base Fields)
我们的底层数据频率为 **1 分钟 (1-Minute Bar)**。时间窗口 `window` 单位均为“分钟”。

**【绝对铁律：两套平行宇宙不可物理混用！】**
1. **宇宙 A：微观盘口宇宙 (纯原始数据)**
   - 适用于短周期微观特征。
   - `close`, `open`, `high`, `low` (交易所连续推送的快照统计出的连续买卖中间价)
   - `close_trd`, `open_trd`, `high_trd`, `low_trd` (里面通过有真实成交的tick统计出的真实成交价，冷门时段可能为 NaN，需结合逻辑意义做处理)
   - `volume` (一分钟新增成交量), `vwap`（一分钟内的）
   - `tick_count`, `tick_count_trd`：交易所推送tick数与真实有成交的tick数。
   - `open_interest`：该分钟期末的持仓量累计值
   - `bid1_qty`, `ask1_qty`：分钟期末的最优买一、卖一挂单量。
   - `bid1`, `ask1`：分钟期末的最优买一、卖一价格。
    可把数字换成1-5来获得买卖方更深档位的价格或挂单量
"""

    # 3. 【客观现象诊断指南】：只描述关注的模式，不提供具体操作指导
    quant_vision_guide = """
## 🧠 专家级特征诊断与“阅图”指南 (10等分未来收益图分析)
在后续层级中，你必须通过观察前置原料的【10等分未来收益图 (单位: bps)】和核心指标来判断当前特征的状态。我们主要关注以下几种现象：
1. 📈 **整体单调性 (Monotonicity)**: 观察从左端(极低取值组)到右端(极高取值组)是否呈现出整体的递增或递减趋势。理想的特征应在两端展现出显著的收益差。
2. 🧲 **U型/倒U型 (U-Shape) 等异形分布**: 观察两端收益是否同号，且相对中间组别较高或较低 (如 [+3, +1, ..., +1, +3])。这是一种经常被观察到的非线性现象。
3. 📉 **尾部震荡与整体趋势**: 越靠近两端的极值组，由于样本分布等原因，出现一定程度的收益上下跳动(锯齿)是正常的。你需要区分“整体有单调趋势但尾部有正常浮动”与“毫无趋势的纯随机噪声”。
4. 🐜 **收益振幅 (Amplitude)**: 观察两端极值组之间的收益率差值绝对大小。如果头尾组的收益率相差极微，说明该特征当前形态下的预测力/波动非常微弱。
"""

    p_text = ""
    # ==================== Layer 1: 基础原料层 ====================
    if layer_num == 1:
        import random
        sampled_features_text = ""
        try:
            combined_pool = {}
            standard_feats = load_foundation_features()
            if standard_feats:
                for k, info in standard_feats.items():
                    combined_pool[k] = {"formula": info.get("formula", ""), "description": info.get("description", "标准衍生特征")}
            
            # --- 关键修改：直接读取你字典里的原汁原味内容 ---
            extra_feats = load_extra_base_fields()
            if extra_feats:
                for k, info in extra_feats.items():
                    # 不要覆盖公式，直接用它 JSON 里的 formula (即 BuyRatio)
                    combined_pool[k] = {"formula": info.get("formula", k), "description": info.get("description", "外部注入特征")}
            
            if combined_pool:
                keys_all = list(combined_pool.keys())
                keys_sampled = random.sample(keys_all, min(10, len(keys_all)))
                sampled_features_text = "\n".join([f"- {k}: {combined_pool[k]['formula']} - {combined_pool[k]['description']}" for k in keys_sampled])
        except Exception:
            pass

        p_text = f"""
## 🎯 当前任务：【Layer 1: 基础原料层 (Raw Ingredients)】
目标：这一层目前没有任何评测指标，你的核心任务是**基于深度的微观逻辑，创造性地构造有意义的基础原料**。
很多时候，单一字段的信息是单薄的。你需要利用基础字段，或者从以下【随机精选的特征库】中挑选特征，将两个或多个特征结合起来，构造出有意义的逻辑关系（例如某些有意义的比值、差值或复合状态）。
【💎 两种极具价值的构造路径 (强烈建议兼顾)】：
1. **多变量逻辑交互**：将两个或多个基础字段结合，构造出有意义的比值、差值或物理状态（如 `volume / (open_interest + 1e-8)`）。
2. **单变量深度变换**：不要只做简单的加减乘除！你可以对单一基础字段直接施加一元算子或基础时序算子（如 `TsDelta(vwap, 5)` 捕捉价格加速度，`TsPct(volume, 5)` 捕捉量能爆发，或 `DistortLog(tick_count)` 压制极端长尾），这往往能产生极其优秀的独立初始正交特征！

【🎁 可直接调用的随机精选特征 (已提供机理，可直接在公式中写名称)】
{sampled_features_text if sampled_features_text else "- 暂无可用随机特征，请使用基础字段。"}

【核心要求】：
1. 充分发挥创造力，构建逻辑严谨且有意义的组合，为后续层级提供坚实的骨架。
强烈建议你不要太依赖随机精选特征，保持一半的输出是你利用原始数字（基础字段）做的有意义的构造/组合。
2. 必须命名为结合机理的 snake_case_name。
3. 严格输出 JSON，格式必须符合 `L1Batch` schema：
{{ "candidates": [ {{ "name": "...", "idea_rationale": "...", "formula": "..." }}, ... ] }}
"""

    # ==================== Layer 2: 截面/相对化层 ====================
    elif layer_num == 2:
        p_text = f"""
## 🎯 当前任务：【Layer 2: 相对化与截面可比层 (Relativization)】
目标：将 Layer 1 的特征改造为在不同品种间具备可比性的相对化指标。
不同品种的原始价格规模、成交量规模、以及盘口流动性（如挂单量）在微观结构上长期存在巨大的天然差异。
你需要结合特征原本的语义和 rationale，思考如何利用该特征本身的历史数据或其他锚点，做一些有意义的计算，使其剥离规模差异，成为更纯粹的相对强度特征。

【可用前置原料】:
{available_inputs_text}

【输出要求】: 
1. 强制覆盖：你必须为上述列表中提供的【每一个】前置原料，都至少设计 1 个相对化变异！（即：确保列表中每个原料的 ID 都至少作为一次 `source_factor` 出现，绝不能遗漏任何一个基因分支）(如果确实已经是相对化的比如比率等形式，也可以不用做额外操作)。
2. 总数量：构思 15 到 20 个变异。
3. 严格输出 JSON，格式符合 `L2Batch` schema。
"""

    # ==================== Layer 3: 分布调整层 ====================
    elif layer_num == 3:
        p_text = f"""
## 🎯 当前任务：【Layer 3: 分布变换与异形修复层 (Distribution Transformation)】
目标：从语义和图形双重角度，判断并调整特征的分布，使其更有可能与未来收益率呈现相对单调的趋势。
在这里，你需要特别关注【阅图指南】中提到的 **U型/倒U型分布** 以及 **振幅不够大** 的问题。
你需要：
1. 结合该特征原本的公式以及 rationale。
2. 观察其 10等分未来收益图 (bps)，深度分析导致这种奇怪形状（如U型）或振幅微弱的潜在原因。
3. 发挥创造力，通过对特征做有逻辑、有意义的变换，试图修正这一问题，促使其转化为具备有效方向性的特征。

【可用前置原料】:
{available_inputs_text}

【输出要求】: 结合原理解释和图形表现进行改造。严格输出 JSON，格式符合 `L3Batch` schema。
1. 强制覆盖：你必须为上述列表中提供的【每一个】前置原料，都至少设计 1 个操作！绝不能遗漏任何一个基因分支。
2. 总数量：构思 15 到 20 个操作。
3. 严格输出 JSON，格式符合 `L3Batch` schema。
"""

    # ==================== Layer 4: 复杂时序/复合交互层 ====================
    elif layer_num == 4:
        step_info = f"（这是 L4 的第 {nesting_step}/3 轮递归进化）" if nesting_step else ""
        p_text = f"""
## 🎯 当前任务：【Layer 4: 高级时序与复合交互层 (Temporal & Interaction)】 {step_info}
目标：这是系统中最关键的层级。你需要紧密结合【10等分未来收益图 (bps)】，构造创造性的时序算子或复合时序操作！
【💎 跨界组合与二元算子强力提醒】：
当你在尝试施加二元算子（如 `TsCorr(x, y, window)`, `TsCov`, `TsFacNeutralRes` 等）时，**绝对不要局限于单一的原料特征**！
你的第二个变量 `y` 也可以跨界调用【基础数据字段池】（如 `volume`, `open_interest`, `vwap`, `close_adj` 等），或者利用我们在前面注入的基础特征。

【深度归因与修正要求】：
在你的 `idea_rationale` 中，你必须针对以下现象进行结合公式实质的【详细归因分析】，并给出针对性的操作：
1. **单调性趋势**：整体趋势是否明显？特别注意，我们甚至可能不怎么关心中间部分的单调性，而是**极其关注头尾部（第1-2组与第9-10组）的单调性与方向**。如果头尾呈现非单调，原因是什么？
2. **振幅大小**：头尾部组之间的 bps 差距是否足够大？如果不够，这说明信号衰减过快或噪音覆盖了信号，该用什么复杂的复合时序结构去提取核心信息？

【可用前置原料】:
{available_inputs_text}

【输出要求】: 必须包含对单调性/振幅的公式级归因，并大胆使用高级时序/复合算子。严格输出 JSON，格式符合 `L4Batch` schema。
1. 强制覆盖：你必须为上述列表中提供的【每一个】前置原料，都至少进行 1 次复合时序操作或判定！绝不能遗漏任何一个基因分支。
2. 强制归因：idea_rationale 必须包含对单调性/振幅的公式级诊断。
3. 使用多种基础时序算子的复合复杂形式。总构思 15 到 20 个。
4. 严格输出 JSON，格式符合 `L4Batch` schema。
"""

    # ==================== Layer 5: 最终降噪与升华层 ====================
    elif layer_num == 5:
        p_text = f"""
## 🎯 当前任务：【Layer 5: 最终升华、过滤与窗口参数网格搜索层 (Final Polish, Filtering & Window Grid Search)】

本层有两个并行的核心任务，都必须完成：

### 任务 A：平滑/升华操作（必做）
对每个前置原料施加最终的平滑、过滤或增强操作，进一步提升单调性和稳定性。
如果当前特征已足够完美，可以设 formula 为 source_factor 本身（NoOp 直通，不做额外处理）。

### 任务 B：窗口参数变体扫描（必做！每个因子至少 5 组，最多 8 组）
这是本层极其关键的任务。你需要**结合该特征的逻辑机理和经济含义**，
为其时序算子中的回看窗口参数，设计 **5-8 组具有显著差异的替代窗口组合**，
并输出对应的完整变体公式。

【窗口变体设计原则】：
1. **必须有显著差异**：不要只改 ±1 分钟！如果原窗口是 10，你应该探索 2, 5, 10, 40, 60, 120, 240 等数量级不同的值。窗口单位均为【分钟】。
2. **必须结合逻辑**：在 window_rationale 中解释为什么这个窗口值值得测试。你需要基于不同回看周期的经济含义来思考：
   - 1-5 分钟级别：捕捉微观结构中的瞬时冲击、短期反转、做市商存货调整等极短周期效应。
   - 10-30 分钟级别：捕捉日内中周期趋势、盘中资金流向惯性、以及典型的交易时段节奏。
   - 40-90 分钟级别：跨越多个交易子时段，捕捉较稳定的日内主趋势和波动率聚集效应。
   - 120-240 分钟级别：接近半日或全天，测试信号在更长持仓周期下的衰减情况和方向稳健性。
   - 480-1440 分钟级别：跨日/隔夜级别，测试因子是否具备跨日持仓的预测力（需警惕隔夜跳空噪声）。
3. **变体公式必须是完整可执行的**：不能只给窗口数字，要给出完整的可执行公式。
4. **覆盖极端值**：至少包含 1 个非常短的窗口（≤原窗口的 1/4）和 1 个非常长的窗口（≥原窗口的 4 倍），以充分探测因子在不同时间尺度上的表现。

【示例——仅展示窗口选择的思考逻辑，绝不限制你使用的算子类型】：
假设一个因子的核心时序窗口为 15 分钟，你可能会考虑以下窗口变体（具体用什么算子、是否嵌套、是否组合其他字段，完全由你根据因子逻辑自由发挥）：
- window_label: "ultra_fast_2min", window_rationale: "2分钟级别探测微观噪声占比和瞬时信号衰减速度"
- window_label: "fast_5min", window_rationale: "5分钟级别捕捉盘中短期反转和做市商行为模式"
- window_label: "medium_30min", window_rationale: "30分钟跨越多个交易子时段，测试中周期趋势稳定性"
- window_label: "slow_90min", window_rationale: "90分钟接近半日，测试信号在更长持仓期是否持续有效"
- window_label: "ultra_slow_240min", window_rationale: "240分钟接近全天，测试跨时段/隔夜的预测残留强度"
- window_label: "no_lookback_raw", window_rationale: "完全移除时序回看，测试瞬时截面的纯净预测力"

【可用前置原料】:
{available_inputs_text}

【输出要求】:
1. 强制覆盖：为列表中【每一个】前置原料至少设计 1 个平滑/升华操作（可为 NoOp）。
2. 强制窗口变体：为每一个因子设计 **5-8 组** 有显著差异的窗口变体。
3. 总候选数上限 18 个（注意每个候选带有 5-8 个变体，token 量较大）。
4. 严格输出 JSON，格式符合 `L5Batch` schema。
"""


    # 4. 拼装神级混合 Prompt
    prompt_final = (
        base_rules_text
        + extra_fields_text
        + "\n" + quant_vision_guide
        + "\n" + p_text
        + "\n\n========================================"
        + "\n【⚠️ 历史严重教训 (绝不可犯)】\n" + kb_lessons_text
        + "\n\n========================================"
        + "\n【🚀 复合算子库 (可直接调用)】\n" + compound_ops_desc
        + "\n\n========================================"
        + "\n【⚙️ 底层原语算子库 (Operator Library)】\n" + operator_lib_text
    )
    
    return prompt_final

# ======================================================
def select_pool_subset(pool: List[dict], ratio: float = 0.5, max_count: int = 25) -> List[dict]:
    """
    Selects a configurable subset of the pool using a hybrid of:
    1. Top-performing factors sorted by absolute full_ic.
    2. Random diversity sampling from the remaining factors.
    """
    if not pool:
        return []

    total_needed = min(max_count, int(len(pool) * ratio))
    if total_needed <= 0:
        total_needed = min(len(pool), 5)

    # Sort pool by absolute full_ic
    sorted_pool = sorted(pool, key=lambda x: abs(x.get("metrics", {}).get("full_ic", 0.0)), reverse=True)

    top_count = total_needed // 2
    random_count = total_needed - top_count

    selected = sorted_pool[:top_count]
    remaining = sorted_pool[top_count:]

    import random
    if remaining and random_count > 0:
        selected.extend(random.sample(remaining, min(random_count, len(remaining))))

    return selected

# ======================================================
# 🌟 绝对保险箱：Alpha 名人堂永久存储
# ======================================================
def save_to_hall_of_fame(factor: dict, layer_num: int):
    """
    无论处于哪一层，只要特征通过了严苛的评估门槛 (passed=True)，
    立即将其永久物理备份至 alpha_hall_of_fame.json，防止在后续进化池中被 LLM 误杀淘汰。
    """
    fame_file = project_root / f"alpha_hall_of_fame_{FOCUS_UNIVERSE.lower()}.json"
    try:
        if fame_file.exists():
            with open(fame_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        # 防止重复写入同一个公式
        existing_formulas = set(item.get("formula") for item in data)
        if factor["formula"] in existing_formulas:
            return 
            
        m = factor.get("metrics", {})
        fame_record = {
            "name": factor["name"],
            "layer_found": layer_num,
            "formula": factor["formula"],
            "rationale": factor.get("rationale", ""),
            "ir": m.get("ir", 0.0),
            "full_ic": m.get("full_ic", 0.0),
            "mean_ic": m.get("ic", 0.0),
            "discovery_time": datetime.now().isoformat()
        }
        data.append(fame_record)
        
        # 按照 IR 的绝对值降序排列，打造排行榜
        data = sorted(data, key=lambda x: abs(x.get("ir", 0.0)), reverse=True)
        
        with open(fame_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"🌟🌟🌟 [名人堂] 发现神级特征 '{factor['name']}' (来自 Layer {layer_num})！已永久备份至 Alpha 保险箱 (IR: {m.get('ir', 0.0):.3f})")
    except Exception as e:
        logger.error(f"Failed to save factor to Hall of Fame: {e}")

# ======================================================

def run_layer_generation(state: dict, layer_num: int):
    """Executes a single layer generation and evaluation round with persistent pools."""
    logger.info("="*75)
    focus_universe = FOCUS_UNIVERSE
    logger.info(f"🧬  [Evolution System] Entering Generation Layer {layer_num} / 5  🧬")
    logger.info("="*75)

    # 1. Gather survivors from previous pools to act as input formulas
    previous_formulas = {}

    # Map histories of prior factors
    for hist_item in state.get("history", []):
        if "formula" in hist_item:
            previous_formulas[hist_item["name"]] = hist_item["formula"]

    # Also load all previous pools into previous_formulas to guarantee lookups succeed
    for pool_key, pool_list in state.get("pools", {}).items():
        for item in pool_list:
            if "formula" in item:
                previous_formulas[item["name"]] = item["formula"]

    # Gather available inputs for prompting
    available_inputs_lines = []
    if layer_num > 1:
        pool_key = f"L{layer_num}_pool"
        full_pool = state["pools"].get(pool_key, [])
        latest_names = set(state.get("latest_outputs", []))
        
        import random
        # 1. 强制提取：找出上一层刚刚生成的“新生代”因子 (保证流水线推进)
        mandatory_items = [item for item in full_pool if item["name"] in latest_names]
        
        # 如果新生代太多，随机截取 10 个
        if len(mandatory_items) > 10:
            mandatory_items = random.sample(mandatory_items, 10)
            
        # 2. 多样性补充：如果新生代不足 10 个，从历史基因池中随机抽取补齐
        target_size = 10
        needed = target_size - len(mandatory_items)
        random_items = []
        if needed > 0:
            remaining_pool = [item for item in full_pool if item["name"] not in latest_names]
            random_items = random.sample(remaining_pool, min(needed, len(remaining_pool)))
            
        selected_subset = mandatory_items + random_items
        logger.info(f"Selected {len(selected_subset)} factors for Layer {layer_num} prompt (Mandatory Latest: {len(mandatory_items)}, Historical Random: {len(random_items)}).")

        for item in selected_subset:
            m = item.get("metrics", {}) or {}
            q_returns_str = format_quantile_returns_10(m)
            available_inputs_lines.append(
                f"- ID: {item['name']}\n"
                f"  机理: {item['rationale']}\n"
                f"  当前公式: {item['formula']}\n"
                f"  指标: Mean Period IC={m.get('ic',0.0):.4f}, Full IC={m.get('full_ic',0.0):.4f}, IR={m.get('ir',0.0):.3f}, MI={m.get('full_mi',0.0):.4f}, ANOVA={m.get('full_anova',0.0):.4f}, MonoIC={m.get('group_spearman',0.0):.4f}"
                f"  10等分未来收益均值图(左特征取值极低->右特征取值极高): {q_returns_str} bps"
            )
        available_inputs_text = "\n".join(available_inputs_lines) if available_inputs_lines else "- 暂无可用前置原料（这不应该发生，除非前置原料池为空）"
    else:
        available_inputs_text = "- 暂无可用前置原料（这是首层输入）"

    # 2. Select appropriate batch pydantic schema for this layer
    response_schema = None
    if layer_num == 1:
        response_schema = L1Batch
    elif layer_num == 2:
        response_schema = L2Batch
    elif layer_num == 3:
        response_schema = L3Batch
    elif layer_num == 4:
        response_schema = L4Batch
    elif layer_num == 5:
        response_schema = L5Batch

    # 3. Special Execution Routing for Layer 4 Recursive Nesting vs Standard Layers
    if layer_num == 4:
        logger.info("======================================================================")
        logger.info("⚡ 启动高级时序复合层（Layer 4）深度 3 轮递归进化叠级 ⚡")
        logger.info("======================================================================")

        # We start with raw Layer 3 materials
        current_l4_materials = selected_subset.copy()
        final_l4_outputs = []

        for nesting_step in range(1, 4):
            logger.info(f"⏳ [Layer 4 Nesting Round {nesting_step}/3] Preparing prompt inputs...")

            # Formulate inputs text
            step_inputs_lines = []
            for item in current_l4_materials:
                m = item.get("metrics", {}) or {}
                q_returns_str = format_quantile_returns_10(m)
                step_inputs_lines.append(
                    f"- ID: {item['name']}\n"
                    f"  机理: {item['rationale']}\n"
                    f"  当前公式: {item['formula']}\n"
                    f"  指标: Mean Period IC={m.get('ic',0.0):.4f}, Full IC={m.get('full_ic',0.0):.4f}, IR={m.get('ir',0.0):.3f}"
                    f"  10等分未来收益均值图(左特征取值极低->右特征取值极高): {q_returns_str} bps"
                )
            step_inputs_text = "\n".join(step_inputs_lines) if step_inputs_lines else "- 暂无可用前置原料"

            # Format custom nesting step prompt
            prompt_step = build_evolutionary_prompt(4, state, step_inputs_text, nesting_step=nesting_step)

            # Invoke LLM via  chat
            logger.info(f"Calling {LLM_MODEL_NAME} for Nesting Step {nesting_step}/3...")
            try:
                chat_obj = Chat(
                    model=LLM_MODEL_NAME,
                    messages=[Message(role="user", content=prompt_step)],
                    temperature=0.7,
                    response_format=L4Batch
                )
                response = chat_obj.create()
                if hasattr(Message, "from_chat_response"):
                    message_obj = Message.from_chat_response(response)
                    raw_resp = message_obj.content
                else:
                    raw_resp = response.choices[0].message.content
            except Exception as e_api:
                logger.error(f"LLM API failure during Nesting Step {nesting_step}: {e_api}")
                break

            # Extract and parse response JSON
            clean_j = raw_resp.strip()
            if clean_j.startswith("```"):
                clean_j = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_j, re.DOTALL).group(1)
            try:
                b_data = json.loads(clean_j)
            except Exception as e_json:
                logger.error(f"JSON Parsing failed during Nesting Step {nesting_step}: {e_json}")
                break

            raw_cand = b_data.get("candidates", [])
            logger.info(f"Parsed {len(raw_cand)} newly evolved nesting actions from {LLM_MODEL_NAME}.")

            # Update previous_formulas so the compiler can resolve parents correctly
            # We must load previous pool items as well as any items generated in the current round
            for item in current_l4_materials:
                if "formula" in item:
                    previous_formulas[item["name"]] = item["formula"]

            step_compiled_items = []
            noop_count = 0
            for cand_r in raw_cand:
                try:
                    action_val = L4Action(**cand_r)
                    name = action_val.name
                    rationale = action_val.idea_rationale

                    if action_val.pattern_op == "NoOp":
                        noop_count += 1

                    # Compile formula with recursive lookups
                    compiled_f = compile_evolution_formula(action_val.dict(), previous_formulas)

                    step_compiled_items.append({
                        "name": name,
                        "layer": 4,
                        "rationale": rationale,
                        "formula": compiled_f,
                        "action": action_val.dict(),
                        "metrics": None
                    })
                except Exception as ve_r:
                    logger.warning(f"⚠️ Nesting compile failed for action: {ve_r}. Skipping.")

            if not step_compiled_items:
                logger.error(f"❌ No valid nesting factors compiled. Stopping Nesting loop early.")
                break

            logger.info(f"Successfully compiled {len(step_compiled_items)} nesting candidates for Step {nesting_step}/3.")

            # Evaluate this nesting batch on GPU
            logger.info(f"⚡ Launching GPU Graph Evaluation for Step {nesting_step}/3...")
            evaluated_step_items = evaluate_evolutionary_pool(step_compiled_items, previous_formulas, focus_universe=focus_universe)
            
            # 💥 拦截：检查是否有过门槛的神级因子，立即存入名人堂！
            for item in evaluated_step_items:
                if item.get("metrics", {}).get("passed", False):
                    save_to_hall_of_fame(item, layer_num=4)

            # Filter non-crashing candidates
            evaluated_step_items = [
                f for f in evaluated_step_items
                if f.get("metrics") is not None and "error_msg" not in f["metrics"]
            ]
            logger.info(f"Retained {len(evaluated_step_items)} valid intermediate nested candidates.")

            # Update pool state
            current_l4_materials = evaluated_step_items
            final_l4_outputs = evaluated_step_items

            # Update previous_formulas for the next nesting step
            for item in current_l4_materials:
                if "formula" in item:
                    previous_formulas[item["name"]] = item["formula"]

            # Early Exit Detection: If ALL compiled actions were "NoOp", or the material list is empty, we exit early!
            if noop_count == len(step_compiled_items) and noop_count > 0:
                logger.info("🚪 [Early Exit] All candidate factors chose 'NoOp' as their temporal operator. Terminating Nesting loop early as factors are logically complete!")
                break
            if not current_l4_materials:
                logger.info("🚪 [Early Exit] Nesting material pool is empty. Terminating Nesting loop early.")
                break

        # Finally, the compiled candidates are populated into the next-layer material pool L5_pool
        evaluated_items = final_l4_outputs

    else:
        # Standard Single-Pass Layer Generation (for Layers 1, 2, 3, 5)
        # 3. Format dynamic theme prompt
        prompt = build_evolutionary_prompt(layer_num, state, available_inputs_text)

        # 4. Invoke LLM via  chat
        logger.info(f"Calling {LLM_MODEL_NAME} to evolve candidate factors for Layer {layer_num}...")
        try:
            chat_obj = Chat(
                model=LLM_MODEL_NAME,
                messages=[Message(role="user", content=prompt)],
                temperature=0.7,
                response_format=response_schema
            )
            response = chat_obj.create()
            if hasattr(Message, "from_chat_response"):
                message_obj = Message.from_chat_response(response)
                raw_response = message_obj.content
            else:
                raw_response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API failure: {e}")
            return

        # Extract and parse response
        clean_json = raw_response.strip()
        if clean_json.startswith("```"):
            clean_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_json, re.DOTALL).group(1)

        try:
            batch_data = json.loads(clean_json)
        except Exception as e:
            logger.error(f"JSON Parsing failed: {e}. Raw response: {raw_response[:500]}")
            return

        # Extract candidates list
        raw_candidates = batch_data.get("candidates", [])
        logger.info(f"Parsed {len(raw_candidates)} newly evolved actions from {LLM_MODEL_NAME}.")

        # 5. Build and validate new layer factors
        new_pool_items = []
        for cand_raw in raw_candidates:
            try:
                # Re-verify through corresponding Pydantic schema
                if layer_num == 1:
                    action_validated = L1Candidate(**cand_raw)
                elif layer_num == 2:
                    action_validated = L2Action(**cand_raw)
                elif layer_num == 3:
                    action_validated = L3Action(**cand_raw)
                elif layer_num == 5:
                    action_validated = L5Action(**cand_raw)

                name = action_validated.name
                rationale = action_validated.idea_rationale

                # Compile immediately to ensure syntactic validity
                compiled_f = compile_evolution_formula(action_validated.dict(), previous_formulas)

                new_pool_items.append({
                    "name": name,
                    "layer": layer_num,
                    "rationale": rationale,
                    "formula": compiled_f,
                    "action": action_validated.dict(),
                    "metrics": None
                })
            except Exception as ve:
                logger.warning(f"⚠️ Validation or local compile failed for action: {ve}. Skipping candidate.")

        if not new_pool_items:
            logger.error("❌ No valid factors survived the local validation/compilation pass. Aborting layer.")
            return

        logger.info(f"Successfully compiled {len(new_pool_items)} new candidates for Layer {layer_num}.")

        # 6. Evaluate all newly built candidates
        evaluated_items = evaluate_evolutionary_pool(new_pool_items, previous_formulas, focus_universe=focus_universe)

        # 💥 拦截：检查是否有过门槛的神级因子，立即存入名人堂！
        for item in evaluated_items:
            if item.get("metrics", {}).get("passed", False):
                save_to_hall_of_fame(item, layer_num=layer_num)


        # Filter candidates based on whether they compiled and evaluated successfully (no crash/runtime errors)
        if layer_num in [1, 2, 3]:
            # Intermediate layers: keep any factor that successfully evaluated without crashing (has valid metrics and no error_msg)
            evaluated_items = [
                f for f in evaluated_items
                if f.get("metrics") is not None and "error_msg" not in f["metrics"]
            ]
            logger.info(f"Retained {len(evaluated_items)} intermediate candidates (regardless of performance threshold) for material pool insertion.")
        elif layer_num == 5:
            # 先按编译成功/失败过滤一次（公式都跑不起来的直接淘汰）
            l5_valid = [
                f for f in evaluated_items
                if f.get("metrics") is not None and "error_msg" not in f["metrics"]
            ]
            logger.info(f"Layer 5 base: {len(l5_valid)}/{len(evaluated_items)} successfully evaluated (before window sweep).")

            if l5_valid:
                # ---- 收集所有窗口变体并批量编译/评测 ----
                all_variants_to_test = []

                for f in l5_valid:
                    action = f.get("action", {})
                    variants = action.get("window_variants", [])

                    if not variants:
                        logger.info(f"  ⚠ {f['name']}: LLM produced 0 window variants. Skipping sweep for this factor.")
                        continue

                    logger.info(f"  🔍 {f['name']}: LLM suggested {len(variants)} window variants -> {[v['window_label'] for v in variants]}")

                    for v in variants:
                        variant_formula_raw = v["formula"]
                        variant_label = v["window_label"]

                        try:
                            compiled_var = compile_evolution_formula(
                                {"formula": variant_formula_raw, "source_factor": action.get("source_factor")},
                                previous_formulas,
                            )
                            compiled_var = macro_expand_formula(compiled_var)
                        except Exception as ce:
                            logger.warning(f"    ✗ Variant '{variant_label}' compilation failed: {ce}")
                            continue

                        variant_name = f"{f['name']}__var_{variant_label}"
                        all_variants_to_test.append({
                            "name": variant_name,
                            "formula": compiled_var,
                            "parent_name": f["name"],
                            "parent_rationale": f.get("rationale", ""),
                            "parent_formula": f["formula"],
                            "parent_passed": f.get("metrics", {}).get("passed", False),
                            "parent_full_ic": f.get("metrics", {}).get("full_ic", 0.0),
                            "variant_label": variant_label,
                            "window_rationale": v.get("window_rationale", ""),
                        })

                if all_variants_to_test:
                    logger.info(f"🌊 [L5 Window Sweep] Evaluating {len(all_variants_to_test)} window variants across {len(l5_valid)} factors...")

                    var_formulas = {v["name"]: v["formula"] for v in all_variants_to_test}
                    try:
                        var_batch_metrics = evaluate_factor_batch(
                            var_formulas,
                            focus_universe=focus_universe,
                            holding_horizon=HOLDING_HORIZON,
                        )
                    except Exception as vex:
                        logger.error(f"Window variant batch evaluation failed: {vex}")
                        var_batch_metrics = {}

                    # ---- 组装结果 ----
                    base_passed = [f for f in l5_valid if f.get("metrics", {}).get("passed", False)]

                    final_l5_variants = []
                    rescued_count = 0    # base 没通过但变体通过了
                    improved_count = 0   # base 通过了且变体更优
                    no_improve_passed = 0  # base 通过了，变体也通过了但没更优

                    for v in all_variants_to_test:
                        vm = var_batch_metrics.get(v["name"], {})
                        variant_passed = vm.get("passed", False)
                        v_abs_ic = abs(vm.get("full_ic", 0.0))
                        parent_abs_ic = abs(v["parent_full_ic"])

                        if variant_passed:
                            if not v["parent_passed"]:
                                rescued_count += 1
                                logger.info(
                                    f"  🎉 RESCUE: {v['parent_name']} :: {v['variant_label']}: "
                                    f"base FAILED (|Full IC|={parent_abs_ic:.6f}) → variant PASSES "
                                    f"(|Full IC|={v_abs_ic:.6f})  |  IR={vm.get('ir', 0.0):.3f}"
                                )
                            elif v_abs_ic > parent_abs_ic + 1e-8:
                                improved_count += 1
                                logger.info(
                                    f"  🏆 {v['parent_name']} :: {v['variant_label']}: "
                                    f"|Full IC| {parent_abs_ic:.6f} → {v_abs_ic:.6f} "
                                    f"(Δ=+{v_abs_ic - parent_abs_ic:.6f})  |  IR={vm.get('ir', 0.0):.3f}"
                                )
                            else:
                                no_improve_passed += 1
                                logger.info(
                                    f"  ✅ {v['parent_name']} :: {v['variant_label']}: "
                                    f"PASSES (|Full IC|={v_abs_ic:.6f}, base={parent_abs_ic:.6f})  |  IR={vm.get('ir', 0.0):.3f}"
                                )

                            final_l5_variants.append({
                                "name": v["name"],
                                "layer": 5,
                                "rationale": (
                                    f"[Window Variant: {v['variant_label']}] {v['window_rationale']}. "
                                    f"Parent: {v['parent_name']} (rationale: {v['parent_rationale']}). "
                                    f"Parent formula: {v['parent_formula']}. "
                                    f"Variant formula: {v['formula']}"
                                ),
                                "formula": v["formula"],
                                "action": {"name": v["name"], "formula": v["formula"]},
                                "metrics": vm,
                                "window_variant": True,
                                "parent_name": v["parent_name"],
                                "variant_label": v["variant_label"],
                                "rescued": not v["parent_passed"],
                            })
                            save_to_hall_of_fame(final_l5_variants[-1], layer_num=5)

                    logger.info(
                        f"🌊 [L5 Window Sweep] Complete: "
                        f"{len(final_l5_variants)} window variants passed threshold "
                        f"(🚑 {rescued_count} rescued from failed base, "
                        f"🏆 {improved_count} improved over already-passing base, "
                        f"✅ {no_improve_passed} passed but no improvement). "
                        f"All retained — no forced pruning."
                    )

                    evaluated_items = base_passed + final_l5_variants
                    logger.info(
                        f"Layer 5 final: {len(evaluated_items)} total "
                        f"({len(base_passed)} base-passed + "
                        f"{len(final_l5_variants)} window variants). "
                        f"All passed to active_pool — you decide what to keep."
                    )
                else:
                    logger.warning("No valid window variants were compiled. Falling back to base-only filtering.")
                    evaluated_items = [f for f in l5_valid if f.get("metrics", {}).get("passed", False)]
            else:
                evaluated_items = []

            logger.info(f"Retained {len(evaluated_items)} final Layer 5 candidates (after window sweep & threshold).")


        # 7. Name Uniqueness & Next-Layer Pool Insertion / Boundary Control
    if layer_num in [1, 2, 3, 4]:
        next_pool_key = f"L{layer_num+1}_pool"
        next_pool = state["pools"].setdefault(next_pool_key, [])
        existing_names = {item["name"] for item in next_pool}

        logger.info(f"Inserting {len(evaluated_items)} newly evaluated candidates into {next_pool_key}...")
        current_layer_new_names = [] # <=== 新增：记录当前层的新生代
        
        for item in evaluated_items:
            orig_name = item["name"]
            unique_name = make_name_unique(orig_name, existing_names)
            if unique_name != orig_name:
                logger.info(f"  -> Name collision resolved: '{orig_name}' -> '{unique_name}'")
                item["name"] = unique_name
                if "action" in item and "name" in item["action"]:
                    item["action"]["name"] = unique_name

            existing_names.add(unique_name)
            next_pool.append(item)
            current_layer_new_names.append(unique_name) # <=== 新增
            
        # 将新生代名单存入 state，供下一层强制提取延续血脉
        state["latest_outputs"] = current_layer_new_names

        # If pool size exceeds 50, trigger LLM pruning loop to trim to exactly 25
        if len(next_pool) > POOL_MAX_SIZE:
            logger.info(f"🔥 {next_pool_key} size ({len(next_pool)}) exceeds limit ({POOL_MAX_SIZE}). Triggering LLM pruning...")
            pruned = prune_pool_via_llm_loop(next_pool, layer_num=layer_num, target_size=int(POOL_MAX_SIZE/2), max_attempts=10)
            state["pools"][next_pool_key] = pruned

    elif layer_num == 5:
        # Layer 5 is the final smoothing & noise filtering layer.
        # Save survivors to active_pool (round-specific propagation pool) for final reports.
        logger.info(f"Layer 5 complete! Retaining {len(evaluated_items)} final polished factors in active_pool.")
        state["active_pool"].extend(evaluated_items)

    save_evolution_state(state)


# ======================================================================
# 8. Main Evolutionary Pipeline Driver
# ======================================================================

def start_evolutionary_pipeline():
    """Starts the end-to-end multi-layer genetic-GP factor miner."""
    logger.info("="*75)
    logger.info("🌟🎯 启动全新 [基于分层基因进化的期货量化因子挖掘机] (GP-Evolution Miner) 🎯🌟")
    logger.info("="*75)

    # Ensure context is pre-loaded resident in VRAM
    state = load_evolution_state()
    focus_universe = FOCUS_UNIVERSE
    try:
        get_or_create_global_context(focus_universe)
    except Exception as e:
        logger.error(f"Failed to load high-frequency tensor context: {e}")
        return

    while True:
        # Load state
        state = load_evolution_state()

        active_round = state.get("active_round", 1)
        current_layer = state.get("current_layer", 1)

        # Verify if current_layer needs resetting (e.g. if previous layer's raw material pool is empty)
        if current_layer > 1:
            prev_pool_key = f"L{current_layer}_pool"
            if not state.get("pools", {}).get(prev_pool_key, []):
                logger.warning(f"Raw material pool '{prev_pool_key}' is empty! Resetting current_layer to 1.")
                state["current_layer"] = 1
                save_evolution_state(state)
                current_layer = 1

        logger.info("\n" + "="*75)
        logger.info(f"🔄  Starting GP Evolution Round {active_round}, Layer {current_layer} / 5")
        logger.info("="*75)

        # Run current layer generation
        run_layer_generation(state, current_layer)

        # Reload state to get latest updates
        state = load_evolution_state()

        # Move to next layer or next round
        if current_layer < 5:
            state["current_layer"] = current_layer + 1
            save_evolution_state(state)
        else:
            # End of round! Print report and increment round
            logger.info("="*75)
            logger.info(f"🏁 Round {active_round} completed! Resetting to Layer 1 for Round {active_round + 1}...")
            logger.info("="*75)

            # Print the survivors of the round
            final_pool = state.get("active_pool", [])
            if final_pool:
                # Sort by annualized IR
                final_pool_sorted = sorted(final_pool, key=lambda x: abs(x.get("metrics", {}).get("ir", 0.0)), reverse=True)
                for idx, f in enumerate(final_pool_sorted):
                    m = f.get("metrics", {}) or {}
                    logger.info(
                        f"🏆 [Round {active_round} Survivor] #{idx+1} ID: {f['name']}\n"
                        f"   公式: {f['formula']}\n"
                        f"   主要指标:\n"
                        f"     - Annualized IR = {m.get('ir', 0.0):.3f}\n"
                        f"     - Whole Year Pooled IC (Full_IC) = {m.get('full_ic', 0.0):.5f}\n"
                        f"     - ANOVA R2 = {m.get('full_anova', 0.0):.5f}\n"
                        f"     - Mutual Information = {m.get('full_mi', 0.0):.5f}\n"
                    )

                    # Export top survivors of each round to research_log.json
                    try:
                        r_log_path = project_root / f"research_log_{FOCUS_UNIVERSE.lower()}.json"
                        if r_log_path.exists():
                            with open(r_log_path, "r", encoding="utf-8") as rf:
                                r_data = json.load(rf)

                            # Prevent duplicates
                            existing_formulas = [kf["formula"] for kf in r_data.get("keeper_factors", [])]
                            if f["formula"] not in existing_formulas:
                                r_data.setdefault("keeper_factors", []).append({
                                    "factor_name": f"Evolved_{f['name']}",
                                    "formula": f["formula"],
                                    "ir": m.get("ir", 0.0),
                                    "ic": m.get("ic", 0.0),
                                    "full_ic": m.get("full_ic", 0.0),
                                    "rationale": f["rationale"],
                                    "timestamp": datetime.now().isoformat()
                                })
                                with open(r_log_path, "w", encoding="utf-8") as wf:
                                    json.dump(r_data, wf, indent=2, ensure_ascii=False)
                    except Exception as le:
                        logger.error(f"Failed to record evolved survivor: {le}")
            else:
                logger.warning("No factors survived the current round.")

            # Transition to next round: increment round, reset layer, clear active_pool (round-specific propagation)
            state["active_round"] = active_round + 1
            state["current_layer"] = 1
            state["active_pool"] = []
            save_evolution_state(state)


if __name__ == "__main__":
    start_evolutionary_pipeline()
