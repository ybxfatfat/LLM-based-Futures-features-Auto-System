"""
manage_libraries.py - Interactive CLI Tool to Manage Foundation Features and Compound Operators.

Allows humans to describe quantitative ideas in Chinese, and leverages LLMs
to translate them into rule-based, fully compiled JSON templates, automatically registering them into
the respective libraries.

Created for High-Frequency Quantitative Research.
Generated with AI & LLM-based Architecture.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from libraries import (
    load_foundation_features,
    load_compound_operators,
    register_foundation_feature,
    register_compound_operator,
)

# ======================================================================
# LLM Client Wrapper (OpenAI Compatible)
# ======================================================================
from pydantic import BaseModel, Field
from openai import OpenAI

# 实例化标准 OpenAI 客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# 默认使用的模型名称，可通过环境变量覆写
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        
    def to_dict(self):
        return {"role": self.role, "content": self.content}

class Chat:
    """
    轻量级 LLM 包装器
    """
    def __init__(self, model: str, messages: list[Message], temperature: float = 0.3):
        self.model = model
        self.messages = messages
        self.temperature = temperature

    def create(self):
        dict_messages = [m.to_dict() for m in self.messages]
        response = client.chat.completions.create(
            model=self.model,
            messages=dict_messages,
            temperature=self.temperature
        )
        return response


def clean_llm_json(content: str) -> str:
    """Safely extracts JSON code block from response."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()
    match_braces = re.search(r"(\{.*\})", content, re.DOTALL | re.MULTILINE)
    if match_braces:
        return match_braces.group(1).strip()
    return content.strip()


def ai_generate_foundation_feature(idea_description: str) -> dict:
    """Invokes LLM to translate a human idea into a custom foundation feature."""
    prompt = f"""
你是一个顶级期货量化研究专家。你需要将用户描述的基础特征想法转换为我们特征系统所需的结构化JSON配置。

用户想法描述：
"{idea_description}"

【基础特征系统规则与可用底层字段】
基础特征是在原始微观字段上进行代数计算得到的复合特征。公式中可以使用的底层字段包括：
"open", "high", "low", "close", "open_trd", "high_trd", "low_trd", "close_trd", "vwap", "volume", "turnover", "bid1", "ask1", "tick_count", "tick_count_trd"
不允许使用任何带 _adj 后缀的复权字段，也不要使用任何时序滑动窗口（如 TsMean 以外的操作），时序窗口留给后置的 Temporal Operators。

你可以使用的内置基础数学算子：
- Abs(x)
- Log(x)
- Sqrt(x)
- Sign(x)
- IndicatorMoreThanZero(x) (x > 0 ? 1 : 0)
- DistortLog(x) (保留符号的对数：sign(x)*log1p(|x|))
- DistortSqrt(x) (保留符号的开根号)
- Add(x, y), Sub(x, y), Prod(x, y), Div(x, y) (或者直接用 +, -, *, /)
- MoreThan(x, y), LessThan(x, y), Equal(x, y)
- IfElse(cond, x, y)
- FillNa(x, fill_val)

【输出 JSON 格式要求】
你必须且只能输出一个合法的 JSON 代码块，不含有任何 markdown、前言或后缀解释。其字段如下：
{{
  "name": "英文大写驼峰或下划线命名，如 Price_Volume_Imbalance",
  "formula": "完整的数学代数公式，例如: 'close_trd * (volume / (TsMean(volume, 60) + 1e-8))'",
  "description": "精简、专业的中文量化逻辑说明",
  "field_type": "必须是 'price', 'volume', 'spread', 'ratio' 之一"
}}

注意：请务必保证代数公式的严谨，防止除 0（建议分母加 + 1e-8）。
"""
    print(f"🤖 正在调用 [{DEFAULT_LLM_MODEL}] 精英量化翻译器设计基础特征...")
    chat_obj = Chat(
        model=DEFAULT_LLM_MODEL,
        messages=[Message(role="user", content=prompt)],
        temperature=0.1
    )
    response = chat_obj.create()
    raw_content = response.choices[0].message.content
    json_str = clean_llm_json(raw_content)
    return json.loads(json_str)


def ai_generate_compound_operator(idea_description: str) -> dict:
    """Invokes LLM to translate a human idea into a custom compound operator."""
    prompt = f"""
你是一个顶级期货量化研究专家。你需要将用户描述的复合时序算子想法转换为我们系统所需的结构化JSON配置。

用户想法描述：
"{idea_description}"

【复合算子系统规则与可用算子】
复合算子由多个内置的一元/二元算子组合而成。它接受参数，并且至少包含一个输入数据占位符 `{{input}}` 和一个历史滚动回看窗口占位符 `{{window}}`。
模板公式格式例如：`TsStd(IfElse(MoreThan({{input}}, 0), {{input}}), 0), {{window}})`。

你可以使用的内置基础数学和时序算子：
- 数学算子: Abs(x), Log(x), Sqrt(x), Sign(x), IndicatorMoreThanZero(x), DistortLog(x), DistortSqrt(x), Add, Sub, Prod, Div, IfElse
- 时序算子 (TS Ops):
  - TsDelay(x, window) (向前平移)
  - TsDelta(x, window) (一阶差分)
  - TsPct(x, window) (稳健收益率)
  - TsMean(x, window) (滚动均值)
  - TsStd(x, window) (滚动标准差)
  - TsZscore(x, window) (滚动标准化)
  - TsSum(x, window) (滚动求和)
  - TsMax(x, window), TsMin(x, window) (滚动最大/最小值)
  - TsMaxDiff(x, window) (x - 滚动最大)
  - TsMinDiff(x, window) (x - 滚动最小)
  - TsArgMax(x, window), TsArgMin(x, window) (最大/最小值发生的位置)
  - TsMedian(x, window) (滚动中位数)
  - TsQuantile(x, window, q) (滚动分位数，q 为 float)
  - TsRank(x, window) (滚动百分比排序)
  - TsSkewness(x, window), TsKurtosis(x, window) (高阶矩)
  - TsEntropy(x, window), TsCv(x, window), TsIr(x, window) (滚动信息比率)
  - EWMA(x, window) (指数移动平均)

【输出 JSON 格式要求】
你必须且只能输出一个合法的 JSON 代码块，不含有任何 markdown、前言或后缀解释。其字段如下：
{{
  "name": "英文大写驼峰或下划线命名，如 Upward_Volatility",
  "parameters": ["input", "window"],
  "formula_template": "含 {{input}} 和 {{window}} 占位符的复合公式，例如: 'TsStd(IfElse(MoreThan({{input}}, 0), {{input}}, 0), {{window}})'",
  "description": "精简、专业的中文量化逻辑说明"
}}
"""
    print(f"🤖 正在调用 [{DEFAULT_LLM_MODEL}] 精英量化翻译器设计复合时序算子...")
    chat_obj = Chat(
        model=DEFAULT_LLM_MODEL,
        messages=[Message(role="user", content=prompt)],
        temperature=0.1
    )
    response = chat_obj.create()
    raw_content = response.choices[0].message.content
    json_str = clean_llm_json(raw_content)
    return json.loads(json_str)


def interactive_menu():
    """Starts the interactive CLI loop."""
    print("=" * 80)
    print("📈 AI 量化特征工程基础库管理系统 (Quant Library Manager CLI) 📈")
    print("=" * 80)
    print("1. 查看当前已注册的基础特征 (List Foundation Features)")
    print("2. 查看当前已注册的复合算子 (List Compound Operators)")
    print("3. 【人机交互设计】描述想法并自动注册新基础特征 (AI Design Foundation Feature)")
    print("4. 【人机交互设计】描述想法并自动注册新复合算子 (AI Design Compound Operator)")
    print("5. 退出系统 (Exit)")
    print("-" * 80)

    try:
        choice = input("请选择操作 [1-5]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 退出特征库管理器。")
        return False

    if choice == "1":
        feats = load_foundation_features()
        print("\n--- 📌 当前已注册基础特征库 ----------------------------------------------")
        if not feats:
            print("目前没有任何特征注册。")
        for k, v in feats.items():
            print(f"🔹 特征名: {v['name']}  |  类型: {v['field_type']}")
            print(f"   公式:   {v['formula']}")
            print(f"   说明:   {v['description']}")
            print("-" * 60)
        print("-------------------------------------------------------------------------")

    elif choice == "2":
        ops = load_compound_operators()
        print("\n--- 📌 当前已注册复合算子库 ----------------------------------------------")
        if not ops:
            print("目前没有任何复合算子注册。")
        for k, v in ops.items():
            print(f"🔸 算子名: {v['name']}  |  参数: {v['parameters']}")
            print(f"   模板:   {v['formula_template']}")
            print(f"   说明:   {v['description']}")
            print("-" * 60)
        print("-------------------------------------------------------------------------")

    elif choice == "3":
        idea = input("\n📝 请用中文描述您的基础特征设计想法 (例如: 最新收盘价减去前一分钟收盘价，再除以买一和卖一档之差加上微小的安全项): ").strip()
        if not idea:
            print("❌ 描述不能为空。")
            return True
        try:
            result = ai_generate_foundation_feature(idea)
            print("\n💡 AI 生成的特征配置如下:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            confirm = input("❓ 是否确认将其保存到基础特征库？[y/N]: ").strip().lower()
            if confirm == "y":
                register_foundation_feature(
                    name=result["name"],
                    formula=result["formula"],
                    description=result["description"],
                    field_type=result["field_type"]
                )
                print(f"🎉 成功：新基础特征 [{result['name']}] 已经成功注册并保存！")
            else:
                print("⚠️ 已取消保存。")
        except Exception as e:
            print(f"❌ 设计基础特征失败: {e}")

    elif choice == "4":
        idea = input("\n📝 请用中文描述您的复合时序算子设计想法 (例如: 先计算输入的滚动标准差，再除以输入的滚动均值，得到滚动变异系数指标): ").strip()
        if not idea:
            print("❌ 描述不能为空。")
            return True
        try:
            result = ai_generate_compound_operator(idea)
            print("\n💡 AI 生成的复合算子配置如下:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            confirm = input("❓ 是否确认将其保存到复合算子库？[y/N]: ").strip().lower()
            if confirm == "y":
                register_compound_operator(
                    name=result["name"],
                    parameters=result["parameters"],
                    formula_template=result["formula_template"],
                    description=result["description"]
                )
                print(f"🎉 成功：新复合算子 [{result['name']}] 已经成功注册并保存！")
            else:
                print("⚠️ 已取消保存。")
        except Exception as e:
            print(f"❌ 设计复合算子失败: {e}")

    elif choice == "5":
        print("👋 退出特征库管理器。")
        return False
    else:
        print("❌ 无效的选项，请重新选择。")

    print("\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Library Manager CLI")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start in interactive CLI mode")
    parser.add_argument("--add-feature", "-f", type=str, help="Describe foundation feature to add via AI")
    parser.add_argument("--add-operator", "-o", type=str, help="Describe compound operator to add via AI")
    args = parser.parse_args()

    # If no arguments are provided, default to interactive
    if not (args.add_feature or args.add_operator):
        args.interactive = True

    if args.interactive:
        active = True
        while active:
            active = interactive_menu()
    elif args.add_feature:
        try:
            res = ai_generate_foundation_feature(args.add_feature)
            register_foundation_feature(res["name"], res["formula"], res["description"], res["field_type"])
            print(f"🎉 [Success] Feature '{res['name']}' registered: {res['formula']}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            sys.exit(1)
    elif args.add_operator:
        try:
            res = ai_generate_compound_operator(args.add_operator)
            register_compound_operator(res["name"], res["parameters"], res["formula_template"], res["description"])
            print(f"🎉 [Success] Operator '{res['name']}' registered: {res['formula_template']}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            sys.exit(1)