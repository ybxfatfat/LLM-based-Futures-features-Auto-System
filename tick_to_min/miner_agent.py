# miner_agent.py
import ast
import os
import json
from pathlib import Path
from openai import OpenAI
from prompts_config import SYSTEM_PROMPT
from rag_vector_store import LocalFactorKB

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = MODULE_DIR / "auto_generated_factors.py"
KB = LocalFactorKB()

def initialize_output_file():
    if not OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("# 🤖 LLM 生成、待人工审查与显式注册的 L2 微观特征候选库\n")
            f.write("from tick_framework import BaseTickFeature\n")
            f.write("import tick_engine as ce\n\n")


def validate_generated_class(class_code: str, feature_name: str) -> str:
    """Validate syntax and the single-class output contract without executing LLM code."""
    if not isinstance(class_code, str) or not class_code.strip():
        raise ValueError("python_class_code 为空")
    if not isinstance(feature_name, str) or not feature_name.isidentifier():
        raise ValueError("feature_name 必须是合法的 Python 类名")

    tree = ast.parse(class_code)
    top_level_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    executable_top_level = [
        node for node in tree.body
        if not (
            isinstance(node, ast.ClassDef)
            or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        )
    ]

    if executable_top_level:
        raise ValueError("python_class_code 只能包含类定义，不能包含导入或顶层执行语句")
    if len(top_level_classes) != 1:
        raise ValueError("python_class_code 必须且只能包含一个顶层类定义")

    class_node = top_level_classes[0]
    class_name = class_node.name
    if class_name != feature_name:
        raise ValueError(
            f"feature_name={feature_name!r} 与生成类名 {class_name!r} 不一致"
        )

    if not any(
        isinstance(base, ast.Name) and base.id == "BaseTickFeature"
        for base in class_node.bases
    ):
        raise ValueError("生成类必须继承 BaseTickFeature")

    method_names = {
        node.name for node in class_node.body if isinstance(node, ast.FunctionDef)
    }
    missing_methods = {"__init__", "update", "get_value"} - method_names
    if missing_methods:
        missing = ", ".join(sorted(missing_methods))
        raise ValueError(f"生成类缺少必要方法: {missing}")

    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise ValueError("生成类不得自行导入模块")

    existing_tree = ast.parse(OUTPUT_FILE.read_text(encoding="utf-8"))
    existing_names = {
        node.name for node in existing_tree.body if isinstance(node, ast.ClassDef)
    }
    if class_name in existing_names:
        raise ValueError(f"候选类 {class_name!r} 已存在，请先人工审查或更名")

    return class_name

def generate_feature(idea_description: str):
    initialize_output_file()
    
    # 1. 查询 RAG 知识库
    print(f"\n🔍 正在 RAG 知识库中检索相关历史经验...")
    similar_records = KB.query_similar_experience(idea_description)
    
    rag_context = ""
    if similar_records:
        rag_context = "\n\n【⚠️ RAG 历史经验告警】\n以下是我们过去尝试过的类似特征及其反思，请务必吸取教训，调整当前特征的计算方式：\n"
        for rec in similar_records:
            metrics = rec.get("metrics", {}) or {}
            win_rate = metrics.get("win_rate", metrics.get("Win_Rate", "N/A"))
            rag_context += f"- 历史特征 [{rec['feature_name']}]: 胜率 {win_rate}\n"
            rag_context += f"  反思结论: {rec['reflection']}\n"
    else:
        print("   (未发现高相似度历史记录，这是全新的探索方向！)")

    # 2. 组装 Prompt 并调用 LLM
    final_system_prompt = SYSTEM_PROMPT + rag_context
    print(f"🧠 正在请求大模型生成严格合规的 C++ 符号计算图代码...")
    
    response = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        response_format={ "type": "json_object" }, # 强制稳定输出 JSON
        messages=[
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": f"新特征灵感：{idea_description}"}
        ],
        temperature=0.3
    )

    try:
        reply_json = json.loads(response.choices[0].message.content)
        class_code = reply_json["python_class_code"]
        feat_name = reply_json["feature_name"]
        mech = reply_json["hypothesis"]["mechanism"]
    except Exception as e:
        print(f"❌ 解析 LLM 输出失败: {e}")
        return

    # 3. 仅保存通过静态校验的候选代码；不执行，也不自动挂载到 run.py
    try:
        validate_generated_class(class_code, feat_name)
    except (SyntaxError, ValueError) as e:
        print(f"❌ 生成代码未通过静态校验，未写入候选库: {e}")
        return

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n# ==========================================\n")
        f.write(f"# 💡 Feature: {feat_name}\n")
        f.write(f"# 🧠 Mechanism: {mech}\n")
        f.write(f"# ==========================================\n")
        f.write(class_code + "\n\n")

    print(f"✅ 候选算子 [{feat_name}] 已通过静态校验并保存至 {OUTPUT_FILE}")
    print("⚠️  该类不会被 run.py 自动加载；请人工审查后再显式加入特征注册列表。")

if __name__ == "__main__":
    print("🧬 LLM-RAG L2 Tick Feature Miner")
    while True:
        idea = input("\n💡 请输入微观特征挖掘灵感 (输入 q 退出): ")
        if idea.lower() in ['q', 'quit', 'exit']:
            break
        generate_feature(idea)
