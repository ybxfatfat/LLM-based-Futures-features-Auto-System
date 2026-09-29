# miner_agent.py
import os
import json
import re
from openai import OpenAI
from prompts_config import SYSTEM_PROMPT
from rag_vector_store import LocalFactorKB

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OUTPUT_FILE = "auto_generated_factors.py"
KB = LocalFactorKB()

def initialize_output_file():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("# 🤖 LLM 自动生成的 L2 微观特征库\n")
            f.write("from tick_framework import BaseTickFeature\n")
            f.write("import tick_engine as ce\n\n")

def generate_feature(idea_description: str):
    initialize_output_file()
    
    # 1. 查询 RAG 知识库
    print(f"\n🔍 正在 RAG 知识库中检索相关历史经验...")
    similar_records = KB.query_similar_experience(idea_description)
    
    rag_context = ""
    if similar_records:
        rag_context = "\n\n【⚠️ RAG 历史经验告警】\n以下是我们过去尝试过的类似特征及其反思，请务必吸取教训，调整当前特征的计算方式：\n"
        for rec in similar_records:
            rag_context += f"- 历史特征 [{rec['feature_name']}]: 胜率 {rec['metrics'].get('win_rate', 'N/A')}\n"
            rag_context += f"  反思结论: {rec['reflection']}\n"
    else:
        print("   (未发现高相似度历史记录，这是全新的探索方向！)")

    # 2. 组装 Prompt 并调用 LLM
    final_system_prompt = SYSTEM_PROMPT + rag_context
    print(f"🧠 正在请求大模型生成严格合规的 C++ 符号计算图代码...")
    
    response = CLIENT.chat.completions.create(
        model="gpt-4o",
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

    # 3. 自动注入本地代码库
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n# ==========================================\n")
        f.write(f"# 💡 Feature: {feat_name}\n")
        f.write(f"# 🧠 Mechanism: {mech}\n")
        f.write(f"# ==========================================\n")
        f.write(class_code + "\n\n")
        
    print(f"✅ 成功提取算子 [{feat_name}] 并无缝注入至 {OUTPUT_FILE}！")
    print(f"   (下次运行 run.py 时将被自动加载)")

if __name__ == "__main__":
    print("🧬 LLM-RAG L2 Tick Feature Miner")
    while True:
        idea = input("\n💡 请输入微观特征挖掘灵感 (输入 q 退出): ")
        if idea.lower() in ['q', 'quit', 'exit']:
            break
        generate_feature(idea)