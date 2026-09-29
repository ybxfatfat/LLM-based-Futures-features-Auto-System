# eval_and_reflect.py
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from rag_vector_store import LocalFactorKB

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
KB = LocalFactorKB()

def evaluate_and_reflect(parquet_path: str, feature_name: str, target_col: str = "Ret_1m_Fwd"):
    """评测特征的真实分钟频收益率，并生成大模型反思"""
    print(f"📊 正在评测高频特征 [{feature_name}] 的真实盈亏能力...")
    
    # 1. 数据对齐与多空分组 (Decile)
    df = pd.read_parquet(parquet_path, columns=[feature_name, target_col]).dropna()
    if len(df) < 1000:
        print("❌ 有效样本过少，无法评估。")
        return
        
    df['Decile'] = pd.qcut(df[feature_name], q=10, labels=False, duplicates='drop')
    grouped_ret = df.groupby('Decile')[target_col].mean() * 10000 # 转换为 bps
    
    try:
        q1_ret = grouped_ret.iloc[0]
        q10_ret = grouped_ret.iloc[-1]
        spread = q10_ret - q1_ret
    except IndexError:
        spread, q1_ret, q10_ret = 0, 0, 0
        
    # 计算胜率与 IC 作为辅助指标
    ic = df[feature_name].corr(df[target_col])
    win_rate = (np.sign(df[feature_name]) == np.sign(df[target_col])).mean()
    
    metrics = {
        "Q10_Minus_Q1_bps": float(spread),
        "Q10_Ret_bps": float(q10_ret),
        "Q1_Ret_bps": float(q1_ret),
        "Full_IC": float(ic),
        "Win_Rate": float(win_rate)
    }
    
    print(f"   ➤ 分组多空收益差: {spread:.2f} bps | 全样本 IC: {ic:.4f}")

    # 2. 从本地读取刚刚生成的类代码和假设
    with open("auto_generated_factors.py", "r", encoding="utf-8") as f:
        full_code = f.read()
    
    # 3. 让 LLM 进行自动反思
    print("🧠 正在生成后验反思诊断报告...")
    reflection_prompt = f"""
    你是一个严苛的高频量化裁判。
    我们刚刚测试了一个名为 {feature_name} 的特征。
    它的测试结果如下：
    - 多空收益差 (Q10-Q1): {spread:.2f} bps (未来 1 分钟收益)
    - 胜率: {win_rate*100:.2f}%
    - IC: {ic:.4f}
    
    如果收益差在 [-1.0, 1.0] bps 之间，说明特征基本上是无效的纯噪声。
    如果收益差绝对值 > 1.5 bps，说明特征具备极强的短线微观预测力。
    
    请根据这个表现，结合微观交易常识，写出一段 100 字左右的反思结论（不要给出修补建议，只要指出为什么会呈现这种结果，例如滑点、被算法单掩盖、逻辑正确但方向相反等）。
    """
    
    response = CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": reflection_prompt}]
    )
    reflection = response.choices[0].message.content.strip()
    
    print(f"   ➤ 裁判反思: {reflection}")
    
    # 4. 录入 RAG 闭环
    # 构造一个虚的 hypothesis dict 以兼容 RAG
    hyp_dict = {"mechanism": f"测试 {feature_name} 的实盘表现"}
    KB.add_experience(feature_name, hyp_dict, "Class Code Saved", metrics, reflection)

if __name__ == "__main__":
    # 示例调用
    # evaluate_and_reflect("./data/features_output/2024/feat_day_20251128_1234.parquet", "StealthAskRetreatMomentum")
    print("运行此脚本以自动化评估特征并注入 RAG 知识库。")