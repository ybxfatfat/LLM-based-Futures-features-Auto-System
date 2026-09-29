# select_state.py
import json
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import os

# ============================================================
# 0. 路径与参数
# ============================================================
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
DATA_DIR = project_root / "data"

# 注意根据你实际想提取的宇宙修改这里的后缀 (index 或 commodity)
JSON_PATH = str(DATA_DIR / "market_states_registry_index.json")
CORR_MATRIX_PATH = str(DATA_DIR / "market_states_corr_matrix.csv")

# 💥 适配新版 State Miner 的独立性指标
TARGET_METRIC = "mutual_info"  

CORR_THRESHOLD = 0.7               
MIN_EVALS = 50                     
# 💥 互信息的阈值通常在 0.1 到 0.4 之间，这里设为 0.15 过滤纯噪声
MIN_METRIC_VAL = 0.15               

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("📦 加载状态变量注册表与相关性矩阵...")
with open(JSON_PATH, 'r') as f:
    registry = json.load(f)

corr_df = pd.read_csv(CORR_MATRIX_PATH, index_col=0)
corr_matrix = corr_df.fillna(0.0).abs()
features = corr_df.columns.tolist()

# ============================================================
# 2. 构建相关图 & 社区发现
# ============================================================
print("🕸️ 构建状态变量相关图...")
G = nx.Graph()
G.add_nodes_from(features)

edges = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        w = corr_matrix.iloc[i, j]
        if w > CORR_THRESHOLD:
            edges.append((features[i], features[j], w))

G.add_weighted_edges_from(edges)
print(f"   图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

print("🧠 社区发现 (Greedy Modularity)...")
communities = list(community.greedy_modularity_communities(G, weight='weight'))
print(f"   共划分出 {len(communities)} 个高内聚社区")

# ============================================================
# 3. 核心提取逻辑
# ============================================================
def get_state_score(state_name, registry_data, metric_name):
    """
    提取状态变量得分。必须满足基础的 passed 和样本量校验。
    """
    payload = registry_data.get(state_name, {})
    stats = payload.get("stats", {})
    
    if not stats.get("passed", False):
        return -np.inf
    if stats.get("valid_evals", 0) < MIN_EVALS:
        return -np.inf
        
    # 💥 修改：从新版字典的 metrics 键中提取指标
    metrics = stats.get("metrics", {})
    val = metrics.get(metric_name, None)
    
    if val is None or np.isnan(val):
        return -np.inf
        
    return val

# ============================================================
# 4. 社区内打擂台
# ============================================================
print(f"\n⚔️ 社区内打擂台 (目标指标: {TARGET_METRIC})...")
selected_states = []

for comm_idx, comm in enumerate(communities):
    best_state = None
    best_score = -np.inf
    
    for state in comm:
        # 有些相关性矩阵里的列可能在 JSON 里没有，需做保护
        if state not in registry:
            continue
            
        score = get_state_score(state, registry, TARGET_METRIC)
        if score > best_score:
            best_score = score
            best_state = state
            
    if best_state is not None and best_score >= MIN_METRIC_VAL:
        selected_states.append({
            "community_idx": comm_idx,
            "state_name": best_state,
            "category": registry[best_state].get("category", "Unknown"),
            "score": best_score,
            "community_size": len(comm)
        })

# 按分数降序排列
selected_states.sort(key=lambda x: x["score"], reverse=True)

print(f"   ✅ 最终保留 {len(selected_states)} 个正交且强力的状态变量。")
if selected_states:
    print(f"   📊 得分范围 ({TARGET_METRIC}): {selected_states[-1]['score']:.2f} ~ {selected_states[0]['score']:.2f}")

# ============================================================
# 5. 存储落盘
# ============================================================
out_list_path = str(DATA_DIR / "best_market_states_list.json")
out_detail_path = str(DATA_DIR / "best_market_states_detail.json")

# 仅保存名称列表
state_names_only = [r["state_name"] for r in selected_states]
with open(out_list_path, "w", encoding="utf-8") as f:
    json.dump(state_names_only, f, indent=2)

# 保存带分数的详情
with open(out_detail_path, "w", encoding="utf-8") as f:
    json.dump(selected_states, f, indent=2, ensure_ascii=False)

print(f"💾 列表已保存: {out_list_path}")
print(f"💾 明细已保存: {out_detail_path}")
print("🎉 完成！")