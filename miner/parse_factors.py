"""
parse_factors.py - AST Parser and ECharts Visualizer for DL Feature Discovery.

Parses the Alpha Hall of Fame (JSON), extracts sub-expressions using AST,
and evaluates these "Roots" based on the performance (IR/IC) of their descendants.
Identifies high-value sub-graphs to be used as raw input features for Time-Series
Deep Learning models (e.g., Transformers, LSTMs).

Created for High-Frequency Quantitative Research.
"""

import json
import ast
from collections import defaultdict
from pathlib import Path

# ==========================================
# 1. 动态路径配置
# ==========================================
project_root = Path(__file__).resolve().parent

# 直接读取进化挖掘机生成的“名人堂”文件（自带指标）
INPUT_HALL_OF_FAME = project_root / 'alpha_hall_of_fame_commodity.json'
OUTPUT_ROOTS_JSON = project_root / 'dl_feature_roots.json'
OUTPUT_HTML_VIS = project_root / 'dl_feature_lineage.html'


# ==========================================
# 2. 健壮的 AST 语法树遍历器
# ==========================================
class SubExpressionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.sub_expressions = set() # 使用 set 防止同一个公式里重复统计自身
        
    def visit(self, node):
        # 捕捉有意义的函数调用 (如 TsMean, TsCorr) 和 二元运算 (如 a / b)
        if isinstance(node, (ast.Call, ast.BinOp)):
            try:
                expr_str = ast.unparse(node)
                # 过滤掉过于简单的纯数字或短变量
                if len(expr_str) > 8 and not expr_str.isdigit():
                   self.sub_expressions.add(expr_str)
            except:
                pass
        self.generic_visit(node)


# ==========================================
# 3. 数据加载与 Root 挖掘逻辑
# ==========================================
def main():
    print("🔍 正在加载 Alpha 名人堂数据...")
    factors = []
    
    if INPUT_HALL_OF_FAME.exists():
        try:
            with open(INPUT_HALL_OF_FAME, 'r', encoding="utf-8") as f:
                factors = json.load(f)
        except Exception as e:
            print(f"❌ 读取名人堂失败: {e}")
    else:
        print(f"⚠️ 未找到 {INPUT_HALL_OF_FAME.name}，启用 Demo 数据集生成可视化演示...")
        factors = [
            {"name": "Vol_Trend_1", "formula": "TsMean(volume, 60) / (TsMean(volume, 120) + 1e-8)", "ir": 1.85, "full_ic": 0.015},
            {"name": "Vol_Trend_2", "formula": "TsCorr(close_trd, TsMean(volume, 60), 30)", "ir": 2.15, "full_ic": 0.018},
            {"name": "Price_Dev", "formula": "(close_trd - TsMean(close_trd, 30)) / (TsStd(close_trd, 30) + 1e-8)", "ir": 1.50, "full_ic": 0.012},
            {"name": "Complex_Momentum", "formula": "TsDelta((close_trd - TsMean(close_trd, 30)) / (TsStd(close_trd, 30) + 1e-8), 5)", "ir": 2.45, "full_ic": 0.021}
        ]

    if not factors:
        print("❌ 没有可用的因子数据，停止执行。")
        return

    print(f"📊 成功加载 {len(factors)} 个高分因子。启动 AST 拆解引擎...")

    # 映射字典: Root 表达式 -> 衍生出的因子列表
    root_to_factors = defaultdict(list)

    for f_data in factors:
        formula = f_data.get("formula", "")
        if not formula:
            continue
            
        try:
            tree = ast.parse(formula, mode='eval')
            visitor = SubExpressionVisitor()
            visitor.visit(tree)
            
            for expr in visitor.sub_expressions:
                # 排除掉整个公式本身（只找真正的子集 Root）
                if expr != formula:
                    root_to_factors[expr].append(f_data)
        except Exception as e:
            print(f"⚠️ 解析公式失败 [{f_data.get('name')}]: {e}")

    # ==========================================
    # 4. 深度学习特征 (DL Features) 价值评分体系
    # ==========================================
    dl_roots = {}
    for root_expr, descendant_factors in root_to_factors.items():
        # 核心逻辑：一个 Root 至少要在 2 个不同的强因子中作为子图出现，才具备“通用底层特征”的价值
        if len(descendant_factors) >= 2:
            # 找到基于这个 Root 变异出的最强因子的 IR
            max_ir = max([abs(f.get("ir", 0.0)) for f in descendant_factors])
            avg_ic = sum([abs(f.get("full_ic", 0.0)) for f in descendant_factors]) / len(descendant_factors)
            
            dl_roots[root_expr] = {
                "frequency": len(descendant_factors),
                "max_descendant_ir": max_ir,
                "avg_descendant_ic": avg_ic,
                "descendant_names": [f["name"] for f in descendant_factors]
            }

    # 排序：优先按孕育出的 Max IR 排序，其次按出现频率
    sorted_roots = dict(sorted(dl_roots.items(), key=lambda x: (x[1]["max_descendant_ir"], x[1]["frequency"]), reverse=True))

    try:
        with open(OUTPUT_ROOTS_JSON, 'w', encoding="utf-8") as f:
            json.dump(sorted_roots, f, indent=4, ensure_ascii=False)
        print(f"✅ 提炼出 {len(sorted_roots)} 个深度学习候选 Root 特征，已保存至 {OUTPUT_ROOTS_JSON.name}")
    except Exception as e:
        print(f"❌ 保存 Root JSON 失败: {e}")

    print("\n🏆 Top 3 最强深度学习候选 Root (Highest Descendant IR):")
    for k, v in list(sorted_roots.items())[:3]:
        print(f"  [Max IR: {v['max_descendant_ir']:.3f} | 衍生族群数: {v['frequency']}] -> {k}")


    # ==========================================
    # 5. 构建 ECharts 关系图数据 (富数据注入)
    # ==========================================
    graph_nodes = []
    graph_links = []
    node_ids = set()

    # 1. 添加 Root 节点 (大小与颜色由 Max IR 决定)
    for root_expr, metrics in sorted_roots.items():
        root_id = f"ROOT_{hash(root_expr)}"
        
        # 节点大小基于衍生出的最强 IR (保障深度学习视角下的重要性视觉突出)
        symbol_size = max(15, min(45, metrics['max_descendant_ir'] * 10))
        
        graph_nodes.append({
            "id": root_id,
            "name": (root_expr[:35] + "...") if len(root_expr) > 35 else root_expr,
            "full_expr": root_expr,
            "category": 0, # 0 = DL Root
            "symbolSize": symbol_size,
            "metrics": f"Max Descendant IR: {metrics['max_descendant_ir']:.3f}<br/>Frequency: {metrics['frequency']}"
        })
        node_ids.add(root_id)

    # 2. 添加 因子 节点，并建立连线
    for f_data in factors:
        formula = f_data.get("formula", "")
        f_name = f_data.get("name", "Unknown")
        ir = f_data.get("ir", 0.0)
        ic = f_data.get("full_ic", 0.0)
        
        factor_id = f"FACTOR_{f_name}"
        
        # 检查这个因子是否包含我们提取出的优质 Root
        has_root = False
        for root_expr in sorted_roots.keys():
            if root_expr in formula:
                has_root = True
                root_id = f"ROOT_{hash(root_expr)}"
                graph_links.append({
                    "source": root_id, # 从 Root 指向最终因子
                    "target": factor_id
                })
        
        # 只有当因子包含 Root，或者我们想展示全貌时才加入（这里为了图表清晰，只展示有连线的族群）
        if has_root:
            if factor_id not in node_ids:
                graph_nodes.append({
                    "id": factor_id,
                    "name": f_name,
                    "full_expr": formula,
                    "category": 1, # 1 = Final Factor
                    "symbolSize": 10,
                    "metrics": f"Annualized IR: {ir:.3f}<br/>Full Year IC: {ic:.5f}"
                })
                node_ids.add(factor_id)

    # ==========================================
    # 6. 生成交互式 HTML 可视化
    # ==========================================
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>DL Feature Lineage Graph</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #101010; color: #fff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            #main {{ width: 100vw; height: 100vh; }}
            .panel {{ position: absolute; top: 20px; left: 20px; z-index: 100; background: rgba(30, 30, 30, 0.8); padding: 15px; border-radius: 8px; border: 1px solid #444; }}
            h2 {{ margin: 0 0 10px 0; font-size: 18px; color: #00e5ff; }}
            p {{ margin: 5px 0; font-size: 13px; color: #ccc; }}
        </style>
    </head>
    <body>
        <div class="panel">
            <h2>Deep Learning Input Features (Sub-graph Lineage)</h2>
            <p><span style="color: #00e5ff; font-weight: bold;">● Cyan (Roots)</span>: Base Sub-expressions (Size reflects Max Descendant IR)</p>
            <p><span style="color: #ff3366; font-weight: bold;">● Pink (Factors)</span>: Evolved End-Node Factors</p>
        </div>
        <div id="main"></div>
        <script>
            var chartDom = document.getElementById('main');
            var myChart = echarts.init(chartDom, 'dark');
            
            const graphNodes = {json.dumps(graph_nodes)};
            const graphLinks = {json.dumps(graph_links)};
            
            const option = {{
                backgroundColor: '#101010',
                tooltip: {{
                    trigger: 'item',
                    backgroundColor: 'rgba(20,20,20,0.9)',
                    borderColor: '#666',
                    textStyle: {{ color: '#fff' }},
                    formatter: function (params) {{
                        if (params.dataType === 'node') {{
                            let typeName = params.data.category === 0 
                                ? '<strong style="color:#00e5ff">Deep Learning Root (Sub-graph)</strong>' 
                                : '<strong style="color:#ff3366">Evolved Alpha Factor</strong>';
                            return typeName + '<br/><hr style="border-color:#444"/>' +
                                   '<b>Expression:</b><br/>' + params.data.full_expr + '<br/><br/>' +
                                   '<b>Metrics:</b><br/>' + params.data.metrics;
                        }}
                    }}
                }},
                color: ['#00e5ff', '#ff3366'],
                legend: [{{
                    data: ['DL Roots (Components)', 'Evolved Factors'],
                    bottom: 20,
                    textStyle: {{ color: '#fff' }}
                }}],
                series: [{{
                    type: 'graph',
                    layout: 'force',
                    nodes: graphNodes,
                    edges: graphLinks,
                    categories: [
                        {{ name: 'DL Roots (Components)' }},
                        {{ name: 'Evolved Factors' }}
                    ],
                    roam: true,
                    label: {{
                        show: true,
                        position: 'right',
                        formatter: '{{b}}',
                        fontSize: 11,
                        color: '#eee'
                    }},
                    force: {{
                        repulsion: 300,
                        edgeLength: 80,
                        gravity: 0.1
                    }},
                    lineStyle: {{
                        color: 'source',
                        curveness: 0.2,
                        opacity: 0.6,
                        width: 1.5
                    }}
                }}]
            }};

            myChart.setOption(option);
        </script>
    </body>
    </html>
    """

    try:
        with open(OUTPUT_HTML_VIS, 'w', encoding="utf-8") as f:
            f.write(html_content)
        print(f"📈 深度学习 Root 血缘分析大屏已生成！请在浏览器中打开: {OUTPUT_HTML_VIS.name}")
    except Exception as e:
        print(f"❌ HTML 渲染失败: {e}")

if __name__ == "__main__":
    main()