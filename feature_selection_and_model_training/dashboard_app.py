import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path

# ==========================================
# ⚙️ 全局配置与动态路径
# ==========================================
st.set_page_config(page_title="AlphaMiner Pro Dashboard", layout="wide")

# 强制使用动态绝对路径读取大文件（完全绕过浏览器上传限制）
project_root = Path(__file__).resolve().parent.parent
DEFAULT_JSON_PATH = str(project_root / "data" / "feature_quality_dashboard_2025.json")

# 用于区分品种属于哪个池子的通用常量集合 (已脱敏为标准代码)
CFFEX_INSTS = {'IC', 'IF', 'IH', 'IM'}

def fmt_pct(val):
    return "N/A" if val is None else f"{val * 100:.2f}%"

def fmt_num(val):
    return "N/A" if val is None else f"{val:.3f}"

def fmt_bps(val):
    return "N/A" if val is None else f"{val:.1f} bps"

# ==========================================
# 💾 数据加载核心 (使用缓存，只读一次)
# ==========================================
@st.cache_data(show_spinner="Loading massive JSON into Server Memory (this may take a few minutes)...")
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 🚨 ===== 清洗意外混入的 Target 特征 ===== 🚨
    # 找出所有以 "Ret_" 开头的特征（包括 Ret_5m_Fwd, Ret_60m_Fwd 等）
    keys_to_remove = [k for k in data.keys() if k.startswith("Ret_")]
    
    # 遍历并安全删除这些脏数据
    for k in keys_to_remove:
        data.pop(k, None) 
        
    return data

# ==========================================
# 🎨 核心 UI 组件 (Feature Centric)
# ==========================================
def render_feature_centric(data, feature, target):
    st.title(f"Feature: {feature}")
    
    # --- Pool Metrics ---
    pools_data = data[feature].get("Pools", {})
    available_pools = sorted(pools_data.keys())
    
    if not available_pools:
        st.warning("No Pool Data Available")
        return
        
    pool_col1, pool_col2 = st.columns([1, 4])
    with pool_col1:
        active_pool = st.selectbox("Pool Metrics:", available_pools, index=available_pools.index("COMMODITY") if "COMMODITY" in available_pools else 0)
    
    target_data = pools_data[active_pool].get(target, {})
    
    # KPIs (完全替换为 Period IC)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Full Year IC", fmt_pct(target_data.get('full_ic')))
    kpi2.metric("Period Mean IC (2000m)", fmt_pct(target_data.get('mean_period_ic')))
    kpi3.metric("Q1 IR (Short)", fmt_num(target_data.get('q_first_ir')))
    kpi4.metric("Q10 IR (Long)", fmt_num(target_data.get('q_last_ir')))
    
    st.divider()
    
    # --- Cross-Instrument Stability ---
    st.subheader("Cross-Instrument Stability")
    cross_metric = st.radio("Select Metric:", ["mean_period_ic", "full_ic"], horizontal=True)
    
    insts_obj = data[feature].get("Instruments", {})
    inst_names, inst_vals, inst_colors = [], [], []
    
    for inst, inst_d in sorted(insts_obj.items()):
        val = inst_d.get(target, {}).get(cross_metric)
        if val is not None:
            inst_names.append(inst)
            inst_vals.append(val)
            inst_colors.append('rgba(34, 197, 94, 0.7)' if val > 0 else 'rgba(239, 68, 68, 0.7)')
            
    if inst_names:
        fig_cross = go.Figure(go.Bar(x=inst_names, y=inst_vals, marker_color=inst_colors))
        fig_cross.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300, xaxis_tickangle=-45)
        st.plotly_chart(fig_cross, use_container_width=True)
    else:
        st.info("No Instrument Data for Cross Stability")
        
    st.divider()
    
    # --- Sub-Scope Drilldown ---
    st.subheader("Sub-Scope Drilldown")
    
    scopes = []
    for p in pools_data.keys(): scopes.append(f"Pools|{p}")
    for i in insts_obj.keys(): scopes.append(f"Instruments|{i}")
    
    if not scopes:
        return
        
    drill_scope = st.selectbox("Select Drilldown Scope:", sorted(scopes))
    level, group = drill_scope.split('|')
    
    # -- 1. Backtest Matrix (Only for Instruments) --
    bt_data = data[feature].get(level, {}).get(group, {}).get("Backtest", {})
    if level == 'Pools' or not bt_data:
        st.info("ℹ️ Time-Series Backtest Table is only available at the Instrument level.")
    else:
        st.markdown("##### Backtest Time-Series Simulation ( Sharpe | Return )")
        bt_modes = [
            {'key': 'z_1m_cont_nofee', 'name': 'Z-Score 1m Continuous (Gross)'},
            {'key': 'z_1m_cont_fee', 'name': 'Z-Score 1m Continuous (Net)'},
            {'key': 'z_1m_intra_nofee', 'name': 'Z-Score 1m Intraday (Gross)'},
            {'key': 'z_30m_intra_nofee', 'name': 'Z-Score 30m Intraday (Gross)'},
            {'key': 'z_30m_intra_fee', 'name': 'Z-Score 30m Intraday (Net)'},
            {'key': 'q_30m_intra_nofee', 'name': 'Q-Extreme 30m Intraday (Gross)'},
            {'key': 'q_30m_intra_fee', 'name': 'Q-Extreme 30m Intraday (Net)'},
            {'key': 'rob_30m_intra_nofee', 'name': 'Robust-Z 30m Intraday (Gross)'},
            {'key': 'rob_30m_intra_fee', 'name': 'Robust-Z 30m Intraday (Net)'}
        ]
        
        # 组装 HTML 表格
        html_table = "<table style='width:100%; text-align:center; border-collapse:collapse;'>"
        html_table += "<tr style='background-color:#f8fafc; border-bottom:1px solid #e2e8f0;'><th style='padding:10px; text-align:left;'>Trading Mode</th><th>Window = 2000</th><th>Window = 4000</th><th>Window = 6000</th></tr>"
        
        for mode in bt_modes:
            html_table += f"<tr style='border-bottom:1px solid #f1f5f9;'><td style='padding:10px; text-align:left; font-weight:bold;'>{mode['name']}</td>"
            for w in [2000, 4000, 6000]:
                shp = bt_data.get(f"w{w}_{mode['key']}_shp")
                ret = bt_data.get(f"w{w}_{mode['key']}_ret")
                
                if shp is None or ret is None:
                    html_table += "<td style='color:#cbd5e1;'>--</td>"
                else:
                    shp_color = "#9333ea" if shp > 0 else "#94a3b8"
                    ret_color = "#10b981" if ret > 0 else "#ef4444"
                    html_table += f"<td><span style='color:{shp_color}; font-weight:900;'>{shp:.3f}</span> <span style='color:#cbd5e1; margin:0 5px;'>|</span> <span style='color:{ret_color}; font-weight:bold;'>{ret:.1f} bps</span></td>"
            html_table += "</tr>"
        html_table += "</table><br>"
        st.markdown(html_table, unsafe_allow_html=True)
        
    # -- 2. Distribution & Decile --
    drill_tgt_data = data[feature].get(level, {}).get(group, {}).get(target, {})
    if drill_tgt_data:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("##### Value Distribution (H1 vs H2)")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(x=drill_tgt_data.get('hist_x', []), y=drill_tgt_data.get('hist_h1', []), name='H1', marker_color='rgba(96, 165, 250, 0.8)'))
            fig_dist.add_trace(go.Bar(x=drill_tgt_data.get('hist_x', []), y=drill_tgt_data.get('hist_h2', []), name='H2', marker_color='rgba(251, 191, 36, 0.8)'))
            fig_dist.update_layout(barmode='stack', margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with chart_col2:
            st.markdown("##### Decile Mean Returns")
            bins = [f"Q{i+1}" for i in range(10)]
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Bar(x=bins, y=drill_tgt_data.get('decile_all', []), name='All', marker_color='#e2e8f0'))
            fig_ret.add_trace(go.Scatter(x=bins, y=drill_tgt_data.get('decile_h1', []), name='H1', mode='lines+markers', line=dict(color='#3b82f6', shape='spline')))
            fig_ret.add_trace(go.Scatter(x=bins, y=drill_tgt_data.get('decile_h2', []), name='H2', mode='lines+markers', line=dict(color='#f59e0b', shape='spline')))
            fig_ret.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig_ret, use_container_width=True)

# ==========================================
# 📊 横向分析 (Scope Centric) - 加入强大的过滤引擎
# ==========================================
def render_scope_centric(data, scope_obj, target):
    level, group = scope_obj.split('|')
    st.title(f"{level[:-1]}: {group}")
    
    # 动态生成所有指标选项 (彻底替换为 period_ic)
    metrics_dict = {
        "mean_period_ic": "IC: Mean Period IC (Abs Rank)",
        "full_ic": "IC: Full Year IC (Abs Rank)",
        "q_last_ir": "IR: Q10 IR (Abs Rank)",
        "q_first_ir": "IR: Q1 IR (Abs Rank)"
    }
    
    modes = [
        ('z_1m_cont_nofee', 'Z 1m Cont (Gross)'), ('z_1m_cont_fee', 'Z 1m Cont (Net)'),
        ('z_1m_intra_nofee', 'Z 1m Intra (Gross)'), ('z_30m_intra_nofee', 'Z 30m Intra (Gross)'),
        ('z_30m_intra_fee', 'Z 30m Intra (Net)'), ('q_30m_intra_nofee', 'Q 30m Intra (Gross)'),
        ('q_30m_intra_fee', 'Q 30m Intra (Net)'), ('rob_30m_intra_nofee', 'Robust 30m Intra (Gross)'),
        ('rob_30m_intra_fee', 'Robust 30m Intra (Net)')
    ]
    for w in [2000, 4000, 6000]:
        for m_key, m_name in modes:
            metrics_dict[f"w{w}_{m_key}_shp"] = f"BT Sharpe: w{w} {m_name}"
            metrics_dict[f"w{w}_{m_key}_ret"] = f"BT Return: w{w} {m_name} (bps)"
            
    metric_keys = list(metrics_dict.keys())
    metric_labels = list(metrics_dict.values())
    
    selected_label = st.selectbox("Select Metric to Rank:", metric_labels)
    metric = metric_keys[metric_labels.index(selected_label)]
    
    is_bt_metric = '_shp' in metric or '_ret' in metric
    use_abs_rank = metric in ['mean_period_ic', 'full_ic', 'q_first_ir', 'q_last_ir']
    
    # 🎯 === 核心更新：特征过滤模块 (Threshold Filter) ===
    st.markdown("---")
    st.markdown("#### 🎯 Filter Features by Threshold")
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_mode = st.selectbox(
            "Filter Condition:", 
            ["No Filter", "Raw Value > Threshold", "Raw Value < Threshold", "Absolute Value |X| > Threshold"]
        )
    with col_f2:
        if filter_mode != "No Filter":
            # 根据是否是收益率/IC动态调整默认步长
            default_step = 0.005 if is_bt_metric and '_ret' not in metric else 0.05
            threshold = st.number_input("Set Threshold value (X):", value=0.0, step=default_step, format="%.4f")
        else:
            threshold = 0.0
    st.markdown("---")
    
    results = []
    for feat in data.keys():
        try:
            val = None
            if is_bt_metric:
                if level == 'Pools':
                    # 混池中位数计算 (基于通用代码判断股指)
                    vals_for_median = []
                    insts_dict = data[feat].get("Instruments", {})
                    for inst, inst_data in insts_dict.items():
                        is_cffex = inst in CFFEX_INSTS
                        if (group == 'CFFEX' and is_cffex) or (group == 'COMMODITY' and not is_cffex):
                            v = inst_data.get("Backtest", {}).get(metric)
                            if v is not None and float(v) != 0.0:  # 剔除无效交易品种
                                vals_for_median.append(v)
                    if vals_for_median:
                        val = float(np.median(vals_for_median))
                else:
                    val = data[feat].get(level, {}).get(group, {}).get("Backtest", {}).get(metric)
            else:
                val = data[feat].get(level, {}).get(group, {}).get(target, {}).get(metric)
                
            if val is not None and float(val) != 0.0:
                # 🚀 执行阈值过滤逻辑
                pass_filter = True
                if filter_mode == "Raw Value > Threshold" and val <= threshold:
                    pass_filter = False
                elif filter_mode == "Raw Value < Threshold" and val >= threshold:
                    pass_filter = False
                elif filter_mode == "Absolute Value |X| > Threshold" and abs(val) <= threshold:
                    pass_filter = False
                
                # 如果通过了筛选，才加入结果列表
                if pass_filter:
                    rank_score = abs(val) if use_abs_rank else val
                    results.append({"feature": feat, "value": val, "score": rank_score})
        except Exception:
            pass
            
    if not results:
        st.info(f"No features meet the criteria for metric: {selected_label}")
        return
        
    # 动态播报统计结果
    if filter_mode != "No Filter":
        cond_str = filter_mode.replace("Threshold", str(threshold))
        st.success(f"✅ Found **{len(results)}** features meeting the condition: **{cond_str}**")
    else:
        st.caption(f"Total valid features evaluated: {len(results)}")
        
    # 排序并取 Top 20
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top20 = results[:20][::-1]
    
    dist_title = "##### Metric Distribution (Pool Median)" if (is_bt_metric and level == 'Pools') else "##### Metric Distribution"
    top_title = "##### Top 20 Leaderboard (Pool Median)" if (is_bt_metric and level == 'Pools') else "##### Top 20 Leaderboard"
    
    col1, col2 = st.columns([1, 2])
    
    # 绘制分布图 (只会绘制满足过滤条件的分布)
    with col1:
        st.markdown(dist_title)
        raw_vals = [r["value"] for r in results]
        fig_dist = go.Figure(go.Histogram(x=raw_vals, nbinsx=50, marker_color='rgba(16, 185, 129, 0.6)'))
        fig_dist.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=550)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    # 绘制排行榜
    with col2:
        st.markdown(top_title)
        is_pct = metric in ['mean_period_ic', 'full_ic']
        is_bps = '_ret' in metric
        
        def fmt(v):
            if is_pct: return f"{v*100:.2f}%"
            if is_bps: return f"{v:.1f} bps"
            return f"{v:.3f}"
            
        y_full_names = [r["feature"] for r in top20]
        y_display_labels = [f[:30] + '..' if len(f) > 30 else f for f in y_full_names]
        
        x_vals = [r["value"] for r in top20]
        texts = [fmt(v) for v in x_vals]
        colors = ['rgba(249, 115, 22, 0.8)' if v > 0 else 'rgba(14, 165, 233, 0.8)' for v in x_vals]
        
        fig_top = go.Figure(go.Bar(
            x=x_vals, y=y_full_names, orientation='h', text=texts, textposition='outside', marker_color=colors
        ))
        
        x_max = max([abs(v) for v in x_vals] + [0.0001]) * 1.25
        x_range = None if (top20[0]["value"] < 0 and not use_abs_rank) else [-x_max, x_max]
        
        fig_top.update_layout(
            margin=dict(l=10, r=40, t=20, b=20), 
            height=550, 
            xaxis=dict(range=x_range),
            yaxis=dict(
                tickmode='array',
                tickvals=y_full_names,
                ticktext=y_display_labels
            )
        )
        st.plotly_chart(fig_top, use_container_width=True)

# ==========================================
# 🚀 主程序流程
# ==========================================
def main():
    st.sidebar.title("🎛️ Controls")
    
    # 默认隐藏复杂的 JSON 路径输入框，让操作更“傻瓜”
    use_custom_path = st.sidebar.checkbox("Use Custom JSON Path", value=False)
    if use_custom_path:
        json_path = st.sidebar.text_input("JSON File Path:", DEFAULT_JSON_PATH)
    else:
        json_path = DEFAULT_JSON_PATH
        st.sidebar.info(f"Using default path: {json_path}")
    
    try:
        data = load_data(json_path)
    except Exception as e:
        st.error(f"Failed to load JSON file. Error: {e}")
        return
        
    features = sorted(list(data.keys()))
    scopes = set()
    for feat in features[:10]:
        pools = data[feat].get("Pools", {}).keys()
        insts = data[feat].get("Instruments", {}).keys()
        for p in pools: scopes.add(f"Pools|{p}")
        for i in insts: scopes.add(f"Instruments|{i}")
    scopes = sorted(list(scopes))
    
    col_mode, col_tgt, _ = st.columns([2, 2, 6])
    with col_mode:
        mode = st.radio("View Mode", ["Feature Centric", "Scope Centric"], horizontal=True)
    with col_tgt:
        target = st.selectbox("Global Target", ["Ret_15m_Fwd", "Ret_30m_Fwd"])
        
    st.sidebar.divider()
    search_query = st.sidebar.text_input("🔍 Search...")
    
    if mode == "Feature Centric":
        filtered_items = [f for f in features if search_query.lower() in f.lower()]
        if not filtered_items:
            st.warning("No features found.")
            return
        active_feature = st.sidebar.radio("Select Feature", filtered_items)
        render_feature_centric(data, active_feature, target)
        
    else:
        filtered_items = [s for s in scopes if search_query.lower() in s.lower()]
        if not filtered_items:
            st.warning("No scopes found.")
            return
        active_scope = st.sidebar.radio("Select Scope", filtered_items)
        render_scope_centric(data, active_scope, target)

if __name__ == "__main__":
    main()