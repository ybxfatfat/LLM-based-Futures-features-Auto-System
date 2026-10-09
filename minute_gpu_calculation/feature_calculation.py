import os
import torch
import pandas as pd
from pathlib import Path
import time
import numpy as np
import math
import pyarrow as pa
import pyarrow.parquet as pq
import re
import datetime

from .tensor_context import GlobalTensorContext
from .dag_scheduler import DAGManager
from .batch_executor import BatchExecutor

HOLDING_HORIZON = 30
Year = 2024

# ======================================================================
# 零、 品种宇宙过滤器 (Universe Filter) - 拒绝国债与残缺品种污染
# ======================================================================
def load_target_instruments():
    """
    获取合法品种代码池。严格按照指定的白名单集合。
    """
    CFFEX_INSTRUMENTS = ['IM', 'IC', 'IF', 'IH']
    COMMODITY_INSTRUMENTS = [
        'AG', 'AL', 'AO', 'AU', 'BR', 'BU', 'CU', 'FU', 'HC', 'NI', 'PB', 'RB', 'RU', 'SN', 'SP', 'SS', 'WR', 'ZN',
        'A', 'B', 'BB', 'C', 'CS', 'EB', 'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V', 'Y',
        'AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'PF', 'PK', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR',
        'BC', 'EC', 'LU', 'NR', 'SC',
        'LC'
    ]
    
    EXCLUDED_INSTRUMENTS = {
        'LC', 'EC', 'AO',   
        'BB', 'FB', 'WR',   
        'RS',               
        'BC', 'LU'          
    }
    
    all_targets = set(CFFEX_INSTRUMENTS + COMMODITY_INSTRUMENTS)
    final_valid_insts = [inst for inst in all_targets if inst not in EXCLUDED_INSTRUMENTS]
    
    print(f"🌍 宇宙过滤器执行完毕！共保留 {len(final_valid_insts)} 个优质品种。")
    return final_valid_insts

# ======================================================================
# 第一步：研究员配置区 (Factor Configuration)
# ======================================================================
from miner.sample_features import FACTOR_FORMULAS as EVOLVED_FORMULAS

FACTOR_FORMULAS = {
    **EVOLVED_FORMULAS,
    'Rolling_Directional_Smoothness_Ratio_15M': 'Sign(TsMean((close - TsMin(close, 15)) / (TsMin(close, 15) + 1e-8), 15) - TsMean((TsMax(close, 15) - close) / (TsMax(close, 15) + 1e-8), 15)) / (IfElse(TsMean((close - TsMin(close, 15)) / (TsMin(close, 15) + 1e-8), 15) - TsMean((TsMax(close, 15) - close) / (TsMax(close, 15) + 1e-8), 15), TsMean((TsMax(close, 15) - close) / (TsMax(close, 15) + 1e-8), 15), TsMean((close - TsMin(close, 15)) / (TsMin(close, 15) + 1e-8), 15)) + 1e-6)',
    'VolAdjMom_10': 'TsPct(close, 10) / (TsStd(TsPct(close, 1), 10) + 1e-8)',
    'VolAdjMom_15': 'TsPct(close, 15) / (TsStd(TsPct(close, 1), 15) + 1e-8)',
    'VolAdjMom_30': 'TsPct(close, 30) / (TsStd(TsPct(close, 1), 30) + 1e-8)',
    'VolAdjMom_60': 'TsPct(close, 60) / (TsStd(TsPct(close, 1), 60) + 1e-8)',
}

import json

MARKET_STATE_REGISTRY = Path(__file__).resolve().parent.parent / "data" / "market_states_registry_index.json"
if os.path.exists(MARKET_STATE_REGISTRY):
    with open(MARKET_STATE_REGISTRY, 'r') as f:
        ms_registry = json.load(f)

    for feature_name, feature_meta in ms_registry.items():
        formula = feature_meta.get('formula', None)
        if formula and isinstance(formula, str) and feature_name not in FACTOR_FORMULAS:
            FACTOR_FORMULAS[feature_name] = formula
            print(f"  ✅ [MarketState] 注入: {feature_name}")

print(f"🧪 总因子数: {len(FACTOR_FORMULAS)}")

LABEL_FORMULAS = {
    'Ret_1m_Fwd': 'TsForwardReturn(close, 1)',
    'Ret_5m_Fwd': 'TsForwardReturn(close, 5)',
    'Ret_15m_Fwd': 'TsForwardReturn(close, 15)',
    'Ret_30m_Fwd': 'TsForwardReturn(close, 30)',
    'Ret_60m_Fwd': 'TsForwardReturn(close, 60)'
}

# ======================================================================
# 全市场压测配置 (Stress Test Configurations)
# ======================================================================
TARGET_START = f'{Year}-01-01'
TARGET_END = f'{Year}-12-31'  
WARM_UP_DAYS = 5          
MODE = 'all_contracts'
FORMULA_BATCH_SIZE = 50   # 🚀 架构升级：按公式切块，彻底消灭 PCIe 瓶颈

def run_factor_pipeline():
    print("="*65)
    print(f"🚀 启动全市场因子生产流水线 (公式切片+Base常驻显存架构, 批大小: {FORMULA_BATCH_SIZE}) 🚀")
    print("="*65)
    
    base_dir = Path(__file__).resolve().parent.parent
    base_panel_dir = base_dir / "data" / "intraday_parquet" / "base_panel"
    out_dir = base_dir / "data" / f"factors_{Year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_insts = set(load_target_instruments())
    panel_paths = [p for p in base_panel_dir.rglob("base_panel_*.parquet") if p.stem.split('_')[2] in target_insts]
    if not panel_paths:
        raise FileNotFoundError("找不到合法的 Base Panel 数据文件！")

    from .utils import calendar_db as simdb
    cal_series = pd.to_datetime(simdb.calendar['current_business_date']).dt.date.unique()
    cal_series = np.sort(cal_series)
    
    start_date = pd.to_datetime(TARGET_START).date()
    end_date = pd.to_datetime(TARGET_END).date()
    
    start_idx = np.searchsorted(cal_series, start_date)
    actual_start_idx = max(0, start_idx - WARM_UP_DAYS)
    warm_up_start_date = cal_series[actual_start_idx]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    print(f"\n📥 正在将全市场全年 Base Panel 一次性加载至显存 (Pin Memory)...")
    print(f"   -> 数据区间: [{warm_up_start_date} -> {TARGET_END}]")
    
    # 1. 终极一击：整年 Base 数据一把梭进 GPU
    global_ctx = GlobalTensorContext(
        panel_paths, 
        target_start=str(warm_up_start_date), 
        target_end=TARGET_END, 
        warm_up_days=0,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    
    if torch.cuda.is_available():
        base_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"✅ Base Panel 驻留完成！当前占用显存: {base_vram:.2f} MB")

    # 2. 因子公式分批逻辑
    all_features = list(FACTOR_FORMULAS.items())
    num_batches = math.ceil(len(all_features) / FORMULA_BATCH_SIZE)
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * FORMULA_BATCH_SIZE
        batch_end = min((batch_idx + 1) * FORMULA_BATCH_SIZE, len(all_features))
        current_batch_features = dict(all_features[batch_start:batch_end])
        
        # 将 Label (预测目标) 合并到第一批计算中，避免冗余文件
        if batch_idx == 0:
            current_batch_features.update(LABEL_FORMULAS)

        print(f"\n{'='*65}")
        print(f"🔥 正在处理公式批次 {batch_idx+1}/{num_batches} (包含 {len(current_batch_features)} 个因子/标签)")
        
        # 3. 为当前批次构建专属计算图 (隔离环境)
        dag = DAGManager()
        target_nodes = {}
        for col_name, formula in current_batch_features.items():
            node_id = dag.add_formula(formula)
            target_nodes[col_name] = node_id

        # 4. 执行 GPU 并发计算
        executor = BatchExecutor(dag, global_ctx)
        executor.run(target_nodes_list=list(target_nodes.values()))
        
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"   -> [VRAM 压测] 批次 {batch_idx+1} 计算峰值显存: {peak_vram_mb:.2f} MB")

        # 5. 直接从显存提取 DataFrame 并砍掉 Warm-up 预热期
        df_batch = executor.extract_multiple_signals(
            target_nodes_dict=target_nodes,
            mode=MODE,
            target_start=TARGET_START,
            target_end=TARGET_END
        )
        
        # =========================================================
        # 🧹 6. 终极 GPU 显存回收 (Garbage Collection: 冲水逻辑)
        # =========================================================
        print("   -> 🧹 正在清理当前批次计算图显存，确立 Base Panel 护城河...")
        for inst_id, inst_ctx in global_ctx.inst_contexts.items():
            keys_to_delete = [k for k in inst_ctx.tensor_map.keys() if k.startswith('Node_')]
            for k in keys_to_delete:
                inst_ctx.delete(k)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 7. 数据后处理与独立写盘 (避免 Parquet 列追加造成的重写灾难)
        if not df_batch.empty:
            if not pd.api.types.is_datetime64_any_dtype(df_batch['index_time']):
                df_batch['index_time'] = pd.to_datetime(df_batch['index_time'])
            df_batch = df_batch.sort_values(['instrument', 'index_time']).reset_index(drop=True)

            # 夜盘交易日对齐
            df_batch['trading_date'] = df_batch['index_time'].dt.date
            is_night = df_batch['index_time'].dt.hour >= 18
            df_batch.loc[is_night, 'trading_date'] = \
                (df_batch.loc[is_night, 'index_time'] + pd.Timedelta(days=1)).dt.date

            # 清理全 NaN 行
            factor_cols = list(current_batch_features.keys())
            df_batch = df_batch.dropna(how='all', subset=factor_cols)

            # 尾部修剪 (仅针对当前批次存在的 Label)
            bar_times = df_batch['index_time'].dt.time
            end_time = datetime.time(15, 0, 0)
            for label_col in LABEL_FORMULAS.keys():
                if label_col in current_batch_features and label_col != 'Ret_1m_Fwd':
                    match = re.search(r'Ret_(\d+)m_Fwd', label_col)
                    if match:
                        horizon = int(match.group(1))
                        dummy_dt = datetime.datetime(2000, 1, 1, 15, 0, 0)
                        cutoff_dt = dummy_dt - datetime.timedelta(minutes=(horizon + 1))
                        cutoff_time = cutoff_dt.time()
                        tail_mask = (bar_times > cutoff_time) & (bar_times <= end_time)
                        df_batch.loc[tail_mask, label_col] = np.nan

            # 独立写盘
            out_file = out_dir / f"factor_dataset_AllMarket_{MODE}_{Year}_batch_{batch_idx+1:03d}.parquet"
            table = pa.Table.from_pandas(df_batch, preserve_index=False)
            pq.write_table(
                table, 
                out_file, 
                compression='snappy',
                use_dictionary=True,
                write_statistics=True
            )
            print(f"   ✅ 批次 {batch_idx+1} 数据已成功落盘至: {out_file.name}")
        
        del df_batch
        
    print(f"\n🎉 极其硬核：全年 {len(all_features)} 个因子全量流式分批生产完毕！")
    print(f"📁 分批数据存放于: {out_dir}/ (后续按 index_time/instrument/contract 横向对齐拼表)")

if __name__ == "__main__":
    run_factor_pipeline()
