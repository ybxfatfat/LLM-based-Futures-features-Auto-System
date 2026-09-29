# update_to_panel.py
import pandas as pd
import numpy as np
from pathlib import Path
import os
import pyarrow.dataset as ds  # 💥 新增：处理分布式文件的工业级库

Year = 2024

def update_base_panel_with_features(base_panel_dir: Path, features_file_path: Path, target_instruments: list, approved_features: list):
    """
    将高频特征宽表完美拼接入按品种存储的 base_panel 中。
    使用 pathlib 保证路径稳定，使用 approved_features 白名单严格控制入库字段。
    """
    print(f"🚀 开始合并特征到 Base Panel...")
    print(f"📖 读取高频特征文件(夹): {features_file_path}")
    
    if not features_file_path.exists():
        print(f"❌ 错误: 找不到特征路径: {features_file_path}")
        return

    # 💥 架构级修复：使用 PyArrow Dataset 健壮地读取分布式落盘的碎片文件夹
    # 它会自动忽略非 parquet 文件，并统一对齐 Schema
    try:
        dataset = ds.dataset(features_file_path, format="parquet")
        df_features = dataset.to_table().to_pandas()
    except Exception as e:
        print(f"❌ 错误: 无法解析 Parquet 目录: {str(e)}")
        return
    
    # 确保用于对齐的键的数据类型绝对一致
    df_features['index_time'] = pd.to_datetime(df_features['index_time'])
    df_features['contract'] = df_features['contract'].astype(str)
    
    # 2. 严格执行白名单过滤！
    join_keys = ['index_time', 'instrument', 'contract']
    
    feature_cols = [col for col in approved_features if col in df_features.columns]
    missing_cols = [col for col in approved_features if col not in df_features.columns]
    
    if missing_cols:
        print(f"⚠️ 警告: 以下审核特征在宽表中未找到，将被忽略: {missing_cols}")
        
    if not feature_cols:
        print("❌ 错误: 白名单中没有任何特征在宽表中存在！停止合并。")
        return
        
    print(f"🧩 审核通过，即将注入 {len(feature_cols)} 个特征列: {feature_cols}")

    # 3. 按品种遍历 Base Panel
    for inst in target_instruments:
        print(f"\n{'='*50}")
        print(f"🔄 正在处理品种: {inst}")
        print(f"{'='*50}")
        
        # 使用 pathlib 优雅拼接路径
        panel_path = base_panel_dir / inst / f"base_panel_{inst}_parquet.parquet"
        
        if not panel_path.exists():
            print(f"❌ 警告: 找不到 {inst} 的 base_panel 文件: {panel_path}")
            continue
            
        print(f"📥 正在读取原始 Base Panel (来源: {panel_path.parent.name}/)...")
        df_base = pd.read_parquet(panel_path)
        
        # 数据类型强转对齐
        df_base['index_time'] = pd.to_datetime(df_base['index_time'])
        df_base['contract'] = df_base['contract'].astype(str)
        
        # 4. 从全量特征表中切分出当前品种的特征
        df_inst_features = df_features[df_features['instrument'] == inst].copy()
        
        # ⚠️ 【极其关键】：只切出 join_keys 和 审核通过的 feature_cols！
        cols_to_keep = join_keys + feature_cols
        df_inst_features = df_inst_features[cols_to_keep]
        
        if df_inst_features.empty:
            print(f"⚠️ 警告: 高频特征表中没有找到 {inst} 的数据。跳过合并。")
            continue
            
        print(f"   -> 提取到 {len(df_inst_features)} 行有效特征。")
        print("🔗 正在执行智能 Left Join 拼接 (保留现有跨年特征)...")
        
        # 不再 drop 现有列！而是用 suffixes 区分老列和新列
        df_merged = pd.merge(
            df_base,
            df_inst_features,
            on=['index_time', 'contract'],
            how='left',
            suffixes=('', '_new')
        )
        
        # 智能打补丁 (Patch)
        for col in feature_cols:
            col_new = f"{col}_new"
            if col_new in df_merged.columns:
                if col in df_merged.columns:
                    mask = df_merged[col_new].notna()
                    df_merged.loc[mask, col] = df_merged.loc[mask, col_new]
                else:
                    df_merged[col] = df_merged[col_new]
                
                df_merged = df_merged.drop(columns=[col_new])
        
        # 3. 排序保证时序绝对正确
        df_merged = df_merged.sort_values(['index_time', 'contract']).reset_index(drop=True)
        
        print("💾 正在回写 Parquet 文件...")
        
        save_path = base_panel_dir / inst / f"base_panel_{inst}_parquet.parquet"
        df_merged.to_parquet(save_path)
        print(f"✅ {inst} 拼接完毕！文件已保存至: {save_path.name}")

if __name__ == "__main__":
    current_script_path = Path(__file__).resolve()
    
    # 💥 脱敏修复：假定这是一个标准的开源项目目录结构
    base_dir = current_script_path.parent.parent
    base_panel_dir = base_dir / "data" / "intraday_parquet" / "base_panel"
    
    # =========================================================
    # 🎯 全市场自由开关与品种池配置
    # =========================================================
    RUN_UPDATE_MODE = ['cffex', 'commodity']
    
    CFFEX_INSTRUMENTS = ['IM', 'IC', 'IF', 'IH']
    COMMODITY_INSTRUMENTS = [
        'AG', 'AL', 'AO', 'AU', 'BR', 'BU', 'CU', 'FU', 'HC', 'NI', 'PB', 'RB', 'RU', 'SN', 'SP', 'SS', 'WR', 'ZN',
        'A', 'B', 'BB', 'C', 'CS', 'EB', 'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V', 'Y',
        'AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'PF', 'PK', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR',
        'BC', 'EC', 'LU', 'NR', 'SC',
        'LC'
    ]

    TARGET_INSTRUMENTS = []
    if 'cffex' in RUN_UPDATE_MODE:
        TARGET_INSTRUMENTS.extend(CFFEX_INSTRUMENTS)
    if 'commodity' in RUN_UPDATE_MODE:
        TARGET_INSTRUMENTS.extend(COMMODITY_INSTRUMENTS)
        
    print(f"🎯 将要拼接的品种总数: {len(TARGET_INSTRUMENTS)}")

    # =========================================================
    # 将绝对路径改为通用相对路径
    # =========================================================
    features_file_path = base_dir / "data" / "features_output" / str(Year)
    
    # =========================================================
    # 📋 审核通过的新特征白名单
    # =========================================================
    APPROVED_FEATURES = [
        'BuyRatio', 'MicroPriceBasis', 'MaxConsecUpTicks', 'MaxConsecUpAmp', 'MaxConsecSellCount', 'MaxConsecSellVolume',
        'VolIncrementCV', 'InfoToxicity', 'ExpOBI', 'SpreadCV','MaxConsecDownTicks','MaxConsecDownAmp',
        'MaxConsecBuyCount', 'MaxConsecBuyVolume',
        'ask1_qty', 'bid1_qty', 'open_interest',
        'ask2_qty', 'bid2_qty', 'ask3_qty', 'bid3_qty', 'ask4_qty', 'bid4_qty', 'ask5_qty', 'bid5_qty',
        'ask2', 'bid2', 'ask3', 'bid3', 'ask4', 'bid4', 'ask5', 'bid5'
    ]
    
    update_base_panel_with_features(base_panel_dir, features_file_path, TARGET_INSTRUMENTS, APPROVED_FEATURES)