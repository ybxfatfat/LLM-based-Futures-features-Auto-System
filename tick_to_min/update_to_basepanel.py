# update_to_panel.py
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

Year = 2024


def _load_feature_batches_wide(features_path: Path, join_keys: list[str], approved_features: list[str]) -> pd.DataFrame:
    """因子批次是按列拆分的：每个文件共享同一组行键，但携带不同的因子列。"""
    if features_path.is_file():
        parquet_files = [features_path]
    else:
        parquet_files = sorted(features_path.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"路径下没有 Parquet 文件: {features_path}")

    feature_series: dict[str, pd.Series] = {}
    for parquet_file in parquet_files:
        schema_cols = set(pq.read_schema(parquet_file).names)
        missing_keys = [key for key in join_keys if key not in schema_cols]
        if missing_keys:
            raise ValueError(f"{parquet_file.name} 缺少行键: {missing_keys}")

        batch_features = [col for col in approved_features if col in schema_cols]
        if not batch_features:
            continue

        batch = pd.read_parquet(parquet_file, columns=join_keys + batch_features)
        batch["index_time"] = pd.to_datetime(batch["index_time"])
        batch["instrument"] = batch["instrument"].astype(str)
        batch["contract"] = batch["contract"].astype(str)

        duplicated = batch.duplicated(join_keys, keep=False)
        if duplicated.any():
            sample = batch.loc[duplicated, join_keys].head(3).to_dict("records")
            raise ValueError(f"{parquet_file.name} 内部存在重复行键，示例: {sample}")

        batch = batch.set_index(join_keys)
        for col in batch_features:
            new_values = batch[col]
            if col in feature_series:
                # 后读入批次的非空值更新先前批次，空值不覆盖已有值。
                new_values = new_values.combine_first(feature_series[col])
            feature_series[col] = new_values

    if not feature_series:
        return pd.DataFrame(columns=join_keys)

    return pd.concat(feature_series, axis=1).reset_index()

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

    join_keys = ['index_time', 'instrument', 'contract']

    # 因子批次是横向分列，必须逐文件按三键对齐，不能当作 Dataset 纵向堆叠。
    try:
        df_features = _load_feature_batches_wide(
            features_file_path, join_keys, approved_features
        )
    except Exception as e:
        print(f"❌ 错误: 无法解析 Parquet 目录: {str(e)}")
        return

    # 2. 严格执行白名单过滤！
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
        df_base['instrument'] = df_base['instrument'].astype(str)
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
        print("🔗 正在按 index_time/instrument/contract 一对一增量更新...")

        if df_base.duplicated(join_keys).any():
            raise ValueError(f"{panel_path.name} 本身存在重复行键，为避免扩行已停止写入")
        if df_inst_features.duplicated(join_keys).any():
            raise ValueError(f"{inst} 的特征宽表存在重复行键，为避免扩行已停止写入")

        original_columns = list(df_base.columns)
        df_merged = df_base.set_index(join_keys, verify_integrity=True)
        incoming = df_inst_features.set_index(join_keys, verify_integrity=True).reindex(df_merged.index)

        # 新的非空值覆盖同键旧值；新数据缺失时保留已存储的历史特征。
        for col in feature_cols:
            if col in df_merged.columns:
                df_merged[col] = incoming[col].combine_first(df_merged[col])
            else:
                df_merged[col] = incoming[col]

        df_merged = df_merged.reset_index()
        new_columns = [col for col in feature_cols if col not in original_columns]
        df_merged = df_merged[original_columns + new_columns]

        # 3. 排序保证时序绝对正确
        df_merged = df_merged.sort_values(['index_time', 'contract']).reset_index(drop=True)
        
        print("💾 正在回写 Parquet 文件...")
        
        save_path = base_panel_dir / inst / f"base_panel_{inst}_parquet.parquet"
        temp_path = save_path.with_name(f"{save_path.name}.tmp")
        df_merged.to_parquet(temp_path, index=False)
        temp_path.replace(save_path)
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
