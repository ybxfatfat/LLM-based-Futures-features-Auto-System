import pandas as pd
import numpy as np
import torch
import gc
from pathlib import Path

_original_read_parquet = pd.read_parquet

def _monkey_patched_read_parquet(*args, **kwargs):
    df = _original_read_parquet(*args, **kwargs)
    if isinstance(df, pd.DataFrame):
        # Core metadata columns we MUST keep for pipeline processing
        keep_metadata = {'contract', 'trading_date', 'main_contract'}

        # Select columns to keep:
        # - Core metadata columns
        # - Any numeric column (floats, ints, booleans)
        # - Any datetime/timestamp column (e.g., index_time)
        cols_to_keep = [
            col for col in df.columns
            if col in keep_metadata
            or pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_datetime64_any_dtype(df[col])
        ]

        if len(cols_to_keep) < len(df.columns):
            df = df[cols_to_keep]

    return df

class InstrumentContext:
    """
    单个品种的独立宇宙 (Instrument-Specific Universe)
    1. 维护紧凑时间轴 T_i，绝无假夜盘空洞。
    2. 引入生命周期掩码 (Lifecycle Mask)，绝无僵尸价格。
    """
    def __init__(self, instrument_id, panel_df, target_start, target_end, warm_up_days, device):
        self.instrument_id = instrument_id
        self.device = device
        self.context_dict = {}

        from .utils.calendar_db import get_date_shift
        real_fetch_start = get_date_shift(pd.to_datetime(target_start), -warm_up_days)

        # 过滤多余的废数据，只让需要的数据进入 GPU
        panel_df = panel_df[(panel_df['index_time'] >= real_fetch_start) & (panel_df['index_time'] <= pd.to_datetime(target_end))]
        
        print(f"[{self.instrument_id}] 正在构建致密张量宇宙...")
        
        # 1. 提取该品种独有的紧凑时间轴 (天然过滤掉了没有夜盘的假空洞)
        self.time_index = np.sort(panel_df['index_time'].unique())
        self.contracts = np.sort(panel_df['contract'].unique())

        # 注意：排除 MACRO_MAIN 的干扰，只从实体数据中提取
        real_df = panel_df[panel_df['contract'] != 'MACRO_MAIN']
        
        # 精准去重，获取 [trading_date -> main_contract] 的干净字典
        unique_main_df = real_df.drop_duplicates(subset=['trading_date'])[['trading_date', 'main_contract']].copy()
        unique_main_df.set_index('trading_date', inplace=True)
        
        # 存为字典格式的 Series，供 Late Masking 瞬间映射
        self.switch_cycle_map = unique_main_df['main_contract']
        
        # 2. 提取每个合约的“绝对生命周期”边界
        lifecycles = panel_df.groupby('contract')['index_time'].agg(['min', 'max'])
        
        # 3. 提取特征列
        exclude_cols = {'index_time', 'contract', 'trading_date'}
        feature_cols = [c for c in panel_df.columns if c not in exclude_cols]
        fill_zero_cols = {'volume', 'tick_count', 'tick_count_trd'}
        
        # 4. 局部的 Pivot
        pivot_df = panel_df.pivot(index='index_time', columns='contract', values=feature_cols)
        pivot_df = pivot_df.reindex(self.time_index)
        
        # ========================================================
        # 【核心逻辑】：生成生命周期掩码矩阵 (Lifecycle Mask)
        # ========================================================
        # 时间轴转列向量 (T_i, 1)
        t_matrix = self.time_index[:, None]  
        
        # 提取当前品种下所有合约的起止时间 (C_i,)
        starts = lifecycles.loc[self.contracts, 'min'].values  
        ends = lifecycles.loc[self.contracts, 'max'].values    
        
        # 广播生成布尔掩码 (T_i, C_i)
        is_alive_mask = (t_matrix >= starts) & (t_matrix <= ends)
        
        # ========================================================
        # 智能填补与掩码截断
        # ========================================================
        for feat in feature_cols:
            feat_df = pivot_df[feat]
            
            if feat in fill_zero_cols:
                # 盘中没成交填 0
                filled_mat = feat_df.fillna(0.0).values
            else:
                # 价格类盘中没成交，前向填充
                filled_mat = feat_df.ffill().values
                
            # 【终极必杀】：用掩码把生命周期外(未上市/已退市)的假数据“干掉”！
            final_mat = np.where(is_alive_mask, filled_mat, np.nan)
            
            # 推送至 GPU
            self.context_dict[feat] = torch.tensor(final_mat.astype(np.float32), device=self.device)

        print(f"[{self.instrument_id}] 张量构建完毕！形状 (T_i, C_i): {self.context_dict[feature_cols[0]].shape}")
        
        # 清理内存
        del panel_df, pivot_df
        gc.collect()

    def get(self, name): 
        if name not in self.context_dict:
            raise KeyError(f"[{self.instrument_id}] Context 中不存在底层特征: {name}")
        return self.context_dict[name]
        
    def set(self, name, tensor_val): 
        self.context_dict[name] = tensor_val

    # 在 InstrumentContext 类的末尾加上：
    def delete(self, name):
        """
        [Garbage Collection: 显存智能回收]
        安全地从显存字典中删除不再被依赖的中间特征，并释放引用。
        """
        if name in self.context_dict:
            del self.context_dict[name]


import concurrent.futures

class GlobalTensorContext:
    """
    全局数据上下文 (Global Tensor Context)
    极速并发版：支持多线程疯狂装载 Parquet，瞬间填满 GPU 显存。
    """
    def __init__(self, panel_paths, target_start, target_end, warm_up_days=10, device='cuda:0', max_workers=50):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.inst_contexts = {}
        self.global_time_index = None 

        self.target_start = target_start
        self.target_end = target_end
        self.warm_up_days = warm_up_days
        
        # 并发加载的核心
        self._load_all_concurrent(panel_paths, max_workers)

    def _worker_load_single_inst(self, path):
        """线程池内的原子工作函数：加载单个品种"""
        path_obj = Path(path)
        inst_id = path_obj.stem.split('_')[2] 
        
        df = _monkey_patched_read_parquet(path)
        
        # 在独立的线程里进行耗时的 CPU Pivot 和 GPU 拷贝
        ctx = InstrumentContext(
            inst_id, df, self.target_start, self.target_end, self.warm_up_days, self.device
        )
        return inst_id, ctx

    def _load_all_concurrent(self, panel_paths, max_workers):
        all_times = set()
        
        print(f"\n[Global Context] 开启 {max_workers} 个并发线程，向 {self.device} 疯狂灌注特征面板...")
        
        # 启用多线程池进行 I/O 与 Pivot 并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有加载任务
            future_to_path = {executor.submit(self._worker_load_single_inst, path): path for path in panel_paths}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_path):
                try:
                    inst_id, ctx = future.result()
                    self.inst_contexts[inst_id] = ctx
                    all_times.update(ctx.time_index)
                except Exception as exc:
                    path = future_to_path[future]
                    print(f"❌ 加载 {path} 时发生严重错误: {exc}")
                    raise exc

        # 统一生成全局时间轴
        self.global_time_index = np.sort(list(all_times))
        print(f"\n[Global Context] 💥 并发装载完成！共加载 {len(self.inst_contexts)} 个品种，全局时间轴长度: {len(self.global_time_index)}")
        
    def get_inst(self, inst_id):
        return self.inst_contexts[inst_id]