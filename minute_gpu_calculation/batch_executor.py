import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .ops import OP_REGISTRY  
from .tensor_context import GlobalTensorContext, InstrumentContext
from .dag_scheduler import DAGManager

class BatchExecutor:
    """
    GPU 批处理执行器 (终极双层并发版)
    严格遵循《六大绝对开发准则》。
    """
    def __init__(self, dag: DAGManager, global_context: GlobalTensorContext, max_concurrent_insts=4):
        self.dag = dag
        self.global_ctx = global_context
        # 【准则六-墙2：并发限流】限制同一时刻发射到 GPU 的品种数量
        self.max_concurrent_insts = max_concurrent_insts 
        
        # 【准则六-墙1：显存回收准备】计算所有节点的出度 (被多少个下游节点依赖)
        self.out_degrees = {n: 0 for n in self.dag.nodes}
        for node, deps in self.dag.dependencies.items():
            for dep in deps:
                if dep.startswith("Node_"):
                    self.out_degrees[dep] += 1

    def _compute_single_inst(self, inst_ctx: InstrumentContext, node_id: str, node_info: dict, physical_op):
        """
        【准则二：品种间并发】
        为单个品种分配独立的 CUDA Stream 进行异步计算。
        """
        stream = torch.cuda.Stream(device=self.global_ctx.device)
        with torch.cuda.stream(stream):
            # 1. 获取输入参数
            inputs = []
            for arg in node_info['args']:
                if isinstance(arg, str) and arg.startswith("Node_"):
                    # 【准则一：绝对物理隔离】从自己品种的独立宇宙中取数据
                    inputs.append(inst_ctx.get(arg))
                else:
                    inputs.append(arg) # 标量常数
                    
            # 2. 调用物理算子进行计算 (准则五：调用 batch_compute)
            result_tensor = physical_op.batch_compute(*inputs)
            
            # 3. 将 (T_i, C_i) 存回该品种的独立字典
            inst_ctx.set(node_id, result_tensor)

    def _execute_cross_sectional_node(self, node_id, node_info):
        """
        截面算子动态对齐桥 (Alignment Bridge)
        将局部时间轴 T_i 投影到 Global_T 上进行截面运算。
        """
        print(f"   [Alignment Bridge] 触发截面对齐: {node_info['op']}")
        op_name = node_info['op']
        arg_node = node_info['args'][0] 
        
        global_T = len(self.global_ctx.global_time_index)
        num_insts = len(self.global_ctx.inst_contexts)
        inst_keys = list(self.global_ctx.inst_contexts.keys())
        
        # 1. 构建全局张量 (填满 NaN)
        global_tensor = torch.full((global_T, num_insts), float('nan'), dtype=torch.float32, device=self.global_ctx.device)
        
        for i, inst_id in enumerate(inst_keys):
            inst_ctx = self.global_ctx.inst_contexts[inst_id]
            local_tensor = inst_ctx.get(arg_node)
            
            # 【准则三：平行宇宙法则 & 准则四：截面只算主力】
            # 截面计算强行提取 MACRO_MAIN 进行运算
            macro_idx = list(inst_ctx.contracts).index('MACRO_MAIN')
            macro_series = local_tensor[:, macro_idx]
            
            # 利用 np.searchsorted 极速投影时间戳
            global_indices = np.searchsorted(self.global_ctx.global_time_index, inst_ctx.time_index)
            global_tensor[global_indices, i] = macro_series
            
        # 2. 执行真正的 GPU 截面计算
        physical_op = OP_REGISTRY[op_name]
        result_global = physical_op.batch_compute(global_tensor)
        
        # 3. Gather 回传
        for i, inst_id in enumerate(inst_keys):
            inst_ctx = self.global_ctx.inst_contexts[inst_id]
            global_indices = np.searchsorted(self.global_ctx.global_time_index, inst_ctx.time_index)
            inst_cs_result = result_global[global_indices, i]
            
            # 存回 MACRO_MAIN，真实合约填 NaN，保证宇宙纯净
            final_result = torch.full_like(inst_ctx.get(arg_node), float('nan'))
            macro_idx = list(inst_ctx.contracts).index('MACRO_MAIN')
            final_result[:, macro_idx] = inst_cs_result
            inst_ctx.set(node_id, final_result)

    def run(self, target_nodes_list=None):
        print("\n" + "="*50)
        print("🚀 开始执行 GPU 双层并发批处理计算图 (启用六大准则防护) 🚀")
        print("="*50)
        start_time = time.time()

        if target_nodes_list:
            for target_node in target_nodes_list:
                if target_node in self.out_degrees:
                    # 增加一层外部引用，保证它在算完图后 out_degree 最小也是 1
                    self.out_degrees[target_node] += 1
                else:
                    print(f"[警告] 试图保护不存在的节点: {target_node}")
        
        # 【准则二：DAG 拓扑层并发】
        layers = self.dag.build_execution_layers()
        
        for layer_idx, layer in enumerate(layers):
            print(f"-> 正在计算 Layer {layer_idx} (包含 {len(layer)} 个算子)...")

            active_streams = []
            
            for node_id in layer:
                node_info = self.dag.nodes[node_id]
                op_name = node_info['op']
                
                # --- 场景 1: 叶子节点 (Feature) ---
                if op_name == 'Feature':
                    feat_name = node_info['args'][0]
                    for inst_ctx in self.global_ctx.inst_contexts.values():
                        inst_ctx.set(node_id, inst_ctx.get(feat_name))
                    continue
                
                # --- 场景 2: 截面算子 (Cs_ 开头) ---
                elif op_name.startswith('Cs'):
                    torch.cuda.synchronize() # 屏障：截面计算前必须等所有流跑完
                    self._execute_cross_sectional_node(node_id, node_info)
                    continue
                
                # --- 场景 3: 时序与数学算子 (触发准则二：品种并发) ---
                else:
                    if op_name not in OP_REGISTRY:
                        raise NotImplementedError(f"缺失算子实现: {op_name}")
                    physical_op = OP_REGISTRY[op_name]

                    # # 【准则六-墙2：并发限流】使用线程池削峰
                    # with ThreadPoolExecutor(max_workers=self.max_concurrent_insts) as pool:
                    #     futures = []
                    #     for inst_ctx in self.global_ctx.inst_contexts.values():
                    #         futures.append(pool.submit(self._compute_single_inst, inst_ctx, node_id, node_info, physical_op))
                    #     # 阻塞等待该算子在所有品种上算完
                    #     for f in futures:
                    #         f.result() 
                    for inst_ctx in self.global_ctx.inst_contexts.values():
                        stream = torch.cuda.Stream(device=self.global_ctx.device)
                        active_streams.append(stream)
                        
                        with torch.cuda.stream(stream):
                            inputs = [inst_ctx.get(arg) if isinstance(arg, str) and arg.startswith("Node_") else arg for arg in node_info['args']]
                            # 异步发射！瞬间返回！
                            result_tensor = physical_op.batch_compute(*inputs)
                            inst_ctx.set(node_id, result_tensor)

            # 等待本层所有 Stream 结束
            for stream in active_streams:
                stream.synchronize()

            # 执行显存回收
            for node_id in layer:
                node_info = self.dag.nodes[node_id]
                for arg in node_info['args']:
                    if isinstance(arg, str) and arg.startswith("Node_"):
                        self.out_degrees[arg] -= 1 
                        if self.out_degrees[arg] == 0:
                            for inst_ctx in self.global_ctx.inst_contexts.values():
                                inst_ctx.delete(arg)
            
            torch.cuda.empty_cache()
                
        elapsed = time.time() - start_time
        print(f"=== 计算图执行完毕！总耗时: {elapsed:.3f} 秒 ===")

    # =========================================================
    # 彻底修复：多列同步掩码提取 (Scalable Multiple Extraction)
    # =========================================================
    def extract_multiple_signals(self, target_nodes_dict, mode='main_only', target_start=None, target_end=None):
        """
        【准则四：延迟掩码机制 (Late Masking)】
        支持一次性提取多个节点（如多个因子+标签），并将它们拼接到同一个 DataFrame 中。
        
        参数:
        - target_nodes_dict: Dict[str, str]。例: {'MOM_20': 'Node_5', 'Label_15m': 'Node_8'}
        - switch_cycle_map: Dict[str, pd.Series]。[TradingDate -> MainContract] 映射。
        - mode: 提取模式 ('main_only', 等)
        - target_start/end: 时间截断 (剔除 warm up 区间)
        """
        import pandas as pd
        import numpy as np
        from .utils import calendar_db as simdb
        
        print(f"\n[Late Masking] 正在使用 '{mode}' 模式提取 {len(target_nodes_dict)} 个特征列...")
        result_dfs = []
        cal_df = simdb.calendar[['current_business_date', 'next_business_date']].copy()
        
        ts_target_start = pd.to_datetime(target_start) if target_start else None
        ts_target_end = pd.to_datetime(target_end) if target_end else None

        for inst_id, inst_ctx in self.global_ctx.inst_contexts.items():
            if not hasattr(inst_ctx, 'switch_cycle_map'):
                print(f"[警告] {inst_id} 上下文中没有主力路由表！")
                continue
            
            # inst_switch = switch_cycle_map[inst_id]
            inst_switch = inst_ctx.switch_cycle_map 
            time_idx = pd.to_datetime(inst_ctx.time_index)
            
            # 1. 时间截断掩码 (剔除预热期)
            mask = np.ones(len(time_idx), dtype=bool)
            if ts_target_start: mask &= (time_idx >= ts_target_start)
            if ts_target_end:   mask &= (time_idx <= ts_target_end)
            
            valid_times = time_idx[mask]
            if len(valid_times) == 0:
                continue
                
            # 2. 精准推算 trading_date
            cal_dates = valid_times.normalize()
            df_time = pd.DataFrame({'index_time': valid_times, 'calendar_date': cal_dates})
            df_time = df_time.merge(cal_df, left_on='calendar_date', right_index=True, how='left')
            
            is_night = df_time['index_time'].dt.hour >= 18
            df_time['trading_date'] = df_time['current_business_date']
            df_time.loc[is_night, 'trading_date'] = df_time.loc[is_night, 'next_business_date']
            trading_dates_str = df_time['trading_date'].dt.strftime('%Y-%m-%d').values
            
            # 3. 执行模式提取路由
            if mode == 'main_only':

                # 将 trading_date 作为键，main_contract 作为值，转换为 pd.Series
                if isinstance(inst_switch, pd.DataFrame):
                    # 确保日期列是字符串格式，方便与 trading_dates_str 匹配
                    dt_series = pd.to_datetime(inst_switch['trading_date']).dt.strftime('%Y-%m-%d')
                    mapping_series = pd.Series(inst_switch['main_contract'].values, index=dt_series)
                else:
                    mapping_series = inst_switch
                # a) 找出每一分钟的主力合约索引
                main_contracts = pd.Series(trading_dates_str).map(inst_switch).fillna('UNKNOWN').values
                contracts_array = inst_ctx.contracts
                contract_to_idx = {c: i for i, c in enumerate(contracts_array)}
                col_indices = np.array([contract_to_idx.get(c, -1) for c in main_contracts])
                
                valid_mask = col_indices != -1
                row_indices = np.arange(len(valid_times))

                # MACRO_MAIN 虚拟列索引
                macro_idx = contract_to_idx.get('MACRO_MAIN', -1)
                
                # b) 构造该品种的基础 DataFrame
                df_out = pd.DataFrame({
                    'index_time': valid_times,
                    'instrument': inst_id,
                    'contract': main_contracts
                })
                
                # c) 核心遍历：瞬间提取所有目标列
                for col_name, node_id in target_nodes_dict.items():
                    tensor_np = inst_ctx.get(node_id).cpu().numpy()
                    valid_tensor = tensor_np[mask, :] # 截断时间
                    
                    extracted_signals = np.full(len(valid_times), np.nan, dtype=np.float32)
                    # 瞬间抽出对角线主力数据！
                    extracted_signals[valid_mask] = valid_tensor[row_indices[valid_mask], col_indices[valid_mask]]

                    # B. 提取宏观宇宙 (MAIN_CONTRACT)
                    macro_signals = np.full(len(valid_times), np.nan, dtype=np.float32)
                    if macro_idx != -1:
                        macro_signals = valid_tensor[:, macro_idx]

                    combined_signals = np.where(np.isnan(extracted_signals), macro_signals, extracted_signals)
                    
                    df_out[col_name] = combined_signals
                    
                result_dfs.append(df_out)

            elif mode == 'all_contracts':
                contracts_array = list(inst_ctx.contracts)
                # 挑出所有真实合约（排除 MACRO_MAIN 虚拟列）
                real_contracts = [c for c in contracts_array if c != 'MACRO_MAIN']
                real_c_indices = [i for i, c in enumerate(contracts_array) if c != 'MACRO_MAIN']
                n_contracts = len(real_contracts)
                n_times = len(valid_times)

                # 每个时间点展开为 n_contracts 行，每行对应一个真实合约
                times_repeated = np.repeat(valid_times, n_contracts)
                contracts_repeated = np.tile(real_contracts, n_times)

                df_out = pd.DataFrame({
                    'index_time': times_repeated,
                    'instrument': inst_id,
                    'contract': contracts_repeated,
                })

                for col_name, node_id in target_nodes_dict.items():
                    tensor_np = inst_ctx.get(node_id).cpu().numpy()
                    valid_tensor = tensor_np[mask, :]                              # (n_times, C_i)
                    real_contract_data = valid_tensor[:, real_c_indices]           # (n_times, n_contracts)
                    df_out[col_name] = real_contract_data.flatten(order='C')       # 逐行展开

                result_dfs.append(df_out)
                
            else:
                raise ValueError(f"未知的提取模式: {mode}")
                
        # 4. 拼接全市场结果并按时间排序
        if not result_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(result_dfs, ignore_index=True)
        final_df = final_df.sort_values(by=['index_time', 'instrument']).reset_index(drop=True)
        
        print(f"[Late Masking] 提取完成！输出 DataFrame 形状: {final_df.shape}")
        return final_df