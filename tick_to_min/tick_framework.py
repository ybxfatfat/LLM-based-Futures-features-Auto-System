# 文件路径: tick_framework.py
import numpy as np
import pandas as pd
import tick_engine as ce

class SymbolicSnapshot:
    """定义合法的快照占位符，LLM 只能点出这里的属性"""
    def __init__(self, contract_size):
        self.nodes = {}
        self.contract_size = ce.ConstNode(float(contract_size))
        self.last_price = self._create_node("last_price")
        self.cum_volume = self._create_node("cum_volume")
        self.cum_amount = self._create_node("cum_amount")
        self.open_interest = self._create_node("open_interest")
        
        for i in range(1, 6):
            setattr(self, f"bid_{i}_prc", self._create_node(f"bid_{i}_prc"))
            setattr(self, f"bid_{i}_qty", self._create_node(f"bid_{i}_qty"))
            setattr(self, f"ask_{i}_prc", self._create_node(f"ask_{i}_prc"))
            setattr(self, f"ask_{i}_qty", self._create_node(f"ask_{i}_qty"))

    def _create_node(self, name):
        node = ce.FeatureNode()
        self.nodes[name] = node
        return node

# ======================================================================
# 升级 1：BaseTickFeature 支持自定义 clear (auto_clear=True/False)
# ======================================================================
class BaseTickFeature:
    def __init__(self):
        # 拆分为需要自动 clear 的，和跨分钟保留的
        self._auto_clear_ops = []
        self._persistent_ops = []

    def register_state_op(self, op: ce.OpNode, auto_clear: bool = True):
        """
        注册状态算子。
        auto_clear=True (默认): 每一分钟末尾被自动清零 (Bucket 模式)。
        auto_clear=False: 横跨整个交易日保留状态。
        """
        if auto_clear:
            self._auto_clear_ops.append(op)
        else:
            self._persistent_ops.append(op)
        return op

    def update(self, snp: SymbolicSnapshot):
        raise NotImplementedError

    def get_value(self) -> dict:
        raise NotImplementedError

# ======================================================================
# 升级 2：新增 FeatureCombiner (神级组合器)
# ======================================================================
class FeatureCombiner(BaseTickFeature):
    """
    接收多个独立的 Feature 类，将它们自动拼接成一个巨大的计算图！
    """
    def __init__(self, feature_classes: list):
        super().__init__()
        # 实例化所有传入的子特征类
        self.features = [cls() for cls in feature_classes]
        
        # 收集所有的状态算子
        for f in self.features:
            self._auto_clear_ops.extend(f._auto_clear_ops)
            self._persistent_ops.extend(f._persistent_ops)

    def update(self, snp: SymbolicSnapshot):
        # 把同一个 snp 喂给所有的子特征，让它们共享叶子节点！
        for f in self.features:
            f.update(snp)

    def get_value(self) -> dict:
        # 将所有子特征的输出字典，合并成一个超级大字典！
        combined_dict = {}
        for f in self.features:
            out_dict = f.get_value()
            for k, v in out_dict.items():
                if k in combined_dict:
                    print(f"⚠️ 警告: 发现重名的特征输出列 '{k}'，将被覆盖！")
                combined_dict[k] = v
        return combined_dict

# 原始行情快照字段到标准字段的映射字典
SNAPSHOT_COLUMN_MAP = {
    'last_prc': 'last_price',
    'trvo': 'cum_volume',
    'mtch_amt': 'cum_amount',
    'vohp': 'open_interest',
    'pursbuy_one_prc': 'bid_1_prc', 'pursbuy_one_qty': 'bid_1_qty',
    'purssell_one_prc': 'ask_1_prc', 'purssell_one_qty': 'ask_1_qty',
    'pursbuy_two_prc': 'bid_2_prc', 'pursbuy_two_qty': 'bid_2_qty',
    'purssell_two_prc': 'ask_2_prc', 'purssell_two_qty': 'ask_2_qty',
    'pursbuy_three_prc': 'bid_3_prc', 'pursbuy_three_qty': 'bid_3_qty',
    'purssell_three_prc': 'ask_3_prc', 'purssell_three_qty': 'ask_3_qty',
    'pursbuy_four_prc': 'bid_4_prc', 'pursbuy_four_qty': 'bid_4_qty',
    'purssell_four_prc': 'ask_4_prc', 'purssell_four_qty': 'ask_4_qty',
    'pursbuy_five_prc': 'bid_5_prc', 'pursbuy_five_qty': 'bid_5_qty',
    'purssell_five_prc': 'ask_5_prc', 'purssell_five_qty': 'ask_5_qty'
}

# ======================================================================
# 全市场合约乘数映射字典 (从 Base Panel 提取)
# ======================================================================
CONTRACT_SIZE_MAP = {
    'AL': 5, 'CU': 5, 'ZN': 5, 'AU': 1000, 'PB': 5, 'RU': 10, 'AO': 20, 
    'RB': 10, 'WR': 10, 'FU': 10, 'AG': 15, 'LC': 1, 'NI': 1, 'SN': 1, 
    'BU': 10, 'SP': 10, 'HC': 10, 'SS': 5, 'SC': 1000, 'LU': 10, 'NR': 10, 
    'BC': 5, 'A': 10, 'B': 10, 'C': 10, 'CS': 10, 'M': 10, 'Y': 10, 
    'P': 10, 'L': 5, 'V': 5, 'BR': 5, 'JM': 60, 'J': 100, 'I': 100, 
    'FB': 10, 'BB': 500, 'JD': 10, 'EG': 10, 'PP': 5, 'RR': 10, 'EB': 5, 
    'PG': 20, 'LH': 16, 'WH': 20, 'PM': 50, 'TA': 5, 'ER': 20, 'LR': 20, 
    'CF': 5, 'SR': 10, 'RO': 10, 'MA': 10, 'FG': 20, 'RM': 10, 'RS': 10, 
    'JR': 20, 'SM': 5, 'SF': 5, 'ZC': 100, 'CJ': 5, 'AP': 10, 'UR': 20, 
    'SA': 20, 'PF': 5, 'PK': 5, 'IF': 300, 'IH': 300, 'IC': 200, 'IM': 200, 
    'TS': 20000, 'TF': 10000, 'T': 10000, 'TL': 10000, 'EC': 50,

    'OI': 10, 'ME': 10
}

# ======================================================================
# 升级 3：引擎执行函数支持提取 persistent_ops，且只传 auto_clear_ops 给 C++
# ======================================================================
def _compile_and_run_single_contract(combiner_instance: FeatureCombiner, df_single_tick: pd.DataFrame, contract_size: float) -> pd.DataFrame:
    """内部函数：处理单个合约的 Tick 数据并调用 C++ 引擎"""
    dummy_snp = SymbolicSnapshot(contract_size)
    
    # 1. 符号追踪与图构建
    combiner_instance.update(dummy_snp)
    output_nodes_dict = combiner_instance.get_value() # 获取的是特征名到节点的字典
    
    # ⚠️ 【极其重要】：只把需要自动清空的算子传给 C++ 去 clear
    state_nodes_to_clear = combiner_instance._auto_clear_ops
    
    # 2. 拓扑排序 (DFS)
    visited = set()
    topo_order = []
    def dfs(node):
        node_id = id(node)
        if node_id in visited: return
        visited.add(node_id)
        for child in node.get_inputs():
            dfs(child)
        topo_order.append(node)

    # 从字典中所有的输出节点开始溯源
    for node in output_nodes_dict.values():
        dfs(node)
    for s_node in state_nodes_to_clear:
        dfs(s_node)
    # 那些跨分钟保留的算子也必须被加入计算拓扑！
    for p_node in combiner_instance._persistent_ops:
        dfs(p_node)
        
    # 3. 准备底层 Arrow/Numpy 内存指针
    input_arrays = {}
    for col_name, f_node in dummy_snp.nodes.items():
        if col_name in df_single_tick.columns:
            input_arrays[col_name] = np.ascontiguousarray(df_single_tick[col_name].to_numpy(dtype=np.float32))
        else:
            input_arrays[col_name] = np.ascontiguousarray(np.zeros(len(df_single_tick), dtype=np.float32))
            
    # 时间戳解析
    ts_series = df_single_tick['index_time']
    timestamps_ms = np.ascontiguousarray((ts_series.astype('int64') // 10**6).to_numpy())

    # 4. 执行 C++ 引擎
    # C++ 现在返回的是 (timestamps_array, dict_of_feature_arrays)
    out_ts_ms, out_features_dict = ce.run_tick_engine(
        timestamps_ms, dummy_snp.nodes, input_arrays, topo_order, output_nodes_dict, state_nodes_to_clear
    )
    
    # 5. 组装输出
    result_dict = {
        'index_time': pd.to_datetime(out_ts_ms, unit='ms'),
        'contract': df_single_tick['secu_cd'].iloc[0]
    }
    result_dict.update(out_features_dict) # 合并所有特征列
    
    return pd.DataFrame(result_dict)

import re

# ======================================================================
# 升级 4：入口函数接收 List 类型的特征类，并内部实例化 Combiner
# ======================================================================
def process_daily_ticks(feature_classes: list, df_daily_tick: pd.DataFrame) -> pd.DataFrame:
    """
    研究员调用的主入口：
    接收某一天的全量快照数据，自动按合约分组计算，返回可直接与 base_panel 拼接的 DataFrame。
    """
    # 1. 极速重命名列，映射为标准名称
    df_mapped = df_daily_tick.rename(columns=SNAPSHOT_COLUMN_MAP)
    
    # 2. 极速解析字符串时间戳
    df_mapped['index_time'] = pd.to_datetime(
        df_mapped['quot_time'].astype(str).str.slice(0, 17), 
        format="%Y%m%d%H%M%S%f"
    )
    
    # 3. 按合约 (secu_cd) 分组计算
    results = []
    grouped = df_mapped.groupby('secu_cd')
    print(f"正在处理 {len(grouped)} 个合约的高频特征...")
    
    for contract, df_single in grouped:
        df_single = df_single.sort_values('index_time')
        
        # 实例化神级组合器，将所有类打包
        combiner = FeatureCombiner(feature_classes)

        # 极速提取品种前缀 (大写)
        prefix_match = re.match(r'^([a-zA-Z]+)', contract)
        prefix = prefix_match.group(1).upper() if prefix_match else 'UNKNOWN'
        
        # 查询乘数字典，找不到默认给 1.0
        c_size = CONTRACT_SIZE_MAP.get(prefix, 1.0)
        
        # 调用核心引擎
        res_df = _compile_and_run_single_contract(combiner, df_single, c_size)
        results.append(res_df)
        
    # 4. 合并所有合约的结果
    if not results:
        return pd.DataFrame()
        
    final_result = pd.concat(results, ignore_index=True)
    final_result = final_result.sort_values(['index_time', 'contract']).reset_index(drop=True)
    return final_result