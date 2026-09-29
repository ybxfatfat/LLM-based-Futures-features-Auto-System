import ast

class DAGManager:
    """
    因子计算图引擎的"大脑"
    负责：解析字符串公式 -> 消除重复计算(CSE) -> 生成并行调度层
    """
    def __init__(self):
        self.node_registry = {}
        self.nodes = {}
        self.dependencies = {}
        self._node_counter = 0

    def _get_or_create_node(self, op_name, *args):
        # 1. 交换律优化
        sorted_args = list(args)
        if op_name in ['Add', 'Mul', 'Max', 'Min', 'Equal']:
            sorted_args = sorted(sorted_args, key=str)
            
        sig_args = ",".join(str(a) for a in sorted_args)
        signature = f"{op_name}:{sig_args}"
        
        if signature in self.node_registry:
            return self.node_registry[signature] 
            
        node_id = f"Node_{self._node_counter}"
        self._node_counter += 1
        
        self.node_registry[signature] = node_id
        self.nodes[node_id] = {'op': op_name, 'args': args, 'signature': signature}
        
        deps = [a for a in args if isinstance(a, str) and a.startswith("Node_")]
        self.dependencies[node_id] = deps
        return node_id

    def _parse_ast(self, node):
        """递归解析 Python AST 抽象语法树"""
        # 1. 基础字段
        if isinstance(node, ast.Name):
            return self._get_or_create_node("Feature", node.id)
            
        # 2. 常数
        elif isinstance(node, ast.Constant):
            return node.value
            
        # 3. 函数调用 (如 TsMean(x, 20), IfElse(cond, x, y))
        elif isinstance(node, ast.Call):
            op_name = node.func.id
            args = [self._parse_ast(arg) for arg in node.args]
            return self._get_or_create_node(op_name, *args)
            
        # 4. 二元运算 (如 a + b)
        elif isinstance(node, ast.BinOp):
            op_map = {ast.Add: 'Add', ast.Sub: 'Sub', ast.Mult: 'Mul', ast.Div: 'Div'}
            op_name = op_map[type(node.op)]
            return self._get_or_create_node(op_name, self._parse_ast(node.left), self._parse_ast(node.right))
            
        # 5. 【新增】：一元运算 (如 -x 转换为 Mul(-1, x))
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return self._get_or_create_node('Mul', -1.0, self._parse_ast(node.operand))
            raise NotImplementedError(f"不支持的一元操作符: {type(node.op)}")
            
        # 6. 【新增】：逻辑比较运算 (如 a > b 转换为 MoreThan(a, b))
        elif isinstance(node, ast.Compare):
            # 为简单起见，暂支持单次比较如 a > b
            op_map = {ast.Gt: 'MoreThan', ast.Lt: 'LessThan', ast.Eq: 'Equal'}
            op_name = op_map[type(node.ops[0])]
            return self._get_or_create_node(op_name, self._parse_ast(node.left), self._parse_ast(node.comparators[0]))
            
        else:
            raise NotImplementedError(f"不支持的语法结构: {type(node)}")

    def add_formula(self, formula_str):
        tree = ast.parse(formula_str, mode='eval').body
        return self._parse_ast(tree)

    def build_execution_layers(self):
        in_degree = {n: len(deps) for n, deps in self.dependencies.items()}
        children_to_parents = {n: [] for n in self.nodes}
        for parent, deps in self.dependencies.items():
            for child in deps:
                children_to_parents[child].append(parent)
                
        ready_queue = [n for n, d in in_degree.items() if d == 0]
        execution_layers = []
        
        while ready_queue:
            execution_layers.append(ready_queue)
            next_queue = []
            for node in ready_queue:
                for parent in children_to_parents[node]:
                    in_degree[parent] -= 1
                    if in_degree[parent] == 0:
                        next_queue.append(parent)
            ready_queue = next_queue
            
        return execution_layers

# =========================================================
# 模拟测试：见证奇迹的时刻
# =========================================================
if __name__ == "__main__":
    dag = DAGManager()
    
    # 模拟研究员提交了 3 个因子公式
    # 注意：TsMean(close_adj, 20) 在这里被多次重复使用！
    formulas = [
        "TsMean(close, 20) + volume",
        "TsMean(close, 20) * 1.5",
        "TsZScore(TsMean(close, 20), 60)",
        "TsRank(SafeDiv(TsMean(Close, 20),Volume),10)"
    ]
    
    print("1. 开始批量解析公式并去重 (CSE)...")
    for idx, f in enumerate(formulas):
        out_node = dag.add_formula(f)
        print(f"公式 {idx+1} 最终输出节点: {out_node}")
        
    print("\n2. 系统全局节点注册表 (总计算量被极大压缩):")
    for node_id, details in dag.nodes.items():
        print(f"  {node_id}: {details['signature']}")

    print("\n3. GPU 并发调度执行计划 (Execution Layers):")
    layers = dag.build_execution_layers()
    
    for i, layer in enumerate(layers):
        print(f"  --- Layer {i} (当前层算子可利用 CUDA Streams 并发执行) ---")
        for node_id in layer:
            op = dag.nodes[node_id]['op']
            args = dag.nodes[node_id]['args']
            print(f"    发射指令 -> {node_id} = 执行 {op}({', '.join(map(str, args))})")