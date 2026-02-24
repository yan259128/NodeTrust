import numpy as np

# ==========================================
# 1. 传统无强化学习代理 (No-RL 基准 - 使用默认固定权重)
# ==========================================
class NoRLAgent:
    """
    传统静态信任评估模型：
    - 使用用户提供的默认固定权重。
    - 采用贪心策略选择可信度最高的节点。
    """

    def __init__(self, node_num):
        self.node_num = node_num
        self.min_selected = (node_num + 1) // 2

        # --- 设置你提供的默认权重 ---
        self.default_w_pack = {
            'dim': np.array([0.3, 0.4, 0.3]),  # perf, rel, sec
            'perf': np.array([0.25, 0.25, 0.2, 0.2, 0.1]),
            'rel': np.array([0.3, 0.3, 0.25, 0.15]),
            'sec': np.array([0.4, 0.2, 0.2, 0.2])
        }

    def select_action(self, state):
        # 从 state 向量中提取：信任总值(索引14) 和 身份认证(索引9)
        # 备注：env.py 中 _get_node_feature 里的第15个值是 trust_total
        trusts = [state[i * 16 + 14] for i in range(self.node_num)]
        is_ids = [state[i * 16 + 9] for i in range(self.node_num)]

        # 挑出已认证节点并排序
        valid_nodes = [(i, trusts[i]) for i in range(self.node_num) if is_ids[i] > 0.5]
        valid_nodes.sort(key=lambda x: x[1], reverse=True)

        # 选出 Top-K
        selected_indices = [x[0] for x in valid_nodes[:self.min_selected]]

        # 补齐逻辑（防止认证节点不足）
        if len(selected_indices) < self.min_selected:
            remaining = [i for i in range(self.node_num) if i not in selected_indices]
            remaining.sort(key=lambda i: trusts[i], reverse=True)
            needed = self.min_selected - len(selected_indices)
            selected_indices.extend(remaining[:needed])

        act_dict = {f'node_{i}': (1 if i in selected_indices else 0) for i in range(self.node_num)}

        # 使用固定的默认权重
        return act_dict, {f'node_{i}': 0.5 for i in range(self.node_num)}, self.default_w_pack, None