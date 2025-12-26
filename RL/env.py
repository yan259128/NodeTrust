# env.py
import util.parameter as parameter
from Reliability import calc_trust as Reliability
import numpy as np
import random


class EdgeEnv:
    def __init__(self, node_num):
        self.evaluator = Reliability.TrustEvaluator()
        self.node_num = node_num
        self.min_selected = (node_num + 1) // 2
        self.nodes = {}
        self.state_dim = []
        self.current_time = 0
        self.scenario_mode = 'normal'  # normal, low_perf, low_rel, low_sec

    def set_scenario(self, mode):
        self.scenario_mode = mode

    def _get_node_feature(self, node):
        features = [
            node.Response_Time, node.Request_Success_Rate, node.Throughput,
            node.bandwidth, node.handle,
            node.task_completion_rate, node.survival_rate, node.verify_transaction_nums,
            node.service_quality,
            1.0 if node.is_identification else 0.0,
            node.leakage, node.attack_rate, node.user_get_data_rate,
            node.trust_current, node.trust_total, node.R_sec
        ]
        return features

    def reset(self):
        self.state_dim = []
        self.nodes = {}
        self.current_time = 0
        for i in range(self.node_num):
            nid = f'node_{i}'
            self.nodes[nid] = node.Node(nid, parameter.Port + i)

        # 初始评估
        all_nodes = list(self.nodes.values())
        for n in all_nodes:
            neighbors = [x for x in all_nodes if x.Id != n.Id]
            self.evaluator.evaluate_node(n, neighbors, 0)
            self.state_dim.extend(self._get_node_feature(n))
        return np.array(self.state_dim)

    def _simulate_environment_drift(self):
        """根据场景模式恶化环境"""
        for node in self.nodes.values():
            node.simulate_dynamic_change()  # 基础波动

            if self.scenario_mode == 'low_perf':
                # 性能环境恶化：延迟大增，带宽大减
                node.Response_Time += random.randint(100, 400)
                node.bandwidth = max(10, node.bandwidth * 0.3)
                node.Throughput = max(50, node.Throughput * 0.4)

            elif self.scenario_mode == 'low_rel':
                # 可靠性环境恶化：掉线率高，任务完成率低
                node.survival_rate = max(40, node.survival_rate - random.uniform(10, 30))
                node.task_completion_rate = max(50, node.task_completion_rate - random.uniform(10, 20))

            elif self.scenario_mode == 'low_sec':
                # 安全环境恶化：被攻击频率高，泄露风险增加
                if random.random() < 0.4:
                    node.attack_rate += random.randint(2, 5)
                    node.leakage += random.uniform(5, 15)

    def step(self, actions: dict, trust_pred: dict, weights_pack: dict, nonce):
        self.current_time = nonce
        next_state = []
        all_nodes_list = list(self.nodes.values())

        # 1. 模拟环境恶化 (发生在评估之前)
        self._simulate_environment_drift()

        # 2. [关键] 计算两种信任值 (用于 Reward 对比)
        # 2.1 使用 Agent 权重计算 T_adaptive (这是真实生效的)
        # print("权重更新")
        self.evaluator.update_all_weights(
            weights_pack['dim'], weights_pack['perf'],
            weights_pack['rel'], weights_pack['sec']
        )
        # print("权重： ", weights_pack)
        # 执行动作带来的状态改变 (如惩罚)
        selected_nodes = []
        for nid, act in actions.items():
            node = self.nodes[nid]
            node.trust_rl = trust_pred[nid]
            if act == 1:
                node.is_select = True
                selected_nodes.append(node)
                if not node.is_identification or node.attack_rate > 8:  # 严格点
                    reporters = [n for n in all_nodes_list if n.Id != nid and n.is_select]
                    level = 'L4' if not node.is_identification else 'L2'
                    self.evaluator.punish_node(node, level, reporters, nonce)
            else:
                node.is_select = False
        # print("Nonce: {}".format(nonce))

        # 正式更新节点状态 (T_adaptive)
        for node in all_nodes_list:
            neighbors = [n for n in all_nodes_list if n.Id != node.Id]
            # 这会更新 node.trust_total
            self.evaluator.evaluate_node(node, neighbors, nonce)
            # 记录下适应性权重算出的值
            node.temp_adaptive_trust = node.trust_total
            # print("{}.temp_adaptive_trust: {}".format(node.Id, node.trust_total))

        # 2.2 使用默认权重计算 T_baseline (仅计算不更新)
        self.evaluator.reset_defaults()
        # print("权重重置")
        for node in all_nodes_list:
            neighbors = [n for n in all_nodes_list if n.Id != node.Id]
            base_val = self.evaluator.calculate_only(node, neighbors, nonce)
            node.trust_baseline = base_val
            # print("Nonce {}, {}.trust_baseline: {}, temp_adaptive_trust: {}".format(nonce, node.Id, node.trust_baseline, node.trust_total))

        # 3. 构建 Next State (包含最新的 adaptive trust)
        for node in all_nodes_list:
            next_state.extend(self._get_node_feature(node))

        # 4. 计算奖励 (传入 actions 以便知道选了谁)
        reward = self.calc_reward_contrast(actions)

        done = (nonce >= 200)  # 演示用200轮
        info = {
            "trust_totals": {k: v.trust_total for k, v in self.nodes.items()},
            "weights": weights_pack
        }
        return np.array(next_state), reward, done, info

    def calc_reward_contrast(self, actions):
        """
        奖励函数
        """
        selected_nodes = [self.nodes[nid] for nid, act in actions.items() if act == 1]

        # 1. 硬约束
        if len(selected_nodes) < self.min_selected: return -20.0
        for node in selected_nodes:
            if not node.is_identification: return -50.0

        total_reward = 0

        # 记录本轮的平均适应性信任值，用于计算趋势
        current_avg_adaptive = 0

        for node in selected_nodes:
            # 增量 = 自适应权重分 - 死板基准分
            diff = node.temp_adaptive_trust - node.trust_baseline

            # 判断节点是否本身是“好人” (Ground Truth)
            is_good_node = (node.R_sec < 10 and node.service_quality > 4.0)

            if is_good_node:
                current_avg_adaptive += node.temp_adaptive_trust

                if diff > 0:
                    # [优化点1] 指数级奖励
                    # 即使 diff 只有 0.1，exp(0.1*5) = 1.64，放大倍数显著
                    # 如果 diff 达到 0.3，exp(0.3*5) = 4.48
                    # 这会让 Agent 极度渴望拉大 gap
                    scale = 5.0
                    reward_boost = (np.exp(diff * scale) - 1.0) * 20.0
                    total_reward += reward_boost
                else:
                    # 策略无效，轻微惩罚
                    total_reward += diff * 5.0
            else:
                # 坏节点，严厉打击洗白行为
                if diff > 0:
                    total_reward -= diff * 100.0  # 极重罚
                else:
                    total_reward += abs(diff) * 10.0

        # 如果本轮的信任值 比 上一轮 还有提升，额外奖励
        # 这鼓励 Agent 不断微调权重，而不是满足于现状
        if hasattr(self, 'last_avg_trust'):
            avg = current_avg_adaptive / max(1, len(selected_nodes))
            delta = avg - self.last_avg_trust
            if delta > 0:
                total_reward += delta * 50.0  # 奖励进步

        # 更新历史记录
        self.last_avg_trust = current_avg_adaptive / max(1, len(selected_nodes))

        # 归一化防止梯度爆炸
        return np.clip(total_reward, -100, 100)