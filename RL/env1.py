# env.py 用于保存上一版本的env
import util.parameter as parameter
from Reliability import calc_trust as Reliability
import numpy as np
import random
import Node.node as node

Nodes = {}

class EdgeEnv:
    def __init__(self, node_num):
        self.evaluator = Reliability.TrustEvaluator()
        self.node_num = node_num
        self.min_selected = (node_num + 1) // 2
        self.nodes = {}
        self.state_dim = []
        self.current_time = 0
        self.scenario_mode = 'normal'  # normal, low_perf, low_rel, low_sec
        self.flag = 0

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
        Nodes = self.nodes
        # 初始评估
        all_nodes = list(self.nodes.values())
        for n in all_nodes:
            neighbors = [x for x in all_nodes if x.Id != n.Id]
            self.evaluator.evaluate_node(n, neighbors, 0)
            self.state_dim.extend(self._get_node_feature(n))
        return np.array(self.state_dim)

    def reset1(self):
        self.state_dim = []
        self.nodes = Nodes
        self.current_time = 0
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
                if random.random() < 0.3:
                    node.attack_rate += random.randint(1, 2)
                    node.leakage += random.uniform(1, 5)

    def step(self, actions: dict, trust_pred: dict, weights_pack: dict, nonce):
        self.current_time = nonce
        next_state = []
        all_nodes_list = list(self.nodes.values())

        # 1. 环境动态恶化
        self._simulate_environment_drift()

        # 2. Agent 调整权重
        # 你的观点是对的：在 low_sec 下，Agent 可能会学出提高 perf/rel 权重以维持分数
        self.evaluator.update_all_weights(
            weights_pack['dim'], weights_pack['perf'],
            weights_pack['rel'], weights_pack['sec']
        )

        # 3. 执行选节点动作
        selected_nodes = []
        selected_trust_values = []  # 记录被选节点的信任值

        for nid, act in actions.items():
            node_obj = self.nodes[nid]
            # 计算当前权重下的信任值
            neighbors = [n for n in all_nodes_list if n.Id != nid]
            current_trust = self.evaluator.calculate_only(node_obj, neighbors, nonce)
            node_obj.trust_current = current_trust  # 更新节点状态

            if act == 1:
                node_obj.is_select = True
                selected_nodes.append(node_obj)
                selected_trust_values.append(current_trust)
            else:
                node_obj.is_select = False

        # 4. 状态更新
        for node_obj in all_nodes_list:
            next_state.extend(self._get_node_feature(node_obj))

        # 5. 计算奖励 (核心修改)
        reward = self.calc_reward_survival(selected_nodes, weights_pack)

        # 6. 计算共识信任 (用于日志显示)
        consensus_trust = np.mean(selected_trust_values) if len(selected_trust_values) > 0 else 0.0

        # 7. 结束条件
        done = (nonce >= 200)

        info = {
            "trust_totals": {k: v.trust_current for k, v in self.nodes.items()},
            "consensus_trust": consensus_trust
        }
        return np.array(next_state), reward, done, info

    def calc_reward_survival(self, selected_nodes, weights_pack):
        # 1. 系统存活惩罚 (System Survival)
        if len(selected_nodes) < self.min_selected:

            return -100.0

        total_reward = 0.0
        # 2. 基础奖励：系统正常运转
        total_reward += 10.0

        # 3. 质量评估 (Quality Assessment)
        avg_attack = 0
        avg_perf = 0

        for node in selected_nodes:
            # --- 安全性维度 (Negative) ---
            if node.attack_rate > 5 or node.leakage > 5:
                # 扣分力度减小，允许少量“坏节点”混入，只要系统不崩
                total_reward -= (node.attack_rate * 2.0)

            total_reward += (node.Throughput / 100.0) * 2.0
            total_reward += (node.trust_current * 10.0)  # 奖励最终信任值高

        # 4. 场景特化逻辑 (Scenario Logic)
        if self.scenario_mode == 'low_sec':
            # 只有当选中了节点，且其中包含了好节点时，给额外奖励
            safe_nodes = [n for n in selected_nodes if n.attack_rate < 5]
            total_reward += len(safe_nodes) * 5.0

        elif self.scenario_mode == 'low_perf':
            # 低性能模式
            avg_resp = np.mean([n.Response_Time for n in selected_nodes])
            total_reward -= avg_resp * 0.1

        # 5. 归一化，防止梯度爆炸
        return np.clip(total_reward, -100, 100)
