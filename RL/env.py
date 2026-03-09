# env.py
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
        # 定义归一化辅助函数
        def norm(val, min_v, max_v):
            return (np.clip(val, min_v, max_v) - min_v) / (max_v - min_v + 1e-6)

        features = [
            # Performance
            norm(node.Response_Time, 1, 500),
            norm(node.Request_Success_Rate, 80, 100),
            norm(node.Throughput, 200, 800),
            norm(node.bandwidth, 1, 1000),
            norm(node.handle, 1, 100),

            # Reliability
            norm(node.task_completion_rate, 80, 100),
            norm(node.survival_rate, 80, 100),
            norm(node.verify_transaction_nums, 0, 10000),
            norm(node.service_quality, 0, 10),

            # Security
            1.0 if node.is_identification else 0.0,
            norm(node.leakage, 0, 100),
            norm(node.attack_rate, 0, 10),
            norm(node.user_get_data_rate, 80, 100),

            # Trust & Stats
            node.trust_current,  # 已经是 0-1
            node.trust_total,  # 已经是 0-1
            norm(node.R_sec, 0, 100)
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
        num_selected = len(selected_nodes)

        # 1. 系统存活奖励/惩罚 (平滑化)
        # 之前的 -100 惩罚太硬，导致 Agent 放弃学习。改为梯度惩罚。
        if num_selected < self.min_selected:
            # 缺一个节点扣 10 分，比直接扣 100 要好引导 Agent
            survival_reward = -10.0 * (self.min_selected - num_selected)
        else:
            survival_reward = 10.0

        if num_selected == 0:
            return -50.0  # 至少选一个，否则重罚

        # 2. 节点质量奖励 (均值化，防止节点数量越多总分越高的偏移)
        # 使用归一化后的指标计算
        avg_trust = np.mean([n.trust_current for n in selected_nodes])

        # 安全性惩罚：如果选中的节点 attack_rate 高，重扣分
        # 对应你给的范围 0-10，如果 attack_rate 是 5，则扣 5*2=10分
        avg_attack = np.mean([n.attack_rate for n in selected_nodes])
        security_penalty = avg_attack * 2.0

        # 性能奖励：Throughput 200-800，归一化后是 0-1
        avg_thru = np.mean([(n.Throughput - 200) / 600.0 for n in selected_nodes])
        perf_reward = avg_thru * 5.0

        total_reward = survival_reward + (avg_trust * 20.0) + perf_reward - security_penalty

        # 3. 场景补偿 (防止环境切换时 Reward 掉到地心)
        if self.scenario_mode == 'low_perf':
            # Response_Time 范围 1-500。如果平均 400ms，扣 (400/100) = 4 分
            avg_resp = np.mean([n.Response_Time for n in selected_nodes])
            total_reward -= (avg_resp / 100.0)

        elif self.scenario_mode == 'low_sec':
            # 在安全模式下，如果 Agent 能选出 attack_rate 低于 2 的节点，给额外奖励
            safe_nodes_count = sum(1 for n in selected_nodes if n.attack_rate < 2)
            total_reward += safe_nodes_count * 2.0

        # 将单步 reward 限制在 [-20, 20] 左右，有利于 TD3 平稳收敛
        return np.clip(total_reward, -50, 50)
