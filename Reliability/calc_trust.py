# calc_trust.py
import math
import numpy as np
import copy


class TrustEvaluator:
    def __init__(self):
        # 1. 因子配置
        self.factor_configs = {
            'performance': [
                ('Response_Time', False, (1, 500)),
                ('Request_Success_Rate', True, (80, 100)),
                ('Throughput', True, (200, 800)),
                ('bandwidth', True, (1, 1000)),
                ('handle', True, (1, 100))
            ],
            'reliability': [
                ('task_completion_rate', True, (80, 100)),
                ('survival_rate', True, (80, 100)),
                ('verify_transaction_nums', True, (0, 10000)),
                ('service_quality', True, (0, 10))
            ],
            'security': [
                ('is_identification', None, (0, 1)),
                ('leakage', False, (0, 100)),
                ('attack_rate', False, (0, 10)),
                ('user_get_data_rate', True, (80, 100))
            ]
        }

        # 2. 默认权重 (Baseline) - 备份一份用于重置
        self.default_dim_weights = {'performance': 0.3, 'reliability': 0.4, 'security': 0.3}
        self.default_factor_weights = {
            'performance': np.array([0.25, 0.25, 0.2, 0.2, 0.1]),
            'reliability': np.array([0.3, 0.3, 0.25, 0.15]),
            'security': np.array([0.4, 0.2, 0.2, 0.2])
        }

        # 3. 当前使用的权重 (初始化为默认)
        self.dim_weights = copy.deepcopy(self.default_dim_weights)
        self.factor_weights = copy.deepcopy(self.default_factor_weights)

        # 4. 评价集与参数
        self.V = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.S = self.V
        self.lambda_decay = 0.1
        self.eta_penalty = 0.05
        self.lambda_recover = 0.05
        self.delta_history = 0.3
        self.malicious_levels = {'L1': 10, 'L2': 30, 'L3': 70, 'L4': 100}

    # --- 权重管理接口 ---
    def update_all_weights(self, dim_w, perf_w, rel_w, sec_w):
        """应用 RL Agent 生成的动态权重"""
        self.dim_weights['performance'] = dim_w[0]
        self.dim_weights['reliability'] = dim_w[1]
        self.dim_weights['security'] = dim_w[2]

        self.factor_weights['performance'] = np.array(perf_w)
        self.factor_weights['reliability'] = np.array(rel_w)
        self.factor_weights['security'] = np.array(sec_w)

    def reset_defaults(self):
        """重置为默认权重 (用于计算 Baseline Trust)"""
        self.dim_weights = copy.deepcopy(self.default_dim_weights)
        self.factor_weights = copy.deepcopy(self.default_factor_weights)

    # --- 辅助计算 ---
    def _normalize(self, value, is_positive, limit):
        if is_positive is None: return 1.0 if value else 0.0
        min_v, max_v = limit
        val = max(min_v, min(value, max_v))
        if max_v == min_v: return 0.0
        return (val - min_v) / (max_v - min_v) if is_positive else (max_v - val) / (max_v - min_v)

    def _calc_membership(self, u):
        membership = np.zeros(len(self.V))
        for k, vk in enumerate(self.V):
            if k == 0 and u <= vk:
                membership[k] = 1
            elif k == len(self.V) - 1 and u >= vk:
                membership[k] = 1
            else:
                if k > 0 and self.V[k - 1] < u < vk:
                    membership[k] = (u - self.V[k - 1]) / (vk - self.V[k - 1])
                elif k < len(self.V) - 1 and vk <= u < self.V[k + 1]:
                    membership[k] = (self.V[k + 1] - u) / (self.V[k + 1] - vk)
        return membership

    def _calc_dim_score(self, node, dim):
        config = self.factor_configs[dim]
        weights = self.factor_weights[dim]
        vals = [self._normalize(getattr(node, k, 0), pos, lim) for k, pos, lim in config]
        R = np.array([self._calc_membership(v) for v in vals])
        B = np.dot(weights, R)
        return np.clip(np.dot(B, self.S), 0, 1)

    # --- 核心功能 ---
    def punish_node(self, node, level, reporters, current_time):
        if level not in self.malicious_levels: return
        p_base = self.malicious_levels[level]
        count = node.malicious_counts.get(level, 0) + 1
        node.malicious_counts[level] = count
        m_r = 1 + 0.5 * math.log(count + 1)
        avg_trust = np.mean([r.trust_current for r in reporters]) if reporters else 0.5
        node.R_sec += p_base * m_r * avg_trust
        node.last_malicious_time = current_time

    def calculate_only(self, node, neighbors, current_time) -> float:
        """
        纯计算模式：不更新节点状态，仅返回 trust_total
        用于在 Reward 计算中对比新旧权重的效果
        """
        # print("")
        # if node.Id == "node_0":
        #     print("performance: ", self.dim_weights['performance'])
        #     print("reliability: ", self.dim_weights['reliability'])
        #     print("security: ", self.dim_weights['security'])

        t_perf = self._calc_dim_score(node, 'performance')
        t_rel = self._calc_dim_score(node, 'reliability')
        t_sec_raw = self._calc_dim_score(node, 'security')

        # 临时应用 R_sec 衰减计算
        r_sec_temp = node.R_sec
        if node.R_sec > 0:
            clean_time = max(0, current_time - node.last_malicious_time)
            r_sec_temp *= math.exp(-self.lambda_recover * clean_time)

        t_sec = t_sec_raw * math.exp(-self.eta_penalty * r_sec_temp)

        t_c = (self.dim_weights['performance'] * t_perf +
               self.dim_weights['reliability'] * t_rel +
               self.dim_weights['security'] * t_sec)

        delta_t = max(0, current_time - node.last_interact_time) if node.last_interact_time > 0 else 0
        t_c_prime = t_c * math.exp(-self.lambda_decay * delta_t / (t_c + 1e-6))

        if neighbors:
            t_p = np.mean([n.trust_current for n in neighbors])
        else:
            t_p = 0.5

        t_h = node.trust_history
        if node.is_first_time:
            alpha, beta, gamma = (0.4, 0.0, 0.6) if neighbors else (1.0, 0.0, 0.0)
        else:
            alpha, beta, gamma = 0.6, 0.4, 0.0

        t_total = alpha * t_c_prime + beta * t_h + gamma * t_p
        return max(0.0, min(1.0, t_total))

    def evaluate_node(self, node, neighbors=None, current_time=0) -> float:
        """标准评估模式：会更新节点状态"""
        # 复用计算逻辑，这里简化写法，直接计算并赋值
        val = self.calculate_only(node, neighbors, current_time)

        # 反推 t_c_prime 用于更新 history (简化处理)
        # 实际应保留中间变量，此处为保证代码简洁直接更新 total
        # 注意：在 calc_reward_contrast 中调用 calculate_only 不会改变节点状态

        node.trust_history = (1 - self.delta_history) * node.trust_history + self.delta_history * val
        node.trust_current = val  # 近似
        node.trust_total = val
        node.is_first_time = False
        node.last_interact_time = current_time
        return val