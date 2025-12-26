# calc_trust.py
import math
import numpy as np
import copy


class TrustEvaluator:
    """
    基于模糊综合评价的动态信任评估模型
    支持 RL 动态权重调整
    """

    def __init__(self):
        # 1. 因子配置 (属性名, 正向指标True/反向False/布尔None, (min, max))
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

        # 4. 评价集与模型参数
        self.V = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.S = self.V
        self.lambda_decay = 0.1  # 时间衰减因子
        self.eta_penalty = 0.05  # 惩罚影响因子
        self.lambda_recover = 0.05  # 信誉恢复因子
        self.delta_history = 0.3  # 历史信任平滑权重 (Alpha for EMA)

        # 恶意行为等级库
        self.malicious_levels = {'L1': 10, 'L2': 30, 'L3': 70, 'L4': 100}

    # ==========================================
    # 权重管理接口 (供 RL Agent 调用)
    # ==========================================
    def _safe_normalize(self, weights, min_limit=0.01):
        """内部工具：确保权重不低于下限且和为1"""
        weights = np.array(weights)
        weights = np.maximum(min_limit, weights)
        return weights / weights.sum()

    def update_all_weights(self, dim_w, perf_w, rel_w, sec_w):
        """应用 RL Agent 生成的动态权重"""
        # 维度权重
        self.dim_weights['performance'] = dim_w[0]
        self.dim_weights['reliability'] = dim_w[1]
        self.dim_weights['security'] = dim_w[2]

        # 子因子权重 (强制安全归一化)
        self.factor_weights['performance'] = self._safe_normalize(perf_w)
        self.factor_weights['reliability'] = self._safe_normalize(rel_w)
        self.factor_weights['security'] = self._safe_normalize(sec_w)

    def reset_defaults(self):
        """重置为默认权重 (用于计算 Baseline Trust)"""
        self.dim_weights = copy.deepcopy(self.default_dim_weights)
        self.factor_weights = copy.deepcopy(self.default_factor_weights)

    # ==========================================
    # 模糊计算辅助方法
    # ==========================================
    def _normalize(self, value, is_positive, limit):
        if is_positive is None: return 1.0 if value else 0.0
        min_v, max_v = limit
        val = max(min_v, min(value, max_v))
        if max_v == min_v: return 0.0
        return (val - min_v) / (max_v - min_v) if is_positive else (max_v - val) / (max_v - min_v)

    def _calc_membership(self, u):
        """向量化计算三角隶属度"""
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
        """计算单维度的模糊综合评分"""
        config = self.factor_configs[dim]
        weights = self.factor_weights[dim]
        # 1. 归一化
        vals = [self._normalize(getattr(node, k, 0), pos, lim) for k, pos, lim in config]
        # 2. 模糊矩阵 R
        R = np.array([self._calc_membership(v) for v in vals])
        # 3. 综合评价 B = W * R
        B = np.dot(weights, R)
        # 4. 解模糊化 (重心法/加权平均)
        return np.clip(np.dot(B, self.S), 0, 1)

    # ==========================================
    # 核心评估逻辑
    # ==========================================
    def punish_node(self, node, level, reporters, current_time):
        """触发惩罚：增加安全信誉账本 R_sec"""
        if level not in self.malicious_levels: return
        p_base = self.malicious_levels[level]

        # 累犯加重
        count = node.malicious_counts.get(level, 0) + 1
        node.malicious_counts[level] = count
        m_r = 1 + 0.5 * math.log(count + 1)

        # 举报者信誉加权
        avg_trust = np.mean([r.trust_current for r in reporters]) if reporters else 0.5

        # 更新账本
        delta = p_base * m_r * avg_trust
        node.R_sec += delta
        node.last_malicious_time = current_time

    def evaluate_node(self, node, neighbors=None, current_time=0) -> float:
        """
        [完整版] 计算信任值并更新节点状态
        包含：R_sec衰减更新、历史信任EMA更新、时间戳更新
        """
        neighbors = neighbors or []

        # 1. 计算各维度原始信任值
        t_perf = self._calc_dim_score(node, 'performance')
        t_rel = self._calc_dim_score(node, 'reliability')
        t_sec_raw = self._calc_dim_score(node, 'security')

        # 2. 安全信誉账本 (R_sec) 处理
        # [副作用] 这里会真实更新节点的 R_sec 值 (随时间洗白)
        if node.R_sec > 0:
            clean_time = max(0, current_time - node.last_malicious_time)
            node.R_sec *= math.exp(-self.lambda_recover * clean_time)
            # 防止浮点数拖尾
            if node.R_sec < 0.01: node.R_sec = 0.0

        # 应用惩罚
        t_sec = t_sec_raw * math.exp(-self.eta_penalty * node.R_sec)

        # 3. 综合当前直接信任 Tc
        # 使用当前时刻的动态权重 self.dim_weights
        t_c = (self.dim_weights['performance'] * t_perf +
               self.dim_weights['reliability'] * t_rel +
               self.dim_weights['security'] * t_sec)

        # 4. 时间衰减 Tc'
        # 距离上次交互时间越久，当前表现的可信度越低
        delta_t = max(0, current_time - node.last_interact_time) if node.last_interact_time > 0 else 0
        decay = math.exp(-self.lambda_decay * delta_t / (t_c + 1e-6))
        t_c_prime = t_c * decay

        # 5. 历史信任 Th
        t_h = node.trust_history

        # 6. 推荐信任 Tp (Neighbors)
        if neighbors:
            t_p = np.mean([n.trust_current for n in neighbors])
        else:
            t_p = 0.5

        # 7. 聚合最终信任 T_total
        if node.is_first_time:
            # 首次加入，更加依赖推荐
            if neighbors:
                alpha, beta, gamma = 0.4, 0.0, 0.6
            else:
                alpha, beta, gamma = 1.0, 0.0, 0.0
        else:
            # 稳定运行，依赖当前和历史
            alpha, beta, gamma = 0.6, 0.4, 0.0

        t_total = alpha * t_c_prime + beta * t_h + gamma * t_p

        # 边界钳制
        t_total = max(0.0, min(1.0, t_total))

        # 8. [副作用] 更新节点状态
        # 使用 EMA 更新历史信任
        node.trust_history = (1 - self.delta_history) * node.trust_history + self.delta_history * t_total
        node.trust_current = t_c_prime  # 记录衰减后的当前值
        node.trust_total = t_total
        node.is_first_time = False
        node.last_interact_time = current_time

        return round(t_total, 4)

    def calculate_only(self, node, neighbors=None, current_time=0) -> float:
        """
        [无副作用版] 仅计算信任值，不更新节点任何状态
        用于 RL 计算 Baseline Reward 时对比使用
        """
        neighbors = neighbors or []

        # 1. 维度计算
        t_perf = self._calc_dim_score(node, 'performance')
        t_rel = self._calc_dim_score(node, 'reliability')
        t_sec_raw = self._calc_dim_score(node, 'security')

        # 2. R_sec 临时计算 (不修改 node.R_sec)
        temp_r_sec = node.R_sec
        if temp_r_sec > 0:
            clean_time = max(0, current_time - node.last_malicious_time)
            temp_r_sec *= math.exp(-self.lambda_recover * clean_time)

        t_sec = t_sec_raw * math.exp(-self.eta_penalty * temp_r_sec)

        # 3. Tc
        t_c = (self.dim_weights['performance'] * t_perf +
               self.dim_weights['reliability'] * t_rel +
               self.dim_weights['security'] * t_sec)

        # 4. Tc' (不修改 last_interact_time)
        delta_t = max(0, current_time - node.last_interact_time) if node.last_interact_time > 0 else 0
        decay = math.exp(-self.lambda_decay * delta_t / (t_c + 1e-6))
        t_c_prime = t_c * decay

        # 5. Th
        t_h = node.trust_history

        # 6. Tp
        if neighbors:
            t_p = np.mean([n.trust_current for n in neighbors])
        else:
            t_p = 0.5

        # 7. Total
        if node.is_first_time:
            if neighbors:
                alpha, beta, gamma = 0.4, 0.0, 0.6
            else:
                alpha, beta, gamma = 1.0, 0.0, 0.0
        else:
            alpha, beta, gamma = 0.6, 0.4, 0.0

        t_total = alpha * t_c_prime + beta * t_h + gamma * t_p
        return max(0.0, min(1.0, t_total))