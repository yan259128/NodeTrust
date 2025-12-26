import time
import random
import numpy as np

# 尝试导入工具包，如果不存在则使用模拟数据，保证代码独立运行不报错
try:
    import util.util as util
except ImportError:
    class MockUtil:
        @staticmethod
        def get_ip(flag): return "192.168.1.100"


    util = MockUtil()


class Node:
    """
    区块链边缘节点结构定义
    适配模型：基于模糊综合评价与安全信誉账本的动态信任评估
    """

    def __init__(self, node_id, port, join_round=0):
        # --- 1. 基础网络标识 ---
        self.Id = node_id
        self.ip = util.get_ip(False)
        self.port = port
        self.join_round = join_round  # 加入轮次
        self.is_select = False  # 是否被选中参与共识/任务

        # --- 2. 信任状态体系 (核心适配部分) ---
        # 当前信任值 Tc (Current)
        self.trust_current = 0.5
        # 历史信任值 Th (History) - 数学模型主要使用标量进行EMA计算
        self.trust_history = 0.0
        # 历史信任记录列表 (用于审计或可视化趋势)
        self.trust_history_list = []
        # 预测信任值 Tp (Prediction)
        self.trust_prediction = 0.0
        # RL 预测值
        self.trust_rl = 0.0
        # 总信任值 Ttotal
        self.trust_total = 0.5
        # 基础计算值
        self.trust_baseline = 0.0

        # --- 3. 动态权重与状态标记 ---
        self.is_first_time = True  # 标记是否首次参与 (影响 alpha, beta, gamma 权重)
        self.last_interact_time = time.time()  # 上次交互时间 (用于时间衰减 Tc')

        # --- 4. 安全信誉账本 (Security Ledger) ---
        self.R_sec = 0.0  # 安全信誉账本惩罚累计值
        self.malicious_counts = {}  # 恶意行为计数器 {'L1': 2, 'L3': 1}
        self.last_malicious_time = 0  # 上次被检测出恶意行为的时间 (用于信誉恢复)

        # --- 5. 性能指标 (Performance) ---
        # 范围需与 TrustEvaluator.factor_configs 一致
        self.Response_Time = np.random.randint(20, 300)  # ms (越小越好)
        self.Request_Success_Rate = round(np.random.uniform(90, 100), 2)  # % (越大越好)
        self.Throughput = np.random.randint(300, 800)  # req/s
        self.bandwidth = np.random.randint(100, 1000)  # MB/s
        self.handle = np.random.randint(10, 100)  # TOPS (处理能力)

        # --- 6. 可靠性指标 (Reliability) ---
        self.task_completion_rate = round(np.random.uniform(90, 100), 2)  # %
        self.survival_rate = round(np.random.uniform(95, 100), 2)  # 在线率 %
        self.verify_transaction_nums = 0  # 验证交易次数 (累加)
        self.service_quality = round(np.random.uniform(7.0, 9.5), 2)  # 评分 (0-10)

        # --- 7. 安全性指标 (Security) ---
        self.is_identification = True  # 身份验证 (Bool)
        self.leakage = 0.0  # 数据泄露风险 % (越小越好)
        self.attack_rate = 0  # 被攻击频率 (次/周期)
        self.user_get_data_rate = round(np.random.uniform(90, 100), 2)  # 查询成功率 %

        # 辅助计数器
        self._batch_counter = 0

    def update_transaction_stats(self, is_valid=True):
        """
        更新交易验证统计
        对应指标：verify_transaction_nums, service_quality
        """
        self.verify_transaction_nums += 1
        self._batch_counter += 1

        # 模拟服务质量随验证次数缓慢提升
        if self._batch_counter >= 10:
            change = 0.02 if is_valid else -0.05
            self.service_quality = min(10.0, max(0.0, self.service_quality + change))
            self.service_quality = round(self.service_quality, 2)
            self._batch_counter = 0

    def update_history_trust(self, new_total_trust):
        """
        更新历史信任值
        包含：EMA平滑更新 和 列表记录
        """
        # 1. 更新列表 (保留最近10次)
        self.trust_history_list.append(self.trust_total)
        if len(self.trust_history_list) > 10:
            self.trust_history_list.pop(0)

        # 2. 更新标量 Th (由 TrustEvaluator 控制具体 EMA 逻辑，这里仅做状态同步)
        # 实际计算通常在 Evaluator 中完成，这里存储结果
        self.trust_history = getattr(self, 'trust_history', 0)

    def simulate_dynamic_change(self):
        """
        模拟节点运行过程中的指标波动 (用于仿真环境)
        """
        # 1. 模拟身份验证状态 (大概率保持 True)
        # self.is_identification = np.random.choice([True, False], p=[0.98, 0.02])

        # 2. 模拟网络抖动导致的存活率/完成率变化
        change_step = random.uniform(-1.5, 1.5)

        # 限制范围辅助函数
        def clamp(val, min_v, max_v):
            return max(min_v, min(val, max_v))

        self.survival_rate = clamp(self.survival_rate + change_step, 0, 100)

        # 联动逻辑：如果存活率下降，任务完成率和成功率通常也会受到影响
        if change_step < 0:
            self.task_completion_rate = clamp(self.task_completion_rate + change_step, 0, 100)
            self.Request_Success_Rate = clamp(self.Request_Success_Rate + change_step, 0, 100)
            self.user_get_data_rate = clamp(self.user_get_data_rate + change_step, 0, 100)
        else:
            # 恢复逻辑
            self.task_completion_rate = clamp(self.task_completion_rate + 0.5, 0, 100)
            self.Request_Success_Rate = clamp(self.Request_Success_Rate + 0.5, 0, 100)

        # 3. 模拟响应时间波动
        self.Response_Time = max(1, self.Response_Time + int(random.gauss(0, 20)))

    def get_feature_vector(self):
        """
        获取用于模糊评价的特征向量
        顺序必须与 TrustEvaluator 中的处理顺序一致
        """
        return {
            'performance': [self.Response_Time, self.Request_Success_Rate, self.Throughput, self.bandwidth,
                            self.handle],
            'reliability': [self.task_completion_rate, self.survival_rate, self.verify_transaction_nums,
                            self.service_quality],
            'security': [self.is_identification, self.leakage, self.attack_rate, self.user_get_data_rate]
        }

    def __str__(self):
        return (f"Node-{self.Id} [Trust: {self.trust_total:.4f} | R_sec: {self.R_sec:.2f} | "
                f"Perf: {self.Response_Time}ms | Rel: {self.survival_rate}%]")