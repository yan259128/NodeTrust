import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli, Dirichlet
import numpy as np
import copy
import env

# --- 神经网络组件 ---
class PPO_ActorCritic(nn.Module):
    def __init__(self, state_dim, node_num):
        super(PPO_ActorCritic, self).__init__()

        # 共享特征提取层
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU()
        )

        # 1. Critic Head: 评估状态价值 V(s)
        self.value_head = nn.Linear(256, 1)

        # 2. Actor Head - 节点选择 (离散动作: Bernoulli)
        self.action_logits = nn.Linear(256, node_num)

        # 3. Actor Head - 权重生成 (连续动作: Dirichlet)
        # Dirichlet 分布需要 alpha 参数 (均需 > 0)，所以使用 Softplus 激活
        self.w_dim_alpha = nn.Sequential(nn.Linear(256, 3), nn.Softplus())
        self.w_perf_alpha = nn.Sequential(nn.Linear(256, 5), nn.Softplus())
        self.w_rel_alpha = nn.Sequential(nn.Linear(256, 4), nn.Softplus())
        self.w_sec_alpha = nn.Sequential(nn.Linear(256, 4), nn.Softplus())

    def forward(self, state):
        feat = self.backbone(state)

        # 价值
        value = self.value_head(feat)

        # 节点选择概率
        prob_select = torch.sigmoid(self.action_logits(feat))

        # 权重分布参数 (加 1.0 是为了保证 alpha > 1，使分布更平滑，防止出现极端的 0 或 1)
        a_dim = self.w_dim_alpha(feat) + 1.0
        a_perf = self.w_perf_alpha(feat) + 1.0
        a_rel = self.w_rel_alpha(feat) + 1.0
        a_sec = self.w_sec_alpha(feat) + 1.0

        return value, prob_select, a_dim, a_perf, a_rel, a_sec


class PPOAgent:
    def __init__(self, state_dim, node_num, device):
        self.device = device
        self.node_num = node_num

        # 超参数
        self.gamma = 0.99
        self.lmbda = 0.95  # GAE 参数
        self.eps_clip = 0.2  # PPO 裁剪范围
        self.K_epochs = 10  # 每次更新迭代次数
        self.entropy_coef = 0.01  # 熵正则化系数，鼓励探索

        self.policy = PPO_ActorCritic(state_dim, node_num).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.policy_old = copy.deepcopy(self.policy)  # 用于计算 Ratio

        # 存储一个 Batch 的经验
        self.buffer = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            val, prob, a_dim, a_perf, a_rel, a_sec = self.policy_old(state_t)

            # 1. 采样节点选择 (离散)
            dist_select = Bernoulli(prob)
            act_select = dist_select.sample()
            log_prob_select = dist_select.log_prob(act_select).sum()

            # 2. 采样四组权重 (连续 - Dirichlet)
            d_dim, d_perf, d_rel, d_sec = Dirichlet(a_dim), Dirichlet(a_perf), Dirichlet(a_rel), Dirichlet(a_sec)
            w_dim, w_perf, w_rel, w_sec = d_dim.sample(), d_perf.sample(), d_rel.sample(), d_sec.sample()

            log_prob_w = d_dim.log_prob(w_dim) + d_perf.log_prob(w_perf) + \
                         d_rel.log_prob(w_rel) + d_sec.log_prob(w_sec)

        # 打包返回给环境
        act_dict = {f'node_{i}': int(act_select[0, i].item()) for i in range(self.node_num)}
        w_pack = {
            'dim': w_dim.cpu().numpy()[0],
            'perf': w_perf.cpu().numpy()[0],
            'rel': w_rel.cpu().numpy()[0],
            'sec': w_sec.cpu().numpy()[0]
        }

        # 记录动作数据用于更新
        action_data = (act_select, w_dim, w_perf, w_rel, w_sec)
        total_log_prob = (log_prob_select + log_prob_w).item()

        return act_dict, w_pack, total_log_prob, val.item(), action_data

    def update(self):
        if len(self.buffer) == 0: return

        # 转换 Buffer 数据
        states = torch.FloatTensor([x[0] for x in self.buffer]).to(self.device)
        # 动作拆解
        old_act_select = torch.cat([x[1][0] for x in self.buffer])
        old_w_dim = torch.cat([x[1][1] for x in self.buffer])
        old_w_perf = torch.cat([x[1][2] for x in self.buffer])
        old_w_rel = torch.cat([x[1][3] for x in self.buffer])
        old_w_sec = torch.cat([x[1][4] for x in self.buffer])

        old_log_probs = torch.FloatTensor([x[2] for x in self.buffer]).to(self.device).view(-1, 1)
        rewards = [x[3] for x in self.buffer]
        masks = [1.0 - x[4] for x in self.buffer]
        values = [x[5] for x in self.buffer]

        # 1. 计算优势函数 (GAE)
        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            # TD Error = r + gamma * V(s') - V(s)
            delta = rewards[i] + self.gamma * (values[i + 1] if i + 1 < len(values) else 0) * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        adv_t = torch.FloatTensor(advantages).to(self.device).view(-1, 1)
        ret_t = torch.FloatTensor(returns).to(self.device).view(-1, 1)
        # 优势归一化
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # 2. 多轮迭代更新
        for _ in range(self.K_epochs):
            val, prob, a_dim, a_perf, a_rel, a_sec = self.policy(states)

            # 计算新的 Log Prob 和 熵
            d_select = Bernoulli(prob)
            d_dim, d_perf, d_rel, d_sec = Dirichlet(a_dim), Dirichlet(a_perf), Dirichlet(a_rel), Dirichlet(a_sec)

            new_log_p_select = d_select.log_prob(old_act_select).sum(dim=1, keepdim=True)
            new_log_p_w = d_dim.log_prob(old_w_dim).view(-1, 1) + d_perf.log_prob(old_w_perf).view(-1, 1) + \
                          d_rel.log_prob(old_w_rel).view(-1, 1) + d_sec.log_prob(old_w_sec).view(-1, 1)
            new_log_probs = new_log_p_select + new_log_p_w

            entropy = d_select.entropy().sum(dim=1, keepdim=True) + d_dim.entropy().view(-1, 1)  # 简化，仅取部分熵

            # PPO 核心: Importance Sampling Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 裁剪损失
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_t

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(val, ret_t) - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 同步旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []


def train_ppo():
    # 1. 初始化
    node_num = 6
    edge_env = env.EdgeEnv(node_num)
    state_dim = node_num * 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPOAgent(state_dim, node_num, device)

    episodes = 100
    print(f"开始 PPO 训练 | 设备: {device}")

    for ep in range(episodes):
        # 2. 场景调度
        if ep < 20:
            edge_env.set_scenario('normal')
        elif ep < 40:
            edge_env.set_scenario('low_perf')
        elif ep < 60:
            edge_env.set_scenario('low_rel')
        elif ep < 80:
            edge_env.set_scenario('low_sec')
        else:
            edge_env.set_scenario('normal')

        s = edge_env.reset()
        ep_reward = 0

        # PPO 收集阶段
        for t in range(50):
            # 3. 采样动作
            act_dict, w_pack, log_p, val, act_data = agent.select_action(s)

            # PPO 默认不输出信任预测，给个默认中性值
            trust_dict_dummy = {f'node_{i}': 0.5 for i in range(node_num)}

            # 4. 执行
            ns, r, d, info = edge_env.step(act_dict, trust_dict_dummy, w_pack, t)

            # 5. 暂存到内存 (On-policy)
            agent.buffer.append((s, act_data, log_p, r, d, val))

            s = ns
            ep_reward += r
            if d: break

        # 6. 更新网络 (一轮 Episode 结束后更新一次)
        agent.update()

        if ep % 5 == 0:
            print(f"PPO Ep {ep} | Reward: {ep_reward:.2f} | Mode: {edge_env.scenario_mode}")


if __name__ == "__main__":
    train_ppo()