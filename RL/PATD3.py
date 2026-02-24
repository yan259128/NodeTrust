# main_rl.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import env


class ModernBlock(nn.Module):
    """
    Linear -> LayerNorm -> SiLU
    比传统的 BatchNorm + ReLU 在 RL 中更稳定
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


class SharedEncoder(nn.Module):
    """共享特征提取器"""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            ModernBlock(state_dim, hidden_dim),
            ModernBlock(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, hidden_dim, action_dim, node_num):
        super().__init__()

        # --- Head 1: 节点选择 (Sigmoid) ---
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_dim), nn.Sigmoid()
        )

        # --- Head 2: 信任预测 (Sigmoid) ---
        self.trust_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, node_num), nn.Sigmoid()
        )

        # --- Head 3: 动态权重群 (Linear Logits) ---
        # 注意：这里输出 Logits，具体的 Softmax 在 forward 中处理
        self.w_dim = nn.Linear(hidden_dim, 3)  # Perf, Rel, Sec
        self.w_perf = nn.Linear(hidden_dim, 5)  # Perf 子因子
        self.w_rel = nn.Linear(hidden_dim, 4)  # Rel 子因子
        self.w_sec = nn.Linear(hidden_dim, 4)  # Sec 子因子

        # 最小权重限制 (5%)，防止某项权重归零
        self.min_w = 0.1

        # [关键优化] 初始化 Bias 使得初始输出接近均匀分布
        self._init_bias_to_balanced()

    def _init_bias_to_balanced(self):
        """初始化最后一层，使初始权重趋向于平权 (1/N)，防止初期崩盘"""
        for layer in [self.w_dim, self.w_perf, self.w_rel, self.w_sec]:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)

    def _constrained_softmax(self, logits, num_classes):
        """
        限制性 Softmax 公式:
        Final = min_w + (1 - N * min_w) * Softmax(logits)
        """
        probs = F.softmax(logits, dim=1)
        remain_space = max(0.0, 1.0 - (num_classes * self.min_w))
        return self.min_w + remain_space * probs

    def forward(self, features):
        # 1. 动作与预测
        act = self.action_head(features)
        trust = self.trust_head(features)

        # 2. 权重计算 (应用限制性 Softmax)
        wd = self._constrained_softmax(self.w_dim(features), 3)
        wp = self._constrained_softmax(self.w_perf(features), 5)
        wr = self._constrained_softmax(self.w_rel(features), 4)
        ws = self._constrained_softmax(self.w_sec(features), 4)

        return act, trust, wd, wp, wr, ws


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, node_num):
        super().__init__()
        # Input Dim = State + Action + Trust_Pred + Weights(3+5+4+4=16)
        input_dim = state_dim + action_dim + node_num + 16

        # Double Q-Learning
        self.q1 = nn.Sequential(
            ModernBlock(input_dim, 256),
            ModernBlock(256, 256),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            ModernBlock(input_dim, 256),
            ModernBlock(256, 256),
            nn.Linear(256, 1)
        )

    def forward(self, state, action, trust, w_flat):
        # 拼接所有输入
        xu = torch.cat([state, action, trust, w_flat], 1)
        return self.q1(xu), self.q2(xu)

    def Q1(self, state, action, trust, w_flat):
        xu = torch.cat([state, action, trust, w_flat], 1)
        return self.q1(xu)


class PATD3Agent:
    def __init__(self, state_dim, action_dim, node_num, device):
        self.device = device
        self.node_num = node_num

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.1
        self.noise_clip = 0.3
        self.policy_freq = 2
        self.trust_coef = 0.5  # 信任预测辅助任务的权重
        self.dirichlet_alpha = 0.6  # 狄利克雷分布参数 (越小越极端，越大越平均)

        # Networks
        self.encoder = SharedEncoder(state_dim).to(device)
        self.encoder_target = copy.deepcopy(self.encoder)

        self.actor = Actor(256, action_dim, node_num).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, node_num).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers (AdamW for better regularization)
        self.actor_optim = optim.AdamW(list(self.encoder.parameters()) + list(self.actor.parameters()), lr=3e-4)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=3e-4)

        self.total_it = 0

    def select_action(self, state, noise_scale=0.1):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.encoder(state_t)
            a, t, wd, wp, wr, ws = self.actor(feat)

            # 转 Numpy
            a = a.cpu().numpy()[0]
            t = t.cpu().numpy()[0]
            wd = wd.cpu().numpy()[0]
            wp = wp.cpu().numpy()[0]
            wr = wr.cpu().numpy()[0]
            ws = ws.cpu().numpy()[0]

        # --- 混合噪声策略 ---
        # 1. 狄利克雷探索 (针对权重)
        def dirichlet_exploration(w, noise_lvl):
            if noise_lvl <= 0: return w
            # 生成符合分布的噪声
            dir_noise = np.random.dirichlet([self.dirichlet_alpha] * len(w))
            # 线性插值混合
            w_new = (1 - noise_lvl) * w + noise_lvl * dir_noise
            # 再次保底限制 (5%)
            w_new = np.maximum(0.1, w_new)
            return w_new / w_new.sum()

        # 2. 高斯探索 (针对节点选择)
        if noise_scale > 0:
            a = np.clip(a + np.random.normal(0, noise_scale, size=a.shape), 0, 1)

            wd = dirichlet_exploration(wd, noise_scale)
            wp = dirichlet_exploration(wp, noise_scale)
            wr = dirichlet_exploration(wr, noise_scale)
            ws = dirichlet_exploration(ws, noise_scale)

        disc_act = (a >= 0.5).astype(int)

        # 打包返回
        w_pack = {'dim': wd, 'perf': wp, 'rel': wr, 'sec': ws}
        w_flat = np.concatenate([wd, wp, wr, ws])

        return disc_act, a, t, w_pack, w_flat

    def train(self, batch):
        self.total_it += 1
        s, a, r, ns, d, t_lbl, w_fl = batch

        # Tensor Conversion
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        t_lbl = torch.FloatTensor(t_lbl).to(self.device)
        w_fl = torch.FloatTensor(w_fl).to(self.device)

        # ----------------------------
        # 1. Critic Update
        # ----------------------------
        with torch.no_grad():
            n_feat = self.encoder_target(ns)
            # 获取 Target Actor 的所有输出
            na, nt, nwd, nwp, nwr, nws = self.actor_target(n_feat)
            nw_flat = torch.cat([nwd, nwp, nwr, nws], 1)

            # Action Smoothing (仅对 Action 加噪声，权重通常由 Softmax 处理已足够)
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (na + noise).clamp(0, 1)

            # Target Q
            tq1, tq2 = self.critic_target(ns, na, nt, nw_flat)
            target_q = r + (1 - d) * self.gamma * torch.min(tq1, tq2)

        # Current Q
        feat = self.encoder(s)
        _, curr_t, _, _, _, _ = self.actor(feat)  # 只用当前的 Trust 预测

        # Critic 输入 Buffer 中的实际权重 (w_fl)
        cq1, cq2 = self.critic(s, a, curr_t.detach(), w_fl)
        critic_loss = F.mse_loss(cq1, target_q) + F.mse_loss(cq2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----------------------------
        # 2. Actor Update (Delayed)
        # ----------------------------
        if self.total_it % self.policy_freq == 0:
            feat = self.encoder(s)
            pa, pt, pwd, pwp, pwr, pws = self.actor(feat)
            pw_flat = torch.cat([pwd, pwp, pwr, pws], 1)

            # Maximize Q-Value
            q_val = self.critic.Q1(s, pa, pt, pw_flat)
            actor_loss = -q_val.mean()

            # Minimize Trust Prediction Error
            trust_loss = F.mse_loss(pt, t_lbl)

            total_loss = actor_loss + self.trust_coef * trust_loss

            self.actor_optim.zero_grad()
            total_loss.backward()
            self.actor_optim.step()

            # Soft Updates
            self.soft_update(self.encoder, self.encoder_target)
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class PrioritizedBuffer:
    def __init__(self):
        self.storage = []
        self.ptr = 0
        self.max_size = 50000

    def add(self, data, priority=False):
        """
        如果 priority=True (高价值样本)，则存储多次，变相增加采样概率
        """
        repeat = 3 if priority else 1

        for _ in range(repeat):
            if len(self.storage) < self.max_size:
                self.storage.append(data)
            else:
                self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in ind]
        return map(np.stack, zip(*batch))

    def reset(self):
        self.__init__()



if __name__ == "__main__":
    # 配置
    node_num = 6
    env = env.EdgeEnv(node_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 Agent
    # State Dim = Node_Num * 16 (Features per node)
    agent = PATD3Agent(node_num * 16, node_num, node_num, device)

    # 初始化 Buffer
    buffer = PrioritizedBuffer()

    print(f"--- Training Start on {device} ---")
    print(f"--- Strategy: Dirichlet Exploration + Exp Contrast Reward ---")

    episodes = 100
    for ep in range(episodes):
        # 1. 场景调度 (Curriculum / Environment Shift)
        # 每20轮切换一次场景，强迫 Agent 学习适应性
        if ep < 20:
            env.set_scenario('normal')
        elif ep < 40:
            env.set_scenario('low_perf')  # 性能环境恶化
        elif ep < 60:
            env.set_scenario('low_rel')  # 可靠性环境恶化
        elif ep < 80:
            env.set_scenario('low_sec')  # 安全性环境恶化
        else:
            env.set_scenario('normal')

        s = env.reset()
        ep_reward = 0

        # 2. 动态噪声衰减
        # 前期 noise_scale 大(0.3)，后期小(0.05)
        expl_noise = max(0.05, 0.3 * (0.96 ** ep))

        # 统计本轮的信任值增益
        trust_gains = []

        for t in range(50):  # 每一轮 50 steps
            # Select Action
            disc_act, raw_act, trust, w_pack, w_flat = agent.select_action(s, noise_scale=expl_noise)

            act_dict = {f'node_{i}': disc_act[i] for i in range(node_num)}
            trust_dict = {f'node_{i}': trust[i] for i in range(node_num)}

            # Step Env
            ns, r, d, info = env.step(act_dict, trust_dict, w_pack, t)

            # 记录本步选中节点的信任提升 (Adaptive - Baseline)
            for nid, node in env.nodes.items():
                if node.is_select:
                    trust_gains.append(node.temp_adaptive_trust - node.trust_baseline)

            trust_lbl = [info['trust_totals'][f'node_{i}'] for i in range(node_num)]

            # [关键] 高价值样本识别
            # 如果奖励显著(>20)，说明这组权重效果很好，标记为优先
            is_high_value = (r > 20.0)

            buffer.add((s, raw_act, r, ns, d, trust_lbl, w_flat), priority=is_high_value)

            # Train
            if len(buffer.storage) > 200:
                agent.train(buffer.sample(64))

            s = ns
            ep_reward += r
            if d: break

        # --- 结果日志 ---
        avg_gain = np.mean(trust_gains) if trust_gains else 0.0
        print(
            f"Ep {ep:<3} | Mode: {env.scenario_mode:<9} | Reward: {ep_reward:>7.1f} | Gain: {avg_gain:+.4f} | Noise: {expl_noise:.2f}")

        # 每10轮打印一次详细权重，用于 Debug 观察 Agent 策略
        if ep % 10 == 0:
            print(f"  > Dim Weights: {w_pack['dim'].round(2)}")
            if env.scenario_mode == 'low_perf':
                print(f"  > Perf Sub-W : {w_pack['perf'].round(2)}")
            elif env.scenario_mode == 'low_rel':
                print(f"  > Rel Sub-W  : {w_pack['rel'].round(2)}")
            elif env.scenario_mode == 'low_sec':
                print(f"  > Sec Sub-W  : {w_pack['sec'].round(2)}")