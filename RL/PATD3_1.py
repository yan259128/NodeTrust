import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import env


class SelfAttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class ModernBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def forward(self, x): return self.net(x)


class DecoupledActor(nn.Module):
    def __init__(self, node_num, feat_dim, action_dim):
        super().__init__()
        self.node_num = node_num
        self.feat_dim = feat_dim

        # 1. 特征提取器 (Attention)
        # 输入假设: [Batch, Node_Num, Feat_Dim]
        self.encoder = SelfAttentionBlock(feat_dim)

        # Flatten后维度
        self.flat_dim = node_num * feat_dim

        # 2. Policy Stream (负责节点选择)
        self.policy_net = nn.Sequential(
            ModernBlock(self.flat_dim, 256),
            ModernBlock(256, 256),
            nn.Linear(256, action_dim),
            nn.Sigmoid()  # 0-1 选择概率
        )

        # 3. Weight Stream (负责参数自适应)
        # 使用 Gumbel-Softmax 的前置 Logits
        self.weight_net = nn.Sequential(
            ModernBlock(self.flat_dim, 256),
            ModernBlock(256, 128)
        )
        self.head_dim = nn.Linear(128, 3)  # Perf, Rel, Sec
        self.head_perf = nn.Linear(128, 5)
        self.head_rel = nn.Linear(128, 4)
        self.head_sec = nn.Linear(128, 4)

        # 4. Trust Stream (辅助任务)
        # 独立分支，防止梯度干扰 Policy
        self.trust_net = nn.Sequential(
            ModernBlock(self.flat_dim, 128),
            nn.Linear(128, node_num),
            nn.Sigmoid()
        )


    def forward(self, state, temperature=1.0):
        # State Reshape: [B, N*F] -> [B, N, F]
        batch_size = state.shape[0]
        state_reshaped = state.view(batch_size, self.node_num, self.feat_dim)

        # Feature Extraction
        features = self.encoder(state_reshaped)
        features_flat = features.reshape(batch_size, -1)

        # A. Policy Output
        action = self.policy_net(features_flat)

        # B. Weights Output (Gumbel Softmax for exploration)
        w_feat = self.weight_net(features_flat)

        # 使用 Gumbel-Softmax 使得权重分布更 smooth 且可导
        # training=True 时引入随机性，eval 时输出确定性 Softmax
        w_dim = F.gumbel_softmax(self.head_dim(w_feat), tau=temperature, hard=False, dim=1)
        w_perf = F.gumbel_softmax(self.head_perf(w_feat), tau=temperature, hard=False, dim=1)
        w_rel = F.gumbel_softmax(self.head_rel(w_feat), tau=temperature, hard=False, dim=1)
        w_sec = F.gumbel_softmax(self.head_sec(w_feat), tau=temperature, hard=False, dim=1)

        # C. Trust Prediction (Detach Encoder gradients here if needed)
        # 这里的 detach 视情况而定，如果想让 Trust Loss 更新 Encoder，则不 detach
        # 通常保留 Encoder 更新有助于学习状态表示
        trust_pred = self.trust_net(features_flat)

        return action, trust_pred, w_dim, w_perf, w_rel, w_sec


class RobustCritic(nn.Module):
    def __init__(self, state_dim, action_dim, node_num):
        super().__init__()
        # Input: State + Action + Trust_Pred + Weights(16)
        # Weights flat dim = 3 + 5 + 4 + 4 = 16
        input_dim = state_dim + action_dim + node_num + 16

        # Q1
        self.q1 = nn.Sequential(
            ModernBlock(input_dim, 256),
            ModernBlock(256, 256),
            nn.Linear(256, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            ModernBlock(input_dim, 256),
            ModernBlock(256, 256),
            nn.Linear(256, 1)
        )

    def forward(self, state, action, trust, w_flat):
        xu = torch.cat([state, action, trust, w_flat], 1)
        return self.q1(xu), self.q2(xu)

    def Q1(self, state, action, trust, w_flat):
        xu = torch.cat([state, action, trust, w_flat], 1)
        return self.q1(xu)


class PATD3Agent:
    def __init__(self, node_num, feat_dim, device):
        self.device = device
        self.node_num = node_num
        self.feat_dim = feat_dim
        self.state_dim = node_num * feat_dim
        self.action_dim = node_num

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_freq = 2

        # 降低 Trust Loss 权重，防止主任务被喧宾夺主
        self.trust_coef = 0.2

        # 探索温度 (Gumbel-Softmax)
        self.temp = 1.0
        self.min_temp = 0.1
        self.temp_decay = 0.995

        self.actor = DecoupledActor(node_num, feat_dim, self.action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=3e-4, weight_decay=1e-4)

        self.critic = RobustCritic(self.state_dim, self.action_dim, node_num).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=3e-4, weight_decay=1e-4)

        self.total_it = 0

    def select_action(self, state, noise_scale=0.1, training=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            # Eval 时 temperature 设小，接近 argmax
            curr_temp = self.temp if training else 0.01
            a, t, wd, wp, wr, ws = self.actor(state_t, temperature=curr_temp)

            a = a.cpu().numpy()[0]
            t = t.cpu().numpy()[0]
            wd = wd.cpu().numpy()[0]
            wp = wp.cpu().numpy()[0]
            wr = wr.cpu().numpy()[0]
            ws = ws.cpu().numpy()[0]

        self.actor.train()

        # Action Noise (仅针对节点选择)
        if training and noise_scale > 0:
            a = np.clip(a + np.random.normal(0, noise_scale, size=a.shape), 0, 1)

        disc_act = (a >= 0.5).astype(int)

        # 权重不需要加噪声，Gumbel-Softmax 已经处理了探索
        w_pack = {'dim': wd, 'perf': wp, 'rel': wr, 'sec': ws}
        w_flat = np.concatenate([wd, wp, wr, ws])

        return disc_act, a, t, w_pack, w_flat

    def train(self, batch):
        self.total_it += 1
        s, a, r, ns, d, t_lbl, w_fl = batch

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        t_lbl = torch.FloatTensor(t_lbl).to(self.device)
        w_fl = torch.FloatTensor(w_fl).to(self.device)  # 来自 Buffer 的旧权重

        # ----------------------------
        # 1. Critic Update
        # ----------------------------
        with torch.no_grad():
            # Target Policy Smoothing (仅对 Action)
            noise = (torch.randn_like(a) * 0.2).clamp(-0.5, 0.5)

            # Target Actor Forward
            na, nt, nwd, nwp, nwr, nws = self.actor_target(ns, temperature=0.01)  # Target 用确定性输出
            na = (na + noise).clamp(0, 1)
            nw_flat = torch.cat([nwd, nwp, nwr, nws], 1)

            tq1, tq2 = self.critic_target(ns, na, nt, nw_flat)
            target_q = r + (1 - d) * self.gamma * torch.min(tq1, tq2)

        # 为了反向传播 Actor 的梯度，这里必须用 detach 的 trust
        with torch.no_grad():
            _, pred_t_detached, _, _, _, _ = self.actor(s, temperature=self.temp)

        current_q1, current_q2 = self.critic(s, a, pred_t_detached, w_fl)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----------------------------
        # 2. Actor Update
        # ----------------------------
        if self.total_it % self.policy_freq == 0:
            # 重新计算所有输出
            pa, pt, pwd, pwp, pwr, pws = self.actor(s, temperature=self.temp)
            pw_flat = torch.cat([pwd, pwp, pwr, pws], 1)

            # Maximize Q-Value
            actor_q = self.critic.Q1(s, pa, pt, pw_flat)
            policy_loss = -actor_q.mean()

            # Trust Prediction Loss
            trust_loss = F.mse_loss(pt, t_lbl)

            # --- [新增] 权重熵正则化 (防止固化) ---
            # 我们希望权重分布不要太早变成 One-Hot，保持一定的探索性
            # 计算所有 Softmax 输出的熵
            dist_d = torch.distributions.Categorical(probs=pwd)
            dist_p = torch.distributions.Categorical(probs=pwp)
            dist_r = torch.distributions.Categorical(probs=pwr)
            dist_s = torch.distributions.Categorical(probs=pws)

            # 熵越大，分布越均匀；我们希望最大化熵(加负号minimize -entropy)
            # 系数 entropy_coef 随着训练衰减，初期大后期小
            entropy = dist_d.entropy().mean() + dist_p.entropy().mean() + \
                      dist_r.entropy().mean() + dist_s.entropy().mean()

            entropy_loss = -0.05 * entropy  # 0.05 是正则化系数

            # 总 Loss
            total_loss = policy_loss + self.trust_coef * trust_loss + entropy_loss

            self.actor_optim.zero_grad()
            total_loss.backward()
            self.actor_optim.step()

            # Soft Updates
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

            # Temperature Decay
            self.temp = max(self.min_temp, self.temp * self.temp_decay)

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ==========================================
# 5. 训练主循环
# ==========================================
class PrioritizedBuffer:
    def __init__(self, max_size=100000):
        self.storage = []
        self.ptr = 0
        self.max_size = max_size
        self.size = 0

    def add(self, data):
        if self.size < self.max_size:
            self.storage.append(data)
            self.size += 1
        else:
            self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        batch = [self.storage[i] for i in ind]
        return map(np.stack, zip(*batch))


if __name__ == "__main__":
    node_num = 6
    # 假设特征维度是 16 (根据你的env.py _get_node_feature 确定)
    feat_dim = 16

    env_instance = env.EdgeEnv(node_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PATD3Agent(node_num, feat_dim, device)
    buffer = PrioritizedBuffer()

    episodes = 200
    for ep in range(episodes):
        # 课程学习：环境切换逻辑
        current_scenario = 'normal'
        if ep < 50:
            current_scenario = 'normal'
        elif ep < 100:
            current_scenario = 'low_perf'
        elif ep < 150:
            current_scenario = 'low_rel'
        elif ep < 200:
            current_scenario = 'low_sec'

        # [关键] 检测环境是否切换
        if env_instance.scenario_mode != current_scenario:
            print(f"\n[!!!] Scenario Shift: {env_instance.scenario_mode} -> {current_scenario}")
            print("[!!!] Clearing Replay Buffer to prevent Experience Bias...")

            # 1. 切换环境
            env_instance.set_scenario(current_scenario)

            # 2. 清空 Buffer
            # 这样 Agent 就不会拿 Normal 模式下的"好数据"来训练 Low_Sec 模式
            buffer = PrioritizedBuffer()

            # 3. 重置探索噪声 (给一点新的探索机会)
            expl_noise = 0.3

            # 4. 重置温度 (让权重重新流动)
            agent.temp = 1.0

        s = env_instance.reset()
        ep_reward = 0
        expl_noise = max(0.05, 0.3 * (0.98 ** ep))

        # --- [新增] 用于统计本轮平均值的累加器 ---
        ep_avg_pred_trust = []  # 记录每一步的 平均预测信任值
        ep_avg_selected_trust = []  # 记录每一步的 选中节点的实际信任值

        for t in range(50):
            # 1. 获取动作和权重
            disc_act, raw_act, pred_trust, w_pack, w_flat = agent.select_action(
                s, noise_scale=expl_noise, training=True
            )

            act_dict = {f'node_{i}': disc_act[i] for i in range(node_num)}
            trust_pred_dict = {f'node_{i}': pred_trust[i] for i in range(node_num)}  # <--- 修复点

            # 2. 传给环境
            ns, r, d, info = env_instance.step(act_dict, trust_pred_dict, w_pack, t)

            # 3. [新增] 计算评估指标
            # (A) 平均预测可信度 (Agent 觉得大家有多好)
            step_avg_pred = np.mean(pred_trust)
            ep_avg_pred_trust.append(step_avg_pred)

            # (B) 选中节点的平均实际可信度 (选出来的人到底好不好)
            # 从 info 中获取当前环境计算出的适应性信任值 (Adaptive Trust)
            real_trust_values = [info['trust_totals'][f'node_{i}'] for i in range(node_num)]

            # 找到被选中节点的索引 (disc_act 为 1 的位置)
            selected_indices = [i for i, x in enumerate(disc_act) if x == 1]
            # print(selected_indices)

            if len(selected_indices) > 0:
                # 计算选中节点的实际信任均值
                selected_real_trust = np.mean([real_trust_values[i] for i in selected_indices])
            else:
                selected_real_trust = 0.0  # 如果没选中任何人

            ep_avg_selected_trust.append(selected_real_trust)

            # 4. 存入 Buffer
            trust_lbl = real_trust_values  # 使用列表作为 Label

            # Reward Scaling (提升稳定性)
            r_scaled = r / 10.0

            buffer.add((s, raw_act, r_scaled, ns, d, trust_lbl, w_flat))

            if buffer.size > 256:
                agent.train(buffer.sample(64))

            s = ns
            ep_reward += r
            if d: break

        # --- 计算本 Episode 的统计数据 ---
        final_pred_score = np.mean(ep_avg_pred_trust)
        final_select_score = np.mean(ep_avg_selected_trust)

        # --- 打印结果 ---
        # Pred: Agent预测的平均分 | Real(Sel): 选中节点的真实平均分
        print(f"Ep {ep:<3} | Mode: {env_instance.scenario_mode:<9} | Reward: {ep_reward:>7.1f} | "
              f"Pred: {final_pred_score:.3f} | Real(Sel): {final_select_score:.3f} | Temp: {agent.temp:.2f}")