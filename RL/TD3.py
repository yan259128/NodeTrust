import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import env  # 导入你的 env.py


# ==========================================================
# 1. 经典模型定义
# ==========================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 标准的三层全连接网络
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # 使用 Sigmoid 将输出限制在 [0, 1]，方便后续映射到动作和权重
        return torch.sigmoid(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 网络
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 网络 (Clipped Double-Q 关键点)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


# ==========================================================
# 2. 经验回放缓冲区
# ==========================================================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )


# ==========================================================
# 3. TD3 算法主体
# ==========================================================
class TD3Agent:
    def __init__(self, state_dim, node_num, device):
        self.device = device
        self.node_num = node_num
        # 总动作维度 = 节点开关(node_num) + 权重(3+5+4+4=16)
        self.action_dim = node_num + 16

        self.actor = Actor(state_dim, self.action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.tau = 0.005
        self.gamma = 0.99
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state, exploration_noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        # 添加探索噪声
        if exploration_noise != 0:
            action = (action + np.random.normal(0, exploration_noise, size=self.action_dim)).clip(0, 1)

        # 拆解动作为 env.py 格式
        # 1. 节点选择 (前 node_num 位)
        selection = (action[:self.node_num] >= 0.5).astype(int)
        act_dict = {f'node_{i}': selection[i] for i in range(self.node_num)}

        # 2. 权重 (后16位) 并进行归一化处理
        w = action[self.node_num:]

        def norm_w(sub_w):
            s = sum(sub_w)
            return sub_w / s if s > 0 else sub_w

        w_pack = {
            'dim': norm_w(w[0:3]),
            'perf': norm_w(w[3:8]),
            'rel': norm_w(w[8:12]),
            'sec': norm_w(w[12:16])
        }

        # 经典 TD3 不做信任预测，传全 0 或默认值
        trust_dict = {f'node_{i}': 0.5 for i in range(self.node_num)}

        return act_dict, trust_dict, w_pack, action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state, action, next_state, reward, not_done = state.to(self.device), action.to(self.device), next_state.to(
            self.device), reward.to(self.device), not_done.to(self.device)

        with torch.no_grad():
            # Target Policy Smoothing: 给目标动作添加噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(0, 1)

            # 计算目标 Q 值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # 更新 Critic
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新 Actor
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


