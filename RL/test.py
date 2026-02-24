import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time

# 导入环境和算法类
from env import EdgeEnv
from PATD3_1 import PATD3Agent, PrioritizedBuffer
from TD3 import TD3Agent, ReplayBuffer
from PPO import PPOAgent
from NoRL import NoRLAgent

# ==========================================
# 1. 实验参数配置
# ==========================================
NODE_NUM = 6
STATE_DIM = NODE_NUM * 16
# MAX_EPISODES = 1000
# STEPS_PER_EPISODE = 500
MAX_EPISODES = 200
STEPS_PER_EPISODE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_scenario(episode):
#     if episode < 100: return 'normal'
#     if episode < 200: return 'low_perf'
#     if episode < 300: return 'low_rel'
#     return 'low_sec'

def get_scenario(episode):
    if episode < 50: return 'normal'
    if episode < 100: return 'low_perf'
    if episode < 150: return 'low_rel'
    return 'low_sec'


# ==========================================
# 2. 核心测试函数
# ==========================================
def run_comparison():
    algorithms = ['PA-TD3', 'TD3', 'PPO', 'Static_weight']

    for algo_name in algorithms:
        print(f"\n>>> 开始测试算法: {algo_name}")
        sta_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        writer = SummaryWriter(log_dir=f"logs/{algo_name}_{sta_time}")
        env = EdgeEnv(NODE_NUM)

        # 初始化对应的 Agent
        if algo_name == 'PA-TD3':
            agent = PATD3Agent(STATE_DIM, NODE_NUM, NODE_NUM, DEVICE)
            buffer = PrioritizedBuffer()
        elif algo_name == 'TD3':
            agent = TD3Agent(STATE_DIM, NODE_NUM, DEVICE)
            buffer = ReplayBuffer(STATE_DIM, NODE_NUM + 16)
        elif algo_name == 'PPO':
            agent = PPOAgent(STATE_DIM, NODE_NUM, DEVICE)
        elif algo_name == 'Static_weight':
            agent = NoRLAgent(NODE_NUM)

        global_step = 0  # 全局步数，用于高频率记录节点变化

        for ep in range(MAX_EPISODES):
            # print("MAX_EPISODES:", ep)
            scenario = get_scenario(ep)
            env.set_scenario(scenario)
            s = env.reset()

            ep_reward = 0
            ep_all_trust = []
            ep_consensus_trust = []

            # 记录本轮每个节点的信任值累积（用于算 Episode 平均）
            node_trust_history = {f'node_{i}': [] for i in range(NODE_NUM)}

            for t in range(STEPS_PER_EPISODE):
                # print("STEPS_PER_EPISODE:", t)
                # A. 动作选择
                if algo_name == 'PA-TD3':
                    disc_act, raw_act, trust_pred, w_pack, w_flat = agent.select_action(s)
                    trust_dict = {f'node_{i}': trust_pred[i] for i in range(NODE_NUM)}
                elif algo_name == 'TD3':
                    act_dict, trust_dict, w_pack, raw_act = agent.select_action(s)
                    disc_act = [act_dict[f'node_{i}'] for i in range(NODE_NUM)]
                elif algo_name == 'PPO':
                    act_dict, w_pack, log_p, val, act_data = agent.select_action(s)
                    trust_dict = {f'node_{i}': 0.5 for i in range(NODE_NUM)}
                elif algo_name == 'Static_weight':
                    act_dict, trust_dict, w_pack, _ = agent.select_action(s)
                    disc_act = [act_dict[f'node_{i}'] for i in range(NODE_NUM)]

                # B. 环境交互
                act_dict_final = {f'node_{i}': (act_dict[f'node_{i}'] if algo_name != 'PA-TD3' else disc_act[i])
                                  for i in range(NODE_NUM)}
                ns, r, d, info = env.step(act_dict_final, trust_dict, w_pack, t)

                # C. 节点级指标采集
                for i in range(NODE_NUM):
                    nid = f'node_{i}'
                    current_trust = env.nodes[nid].trust_total
                    node_trust_history[nid].append(current_trust)

                    # [新增] 每一单步记录一次具体节点信任值，观察细微波动
                    # writer.add_scalar(f'Individual_Node_Trust_Step/{nid}', current_trust, global_step)

                # D. 整体指标计算
                avg_all = np.mean([env.nodes[f'node_{i}'].trust_total for i in range(NODE_NUM)])
                selected_trusts = [env.nodes[nid].trust_total for nid, act in act_dict_final.items() if act == 1]
                avg_consensus = np.mean(selected_trusts) if selected_trusts else 0

                ep_all_trust.append(avg_all)
                ep_consensus_trust.append(avg_consensus)

                # ... (保持与之前代码一致) ...
                if algo_name == 'PA-TD3':
                    trust_lbl = [info['trust_totals'][f'node_{i}'] for i in range(NODE_NUM)]
                    buffer.add((s, raw_act, r, ns, d, trust_lbl, w_flat), priority=(r > 20))
                    if len(buffer.storage) > 100: agent.train(buffer.sample(64))
                elif algo_name == 'TD3':
                    buffer.add(s, raw_act, ns, r, d)
                    if buffer.size > 100: agent.train(buffer, batch_size=64)
                elif algo_name == 'PPO':
                    agent.buffer.append((s, act_data, log_p, r, d, val))

                s = ns
                ep_reward += r
                global_step += 1
                if d: break

            if algo_name == 'PPO': agent.update()

            # --- TensorBoard 记录 (Episode 级别) ---
            # 1. 记录每个节点在本轮的平均信任值（曲线更平滑）
            for i in range(NODE_NUM):
                nid = f'node_{i}'
                writer.add_scalar(f'Node_Trust_Episode/{nid}', np.mean(node_trust_history[nid]), ep)
            if ep % 10 == 0:
                # 2. 核心性能指标
                writer.add_scalar('Summary_Metrics/Avg_All_Nodes_Trust', np.mean(ep_all_trust), ep)
                writer.add_scalar('Summary_Metrics/Avg_Consensus_Nodes_Trust', np.mean(ep_consensus_trust), ep)
                writer.add_scalar('Summary_Metrics/Total_Reward', ep_reward, ep)

            # 3. 环境场景标识
            scene_map = {'normal': 0, 'low_perf': 1, 'low_rel': 2, 'low_sec': 3}
            writer.add_scalar('Env/Scenario_Index', scene_map[scenario], ep)

            if ep % 10 == 0:
                print(
                    f"[{algo_name}] Ep: {ep} | Scene: {scenario} | Reward: {ep_reward:.1f} | Consensus Trust: {np.mean(ep_consensus_trust):.4f}")
                print("上级权重", w_pack['dim'], "\n",
                      "性能", w_pack['perf'], "\n",
                      "可靠性", w_pack['rel'], "\n",
                      "安全性", w_pack['sec'])
        writer.close()


if __name__ == "__main__":
    if not os.path.exists('logs'): os.makedirs('logs')
    run_comparison()
