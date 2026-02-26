import time
import hashlib
from Blockchain.VRF import ECVRF


class ConsensusEngine:
    """
    多算法共识引擎：支持 TW_BFT (优化版), PBFT, PoW
    """

    def __init__(self, node_obj, mode="TW_BFT"):
        self.node = node_obj
        self.mode = mode  # "TW_BFT", "PBFT", "PoW"

        # BFT 选票池: { block_hash: { voter_id: signature } }
        self.prepare_votes = {}
        self.commit_votes = {}

        # PoW 配置
        self.difficulty = 4
        self.current_proposal = None

    # ==========================================
    # 1. TW_BFT & PBFT 核心逻辑
    # ==========================================
    def check_lottery(self, seed_hash):
        """
        TW_BFT 信任加权抽奖：
        阈值根据节点 trust_total 动态变化，高信任节点中奖率更高。
        """
        beta, pi = ECVRF.prove(self.node.private_key, seed_hash.encode())
        val = int.from_bytes(beta[:8], 'big') / (2 ** 64)

        # 动态阈值优化：基础 0.4 + 信任加权 (最高可达 0.9)
        trust_val = getattr(self.node, 'trust_total', 0.5)
        threshold = 0.4 + (trust_val * 0.5)

        return (val < threshold), beta, pi

    def is_pbft_leader(self, current_height, active_nodes_list):
        """ PBFT 确定性 Leader 轮询 """
        if not active_nodes_list: return False
        sorted_nodes = sorted(active_nodes_list)
        leader_idx = (current_height + 1) % len(sorted_nodes)
        return sorted_nodes[leader_idx] == self.node.Id

    # ==========================================
    # 2. PoW 逻辑
    # ==========================================
    def run_pow_mining(self, block):
        target = "0" * self.difficulty
        print(f"[*] PoW Mining (Diff: {self.difficulty})...")
        while True:
            hash_res = block.calculate_header_hash()
            if hash_res.startswith(target):
                block.h = hash_res
                return True
            block.nonce += 1
            if block.nonce % 2000 == 0: time.sleep(0.001)

    def verify_pow(self, block):
        target = "0" * self.difficulty
        hash_res = block.calculate_header_hash()
        return hash_res == block.h and hash_res.startswith(target)

    # ==========================================
    # 3. 投票统计逻辑
    # ==========================================
    def collect_vote(self, b_hash, voter_id, sig, active_count, phase="PREPARE"):
        target_pool = self.prepare_votes if phase == "PREPARE" else self.commit_votes
        if b_hash not in target_pool:
            target_pool[b_hash] = {}

        target_pool[b_hash][voter_id] = sig
        threshold = (active_count * 2 // 3) + 1

        return len(target_pool[b_hash]) >= threshold

    def clear_votes(self, b_hash=None):
        if b_hash:
            self.prepare_votes.pop(b_hash, None)
            self.commit_votes.pop(b_hash, None)
        else:
            self.prepare_votes.clear()
            self.commit_votes.clear()