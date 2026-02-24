import time
import hashlib
from Blockchain.VRF import ECVRF
# import VRF


class ConsensusEngine:
    """ 多算法共识引擎：支持 TW_BFT, PBFT, PoW """

    def __init__(self, node_obj, mode="TW_BFT"):
        self.node = node_obj
        self.mode = mode  # "TW_BFT", "PBFT", "PoW"

        # PBFT 状态库
        self.prepare_votes = {}
        self.commit_votes = {}

        # PoW 难度
        self.difficulty = 4
        self.current_proposal = None

    def check_lottery(self, seed_hash):
        """ TW_BFT : 基于信任值的抽奖 """
        beta, pi = ECVRF.prove(self.node.private_key, seed_hash.encode())
        val = int.from_bytes(beta[:8], 'big') / (2 ** 64)
        threshold = self.node.trust_total * 0.2
        return (val < threshold), beta, pi

    def run_pow_mining(self, block):
        """ PoW: 算力挖掘 """
        target = "0" * self.difficulty
        while True:
            hash_res = block.calculate_header_hash()
            if hash_res.startswith(target):
                block.h = hash_res
                return True
            block.nonce += 1
            if block.nonce % 1000 == 0: time.sleep(0.001)

    def collect_vote(self, b_hash, voter_id, sig, active_count, phase="PREPARE"):
        """ 投票收集 (针对 TW_BFT 和 PBFT) """
        target_pool = self.prepare_votes if phase == "PREPARE" else self.commit_votes
        if b_hash not in target_pool: target_pool[b_hash] = {}
        target_pool[b_hash][voter_id] = sig

        threshold = (active_count * 2 // 3) + 1
        return len(target_pool[b_hash]) >= threshold