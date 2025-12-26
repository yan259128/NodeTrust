from VRF import ECVRF


class VRF_BFT_Consensus:
    """ 混合共识引擎：结合信任分评估的 VRF 记账抽奖与 BFT 多数验证 """

    def __init__(self, node_obj):
        self.node = node_obj
        self.pending_votes = {}  # {block_hash: {voter_id: signature}}
        self.current_proposal = None  # 本节点发起的区块提议

    def check_lottery(self, seed_hash):
        """ 执行 VRF 抽奖。阈值受 node.trust_total 影响 """
        beta, pi = ECVRF.prove(self.node.private_key, seed_hash.encode())
        val = int.from_bytes(beta[:8], 'big') / (2 ** 64)
        # 信任度(0.1-1.0)越高，中奖范围越大
        return (val < self.node.trust_total * 0.2), beta, pi

    def collect_vote(self, b_hash, voter_id, sig, active_count):
        """ 收集分片内投票。判断是否满足 2/3 多数规则 (BFT) """
        if b_hash not in self.pending_votes: self.pending_votes[b_hash] = {}
        self.pending_votes[b_hash][voter_id] = sig
        # 阈值计算：活跃节点(含自己)的 2/3 以上
        threshold = (active_count * 2 // 3) + 1
        return len(self.pending_votes[b_hash]) >= threshold
