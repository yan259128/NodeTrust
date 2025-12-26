import asyncio
import msgpack
import time

from cryptography.hazmat.primitives import serialization

from Blockchain.VRF import ECVRF
from Blockchain.block import Block
from Blockchain.transaction import TraceTransaction
from Blockchain.transaction import TraceTransaction


class TraceabilityService:
    """ 业务逻辑中枢：管理心跳、分片路由过滤、同步及两阶段提交共识 """

    def __init__(self, b_node, server, client):
        self.node = b_node
        self.server, self.client = server, client
        self.shard_id = b_node.shard_id
        self.chain = b_node.chain_manager
        self.tx_pool = b_node.tx_pool
        self.consensus = b_node.consensus_engine
        self.active_shard_peers = {}  # 维护本分片活跃节点状态
        self.syncing = False  # 同步状态锁

    async def handle_incoming(self, topic, payload):
        """ 统一消息分发与分片过滤 """
        if topic == "HEARTBEAT":
            d = msgpack.unpackb(payload)
            # 地理分片过滤：仅统计同一分片节点以维护共识基数
            if d['shard_id'] == self.shard_id:
                self.active_shard_peers[d['id']] = time.time()
            self.client.add_peer(d['ip'], d['port'])

        elif topic == "TX":
            tx = TraceTransaction.unpack(payload)
            # 路由过滤：只处理属于本分片地理编码的交易
            if tx.shard_id == self.shard_id:
                if self.tx_pool.add_tx(tx):
                    print(f"[{self.shard_id}] Tx Accepted: {tx.h[:8]}")

        elif topic == "PROPOSAL":
            b = Block.unpack(payload)
            # 验证提议合法性：前序一致 + VRF验证
            if b.prev_h == self.chain.tip and ECVRF.verify(b.pub, self.chain.tip.encode(), b.pi, b.beta):
                # 发送投票(签名)给全网
                sig = self.node.private_key.sign(b.h.encode())
                await self.server.broadcast("VOTE", msgpack.packb({"h": b.h, "id": self.node.Id, "sig": sig}))

        elif topic == "VOTE":
            d = msgpack.unpackb(payload)
            # 计算分片内动态共识阈值
            total_active = len(self.active_shard_peers) + 1
            if self.consensus.collect_vote(d['h'], d['id'], d['sig'], total_active):
                if self.consensus.current_proposal and self.consensus.current_proposal.h == d['h']:
                    await self.commit_block(self.consensus.current_proposal)

    async def commit_block(self, block):
        """ 最终上链操作：更新账本并清理内存池 """
        if self.chain.add_block(block):
            tx_hashes = [TraceTransaction.unpack(t).h for t in block.txs]
            self.tx_pool.remove_txs(tx_hashes)
            self.consensus.current_proposal = None
            print(f"[Chain] Block #{block.idx} committed in {self.shard_id}")

    async def handle_rep_requests(self, req):
        """ 处理点对点同步与握手请求 """
        t = req.get("type")
        if t == "JOIN": return {"status": "OK", "height": self.chain.height}
        if t == "GET_BLOCK":
            raw = self.chain.get_block_by_idx(req.get("idx"))
            return {"status": "OK", "data": list(raw) if raw else None}
        return {"status": "ERR"}

    async def heartbeat_loop(self):
        """ 广播包含地理分片ID的心跳包 """
        while True:
            hb = {"id": self.node.Id, "shard_id": self.shard_id, "ip": "127.0.0.1", "port": self.node.port}
            await self.server.broadcast("HEARTBEAT", msgpack.packb(hb))
            await asyncio.sleep(3)

    async def sync_blockchain(self, ip, port):
        """ 并行数据同步任务 """
        self.syncing = True
        res = await self.client.fetch_peer(ip, port, {"type": "JOIN"})
        if res and res["height"] > self.chain.height:
            print(f"[*] Syncing {res['height'] - self.chain.height} blocks...")
            for i in range(self.chain.height + 1, res["height"] + 1):
                b_res = await self.client.fetch_peer(ip, port, {"type": "GET_BLOCK", "idx": i})
                if b_res and b_res["data"]:
                    self.chain.add_block(Block.unpack(bytes(b_res["data"])))
        self.syncing = False

    async def run_consensus_logic(self):
        """ 周期性记账抽奖任务 """
        while True:
            await asyncio.sleep(5)
            if len(self.tx_pool.pool) < 1 or self.syncing: continue
            win, beta, pi = self.consensus.check_lottery(self.chain.tip)
            if win:
                # 打包交易包发起提议
                tx_batch = [t.pack_signed() for t in self.tx_pool.get_batch(10)]
                pub = self.node.public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
                prop = Block(self.chain.height + 1, tx_batch, self.chain.tip, self.node.Id, pub, pi, beta)
                prop.sign_block(self.node.private_key)
                self.consensus.current_proposal = prop
                await self.server.broadcast("PROPOSAL", prop.pack())
