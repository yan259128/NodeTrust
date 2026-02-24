import asyncio
import time
import msgpack
import hashlib
from cryptography.hazmat.primitives import serialization

# 导入区块链核心模块
from Blockchain.VRF import ECVRF
from Blockchain.block import Block
from Blockchain.transaction import TraceTransaction


class TraceabilityService:
    """
    业务逻辑中枢：
    1. 负责多模式共识（TW_BFT, PBFT, PoW）的逻辑切换与状态机维护
    2. 实现地理分片路由过滤
    3. 记录真实性能指标（用于TPS与延迟测量）
    4. 节点存活监控（心跳）与数据并行同步
    """

    def __init__(self, b_node, server, client):
        self.node = b_node
        self.server = server
        self.client = client

        # 挂载核心组件
        self.chain = b_node.chain_manager
        self.tx_pool = b_node.tx_pool
        self.consensus = b_node.consensus_engine

        # 属性配置
        self.mode = self.consensus.mode  # "TW_BFT", "PBFT", "PoW"
        self.shard_id = b_node.shard_id  # 地理分片ID
        self.active_shard_peers = {}  # 同分片内的活跃节点 {node_id: last_seen}

        self.syncing = False  # 同步状态锁
        self.HEARTBEAT_INTERVAL = 3  # 心跳发送间隔
        self.OFFLINE_THRESHOLD = 10  # 节点离线判定阈值

    # ================= 核心消息分发器 =================

    async def handle_incoming(self, topic, payload):
        """ 统一处理所有入站 ZMQ 广播消息 """
        if topic == "HEARTBEAT":
            await self._handle_heartbeat(payload)

        elif topic == "TX":
            await self._handle_transaction(payload)

        elif topic == "PROPOSAL":
            await self._handle_proposal(payload)

        elif topic == "VOTE":
            await self._handle_vote(payload)

    # ================= 消息处理子逻辑 =================

    async def _handle_heartbeat(self, payload):
        """ 处理心跳：更新活跃节点并自动发现邻居 """
        data = msgpack.unpackb(payload)
        # 只有属于同一地理分片的节点才被计入共识阈值基数
        if data['shard_id'] == self.shard_id:
            self.active_shard_peers[data['id']] = time.time()

        # 无论是否同分片，都建立物理连接，方便跨片通信/查询
        self.client.add_peer(data['ip'], data['port'])

    async def _handle_transaction(self, payload):
        """ 处理溯源交易：性能监测起始点 """
        # 性能埋点：记录交易进入系统的绝对时间
        entry_time = time.time()
        tx = TraceTransaction.unpack(payload)

        # 地理分片路由：边缘节点仅处理本片内的交易，提升局部TPS
        if tx.shard_id == self.shard_id:
            if self.tx_pool.add_tx(tx):
                # [测量用日志] 交易进入池
                print(f"[BENCHMARK] TX_ENTRY|{tx.h}|{entry_time}")

    async def _handle_proposal(self, payload):
        """ 处理区块提议：根据不同共识模式执行验证 """
        block = Block.unpack(payload)

        if self.mode == "PoW":
            # PoW: 直接验证哈希难度，成功则直接上链
            if self.consensus.verify_pow(block) and block.prev_h == self.chain.tip:
                await self.commit_block(block)

        elif self.mode in ["PBFT", "TW_BFT"]:
            # BFT类共识验证
            is_valid = False
            if self.mode == "TW_BFT":
                # TW_BFT 额外验证 VRF 抽奖证明
                is_valid = (block.prev_h == self.chain.tip and
                            ECVRF.verify(block.pub, self.chain.tip.encode(), block.pi, block.beta))
            else:
                # PBFT 验证提议合法性
                is_valid = (block.prev_h == self.chain.tip)

            if is_valid:
                # 第一阶段：发送 PREPARE 票
                sig = self.node.private_key.sign(block.h.encode())
                vote_msg = {"h": block.h, "id": self.node.Id, "sig": sig, "phase": "PREPARE"}
                await self.server.broadcast("VOTE", msgpack.packb(vote_msg))

    async def _handle_vote(self, payload):
        """ 处理投票：维护 PBFT 状态机及 2/3 阈值判断 """
        data = msgpack.unpackb(payload)
        b_hash, voter_id, sig, phase = data['h'], data['id'], data['sig'], data['phase']

        # 动态计算共识基数（当前分片在线人数 + 自己）
        active_count = len(self.active_shard_peers) + 1

        # 收集并检查是否达到 2/3 多数
        if self.consensus.collect_vote(b_hash, voter_id, sig, active_count, phase):
            # TW_BFT: 一阶段完成即 Commit
            # PBFT: 必须进入 COMMIT 阶段并完成后才 Commit
            if self.mode == "TW_BFT" or (self.mode == "PBFT" and phase == "COMMIT"):
                if self.consensus.current_proposal and self.consensus.current_proposal.h == b_hash:
                    await self.commit_block(self.consensus.current_proposal)

            elif self.mode == "PBFT" and phase == "PREPARE":
                # PBFT 特有：PREPARE 阶段达成，发起 COMMIT 投票
                sig = self.node.private_key.sign(b_hash.encode())
                vote_msg = {"h": b_hash, "id": self.node.Id, "sig": sig, "phase": "COMMIT"}
                await self.server.broadcast("VOTE", msgpack.packb(vote_msg))

    async def commit_block(self, block):
        """ 区块最终上链：性能监测终点 """
        commit_time = time.time()
        if self.chain.add_block(block):
            tx_hashes = []
            for t_bin in block.txs:
                tx = TraceTransaction.unpack(t_bin)
                tx_hashes.append(tx.h)

            # [测量用日志] 格式：BLOCK_COMMIT | 交易数量 | 交易哈希列表 | 上链时间
            txs_str = ",".join(tx_hashes)
            print(f"[BENCHMARK] BLOCK_COMMIT|{len(tx_hashes)}|{txs_str}|{commit_time}")

            # 清理本地交易池
            self.tx_pool.remove_txs(tx_hashes)
            self.consensus.current_proposal = None

            # 更新节点验证统计
            self.node.update_transaction_stats(is_valid=True)
            return True
        return False

    # ================= 后台周期性任务 =================

    async def run_consensus_logic(self):
        """ 主出块循环：负责主动提议区块 """
        while True:
            # 性能优化：无交易或正在同步时不执行抽奖/挖矿
            await asyncio.sleep(2)
            if len(self.tx_pool.pool) < 1 or self.syncing:
                continue

            # 打包当前池内最早的交易
            tx_batch = [t.pack_signed() for t in self.tx_pool.get_batch(20)]
            pub_bytes = self.node.public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )

            if self.mode == "PoW":
                # 模式1: PoW 算力竞争
                new_block = Block(self.chain.height + 1, tx_batch, self.chain.tip, self.node.Id, pub_bytes)
                if self.consensus.run_pow_mining(new_block):
                    new_block.sign_block(self.node.private_key)
                    await self.server.broadcast("PROPOSAL", new_block.pack())
                    await self.commit_block(new_block)

            elif self.mode == "TW_BFT":
                # 模式2: 信任加权 VRF 抽奖
                win, beta, pi = self.consensus.check_lottery(self.chain.tip)
                if win:
                    prop = Block(self.chain.height + 1, tx_batch, self.chain.tip, self.node.Id, pub_bytes, pi, beta)
                    prop.sign_block(self.node.private_key)
                    self.consensus.current_proposal = prop
                    await self.server.broadcast("PROPOSAL", prop.pack())

            elif self.mode == "PBFT":
                # 模式3: PBFT 视图轮询（Leader 提议）
                if self._is_leader():
                    prop = Block(self.chain.height + 1, tx_batch, self.chain.tip, self.node.Id, pub_bytes)
                    prop.sign_block(self.node.private_key)
                    self.consensus.current_proposal = prop
                    await self.server.broadcast("PROPOSAL", prop.pack())

    def _is_leader(self):
        """ 针对 PBFT 的简单 Leader 选取：通过高度对分片内活跃节点取模 """
        active_ids = sorted(list(self.active_shard_peers.keys()) + [self.node.Id])
        if not active_ids: return False
        # Leader 会随高度轮转，防止单节点故障
        leader_idx = (self.chain.height + 1) % len(active_ids)
        return active_ids[leader_idx] == self.node.Id

    async def heartbeat_loop(self):
        """ 发送包含地理分片信息的存活信号 """
        while True:
            hb_data = {
                "id": self.node.Id,
                "shard_id": self.shard_id,
                "ip": "127.0.0.1",
                "port": self.node.port
            }
            await self.server.broadcast("HEARTBEAT", msgpack.packb(hb_data))
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

    async def monitor_peers_loop(self):
        """ 监测邻居节点状态，清理离线节点 """
        while True:
            await asyncio.sleep(5)
            now = time.time()
            # 找出超时的节点
            dead_nodes = [nid for nid, last in self.active_shard_peers.items()
                          if now - last > self.OFFLINE_THRESHOLD]
            for nid in dead_nodes:
                print(f"[*] Node {nid} is detected OFFLINE.")
                del self.active_shard_peers[nid]

    # ================= 数据同步 RPC 接口 =================

    async def handle_rep_requests(self, req):
        """ 响应其他节点的 REQ 同步请求 """
        req_type = req.get("type")
        if req_type == "JOIN":
            return {"status": "OK", "height": self.chain.height}
        elif req_type == "GET_BLOCK":
            idx = req.get("idx")
            raw_data = self.chain.get_block_by_idx(idx)
            # ZMQ REP 发送 JSON，二进制数据需转换为列表
            return {"status": "OK", "data": list(raw_data) if raw_data else None}
        return {"status": "ERR"}

    async def sync_blockchain(self, ip, port):
        """ 节点加入时的并行同步任务 """
        self.syncing = True
        print(f"[*] Node {self.node.Id} starting sync from seed {ip}:{port}...")
        res = await self.client.fetch_peer(ip, port, {"type": "JOIN"})
        if res and res["height"] > self.chain.height:
            for i in range(self.chain.height + 1, res["height"] + 1):
                b_res = await self.client.fetch_peer(ip, port, {"type": "GET_BLOCK", "idx": i})
                if b_res and b_res["data"]:
                    # 将接收到的区块二进制上链
                    self.chain.add_block(Block.unpack(bytes(b_res["data"])))
        self.syncing = False
        print(f"[*] Node {self.node.Id} sync completed. Height: {self.chain.height}")

    async def consumer_query(self, p_id):
        """ 防伪查询接口：每查询一次，扫码次数加 1 """
        count = self.chain.update_scan_stats(p_id)
        return count