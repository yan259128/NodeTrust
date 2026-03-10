import asyncio
import time
import msgpack
import random
from cryptography.hazmat.primitives import serialization
from Blockchain.VRF import ECVRF
from Blockchain.block import Block
from Blockchain.transaction import TraceTransaction


class TraceabilityService:
    def __init__(self, b_node, server, client):
        self.node = b_node
        self.server = server
        self.client = client
        self.chain = b_node.chain_manager
        self.tx_pool = b_node.tx_pool
        self.consensus = b_node.consensus_engine

        self.mode = self.consensus.mode
        self.shard_id = b_node.shard_id

        self.active_shard_peers = {}
        self.syncing = False
        self.best_proposals = {}
        self.height_candidates = {}
        self.vote_tasks = {}
        self.recovering_hashes = set()

        self.HEARTBEAT_INTERVAL = 3
        self.OFFLINE_THRESHOLD = 10
        self.SCAN_ALERT_THRESHOLD = 5

        # 预存公钥，避免高频签名时的重复序列化开销
        self.node_pub_bin = self.node.public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)

        # --- 性能优化：异步存证队列 ---
        self.evidence_queue = asyncio.Queue()
        # 启动后台存证处理器，不阻塞主线程
        asyncio.create_task(self._evidence_processor())

    async def _evidence_processor(self):
        """ 存证协程：从队列获取任务，执行签名、入池和广播 """
        while True:
            p_id, loc = await self.evidence_queue.get()
            try:
                # 找到前序指针，维持 O(k) 逻辑链
                latest_h_bin = self.chain.state_db.get(p_id.encode())
                ph = latest_h_bin.decode() if latest_h_bin else ""

                scan_tx = TraceTransaction(p_id, "高速扫码存证", TraceTransaction.OP_SCAN, loc, self.node_pub_bin,
                                           ph=ph)
                scan_tx.sign_tx(self.node.private_key)

                if self.tx_pool.is_new(scan_tx.h):
                    if self.tx_pool.add_tx(scan_tx):
                        await self.server.broadcast("TX", scan_tx.pack_signed())
            except:
                pass
            finally:
                self.evidence_queue.task_done()

    async def handle_incoming(self, topic, payload):
        """ 广播消息分发逻辑 """
        if topic == "HEARTBEAT":
            await self._handle_heartbeat(payload)
        elif topic == "TX":
            await self._handle_transaction(payload)
        elif topic == "PROPOSAL":
            await self._handle_proposal(payload)
        elif topic == "VOTE":
            await self._handle_vote(payload)
        elif topic == "COMMIT_BLOCK":
            await self._handle_committed_block(payload)

    async def _handle_heartbeat(self, payload):
        data = msgpack.unpackb(payload)
        if data['shard_id'] == self.shard_id:
            self.active_shard_peers[data['id']] = {
                "last_seen": time.time(), "height": data.get("height", 0),
                "port": data["port"], "ip": data["ip"]
            }
        self.client.add_peer(data['ip'], data['port'])

    async def handle_client_transaction(self, payload):
        """ 接收 PULL 接口交易 """
        try:
            tx = TraceTransaction.unpack(payload)
        except:
            return
        if self.tx_pool.is_new(tx.h):
            if tx.shard_id == self.shard_id:
                if self.tx_pool.add_tx(tx):
                    print(f"[BENCHMARK] TX_ENTRY|{tx.h}|{tx.ts}|{time.time()}")
            await self.server.broadcast("TX", payload)

    async def _handle_transaction(self, payload):
        """ 接收 SUB 广播交易 """
        try:
            tx = TraceTransaction.unpack(payload)
        except:
            return
        if self.tx_pool.is_new(tx.h):
            if tx.shard_id == self.shard_id:
                if self.tx_pool.add_tx(tx):
                    print(f"[BENCHMARK] TX_ENTRY|{tx.h}|{tx.ts}|{time.time()}")

    async def handle_rep_requests(self, req):
        """ RPC 处理器：实现极致 QPS 查询响应 """
        req_type = req.get("type")

        if req_type == "QUERY_TRACE":
            p_id = req.get("p_id")
            # 1. 内存/磁盘 高速跳转回溯
            path, scan_count = self.chain.trace_back(p_id)

            if path is None:
                return {"status": "NOT_FOUND"}

            # 2. 查询即存证（带限流保护）
            # 如果 scan_count >= 1 或是预警状态，则 100% 存证；
            # 否则（绝大多数正常查询）采样 0.5%，保护系统不被共识压垮。
            sample_rate = 1.0 if scan_count > 0 else 0.1
            if random.random() < sample_rate:
                # 放入异步队列，响应耗时几乎为
                self.evidence_queue.put_nowait((p_id, req.get("loc", 0)))
            # self.evidence_queue.put_nowait((p_id, req.get("loc", 0)))

            return {
                "status": "OK",
                "data": path,
                "scan_count": scan_count,
                "is_alert": scan_count >= self.SCAN_ALERT_THRESHOLD
            }

        elif req_type == "JOIN":
            return {"status": "OK", "height": self.chain.height}
        elif req_type == "GET_BLOCK":
            idx = req.get("idx")
            raw_data = self.chain.get_block_by_idx(idx)
            return {"status": "OK", "data": list(raw_data) if raw_data else None}
        return {"status": "ERR"}

    # ==========================================
    # 核心共识流程逻辑 (完全保留原有代码)
    # ==========================================

    async def _handle_proposal(self, payload):
        block_obj = Block.unpack(payload)
        if block_obj.idx <= self.chain.height: return

        # PBFT Leader 过滤逻辑
        if self.mode == "PBFT":
            all_node_ids = list(self.active_shard_peers.keys()) + [self.node.Id]
            if not self.consensus.is_pbft_leader(self.chain.height, all_node_ids):
                pass  # 仅作为观察者

        self.best_proposals[block_obj.h] = block_obj
        if block_obj.idx not in self.height_candidates or block_obj.h < self.height_candidates[block_obj.idx].h:
            self.height_candidates[block_obj.idx] = block_obj

        if block_obj.idx not in self.vote_tasks or self.vote_tasks[block_obj.idx].done():
            self.vote_tasks[block_obj.idx] = asyncio.create_task(self._wait_and_vote(block_obj.idx))

    async def _wait_and_vote(self, height):
        # 略微缩短等待时间提高单机吞吐
        wait_time = 0.3 if self.mode == "PBFT" else 0.5
        await asyncio.sleep(wait_time)

        if height <= self.chain.height: return
        target = self.height_candidates.get(height)

        if target and target.prev_h == self.chain.tip:
            self.consensus.current_proposal = target
            sig = self.node.private_key.sign(target.h.encode())
            vote_msg = {"h": target.h, "idx": height, "id": self.node.Id, "sig": sig, "phase": "PREPARE"}
            await self.server.broadcast("VOTE", msgpack.packb(vote_msg))

    async def _handle_vote(self, payload):
        data = msgpack.unpackb(payload)
        b_hash, b_idx, phase = data['h'], data.get('idx', -1), data['phase']
        if b_idx != self.chain.height + 1: return

        active_count = len(self.active_shard_peers) + 1
        if self.consensus.collect_vote(b_hash, data['id'], data['sig'], active_count, phase):
            block_to_commit = self.best_proposals.get(b_hash)
            if block_to_commit:
                if self.mode == "TW_BFT" and phase == "PREPARE":
                    await self.commit_block(block_to_commit)
                    self._clear_cache(block_to_commit.idx)
                elif self.mode == "PBFT":
                    if phase == "PREPARE":
                        sig = self.node.private_key.sign(b_hash.encode())
                        commit_vote = {"h": b_hash, "idx": b_idx, "id": self.node.Id, "sig": sig, "phase": "COMMIT"}
                        await self.server.broadcast("VOTE", msgpack.packb(commit_vote))
                    elif phase == "COMMIT":
                        await self.commit_block(block_to_commit)
                        self._clear_cache(block_to_commit.idx)
            else:
                if b_hash not in self.recovering_hashes:
                    self.recovering_hashes.add(b_hash)
                    asyncio.create_task(self._recover_missing_block(b_hash, b_idx))

    async def _handle_committed_block(self, payload):
        block_obj = Block.unpack(payload)
        if block_obj.idx == self.chain.height + 1 and block_obj.prev_h == self.chain.tip:
            await self.commit_block(block_obj, broadcast=False)
            self._clear_cache(block_obj.idx)

    async def _recover_missing_block(self, b_hash, b_idx):
        try:
            if b_idx != self.chain.height + 1: return
            for pid, pdata in self.active_shard_peers.items():
                res = await self.client.fetch_peer(pdata["ip"], pdata["port"], {"type": "GET_BLOCK", "idx": b_idx})
                if res and res.get("status") == "OK" and res.get("data"):
                    sync_block = Block.unpack(bytes(res["data"]))
                    if sync_block.h == b_hash:
                        await self.commit_block(sync_block)
                        self._clear_cache(b_idx)
                        break
        finally:
            self.recovering_hashes.discard(b_hash)

    def _clear_cache(self, height):
        self.height_candidates.pop(height, None)
        self.vote_tasks.pop(height, None)
        self.consensus.clear_votes()
        keys_to_del = [k for k, b in self.best_proposals.items() if b.idx <= height]
        for k in keys_to_del: self.best_proposals.pop(k, None)

    async def commit_block(self, block_obj, broadcast=True):
        if self.chain.add_block(block_obj):
            tx_hashes = [TraceTransaction.unpack(t).h for t in block_obj.txs]
            print(f"[BENCHMARK] BLOCK_COMMIT|{len(block_obj.txs)}|{','.join(tx_hashes)}|{time.time()}")
            self.tx_pool.remove_txs(tx_hashes)
            self.consensus.current_proposal = None
            if broadcast:
                await self.server.broadcast("COMMIT_BLOCK", block_obj.pack())
            return True
        return False

    async def run_consensus_logic(self):
        """ 主共识循环逻辑，覆盖所有算法模式 """
        await asyncio.sleep(5)
        while True:
            await asyncio.sleep(0.5)
            if len(self.tx_pool.pool) < 1 or self.syncing or not self.active_shard_peers: continue

            # 高度校验
            max_peer_h = max([p["height"] for p in self.active_shard_peers.values()], default=0)
            if self.chain.height < max_peer_h: continue

            tx_batch = [t.pack_signed() for t in self.tx_pool.get_batch(100)]
            pub_bytes = self.node.public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

            # --- 模式 1: PBFT 确定性 Leader 轮询 ---
            if self.mode == "PBFT":
                all_ids = sorted(list(self.active_shard_peers.keys()) + [self.node.Id])
                if self.consensus.is_pbft_leader(self.chain.height, all_ids):
                    new_idx = self.chain.height + 1
                    prop = Block(new_idx, tx_batch, self.chain.tip, self.node.Id, pub_bytes)
                    prop.sign_block(self.node.private_key)
                    await self._handle_proposal(prop.pack())
                    await self.server.broadcast("PROPOSAL", prop.pack())

            # --- 模式 2: TW_BFT 信任加权抽奖 ---
            elif self.mode == "TW_BFT":
                win, beta, pi = self.consensus.check_lottery(self.chain.tip)
                if win:
                    new_idx = self.chain.height + 1
                    prop = Block(new_idx, tx_batch, self.chain.tip, self.node.Id, pub_bytes, pi, beta)
                    prop.sign_block(self.node.private_key)
                    await self._handle_proposal(prop.pack())
                    await self.server.broadcast("PROPOSAL", prop.pack())

            # --- 模式 3: PoW 算力竞赛 ---
            elif self.mode == "PoW":
                prop = Block(self.chain.height + 1, tx_batch, self.chain.tip, self.node.Id, pub_bytes)
                if self.consensus.run_pow_mining(prop):
                    prop.sign_block(self.node.private_key)
                    await self.server.broadcast("PROPOSAL", prop.pack())
                    await self.commit_block(prop)

    async def heartbeat_loop(self):
        while True:
            hb = {"id": self.node.Id, "shard_id": self.shard_id, "ip": "127.0.0.1", "port": self.node.port,
                  "height": self.chain.height}
            await self.server.broadcast("HEARTBEAT", msgpack.packb(hb))
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

    async def monitor_peers_loop(self):
        while True:
            await asyncio.sleep(5)
            now = time.time()
            dead = [nid for nid, d in self.active_shard_peers.items() if now - d["last_seen"] > self.OFFLINE_THRESHOLD]
            for nid in dead: del self.active_shard_peers[nid]