import asyncio
import time
import msgpack
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
        self.best_proposals = {}  # {hash: block_obj}
        self.height_candidates = {}  # {height: best_block_obj}
        self.vote_tasks = {}
        self.recovering_hashes = set()

        self.HEARTBEAT_INTERVAL = 3
        self.OFFLINE_THRESHOLD = 10

    async def handle_incoming(self, topic, payload):
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
                "last_seen": time.time(),
                "height": data.get("height", 0),
                "port": data["port"],
                "ip": data["ip"]
            }
        self.client.add_peer(data['ip'], data['port'])

    async def _handle_transaction(self, payload):
        try:
            tx = TraceTransaction.unpack(payload)
        except:
            return
        if tx.shard_id == self.shard_id:
            if self.tx_pool.add_tx(tx):
                print(f"[BENCHMARK] TX_ENTRY|{tx.h}|{time.time()}")
                await self.server.broadcast("TX", payload)

    async def _handle_proposal(self, payload):
        block = Block.unpack(payload)
        if block.idx <= self.chain.height: return

        if self.mode == "PBFT":
            all_node_ids = list(self.active_shard_peers.keys()) + [self.node.Id]
            if not self.consensus.is_pbft_leader(self.chain.height, all_node_ids):
                # 如果提议者不是我们计算出的 Leader，且不是我们自己，则需警惕
                # 这里简化处理：依然缓存，但如果哈希不对可能无法达成共识
                pass

        self.best_proposals[block.h] = block
        h_idx = block.idx
        if h_idx not in self.height_candidates or block.h < self.height_candidates[h_idx].h:
            self.height_candidates[h_idx] = block

        if h_idx not in self.vote_tasks or self.vote_tasks[h_idx].done():
            self.vote_tasks[h_idx] = asyncio.create_task(self._wait_and_vote(h_idx))

    async def _wait_and_vote(self, height):
        # PBFT 响应速度可以比 TW_BFT 快，因为 Leader 是确定的
        wait_time = 0.5 if self.mode == "PBFT" else 1.2
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
        b_hash, b_idx = data.get('h'), data.get('idx', -1)
        voter_id, sig, phase = data['id'], data['sig'], data['phase']

        if b_idx != self.chain.height + 1: return

        active_count = len(self.active_shard_peers) + 1
        threshold = (active_count * 2 // 3) + 1

        if self.consensus.collect_vote(b_hash, voter_id, sig, active_count, phase):
            block_to_commit = self.best_proposals.get(b_hash)
            if block_to_commit:
                # TW_BFT
                if self.mode == "TW_BFT" and phase == "PREPARE":
                    await self.commit_block(block_to_commit)
                    self._clear_cache(block_to_commit.idx)

                # PBFT
                elif self.mode == "PBFT":
                    if phase == "PREPARE":
                        # 收到 2/3 Prepare，进入 Commit 阶段
                        sig = self.node.private_key.sign(b_hash.encode())
                        vote_msg = {"h": b_hash, "idx": b_idx, "id": self.node.Id, "sig": sig, "phase": "COMMIT"}
                        print(f"[*] PBFT Prepare finished for {b_hash[:12]}, broadcasting COMMIT")
                        await self.server.broadcast("VOTE", msgpack.packb(vote_msg))
                    elif phase == "COMMIT":
                        # 收到 2/3 Commit，正式上链
                        await self.commit_block(block_to_commit)
                        self._clear_cache(block_to_commit.idx)
            else:
                if b_hash not in self.recovering_hashes:
                    self.recovering_hashes.add(b_hash)
                    asyncio.create_task(self._recover_missing_block(b_hash, b_idx))

    async def _handle_committed_block(self, payload):
        block = Block.unpack(payload)
        if block.idx == self.chain.height + 1 and block.prev_h == self.chain.tip:
            print(f"[*] QuickSync: Received committed block {block.h[:12]}")
            await self.commit_block(block, broadcast=False)
            self._clear_cache(block.idx)

    async def _recover_missing_block(self, b_hash, b_idx):
        try:
            if b_idx != self.chain.height + 1: return
            found = False
            for pid, pdata in self.active_shard_peers.items():
                res = await self.client.fetch_peer(pdata["ip"], pdata["port"], {"type": "GET_BLOCK", "idx": b_idx})
                if res and res.get("status") == "OK" and res.get("data"):
                    sync_block = Block.unpack(bytes(res["data"]))
                    if sync_block.h == b_hash:
                        await self.commit_block(sync_block)
                        self._clear_cache(b_idx)
                        found = True
                        break
            if not found and b_idx == self.chain.height + 1:
                print(f"[RECOVERY] Generating Empty Block for height {b_idx}...")
                pub_bytes = self.node.public_key.public_bytes(serialization.Encoding.Raw,
                                                              serialization.PublicFormat.Raw)
                recovery_block = Block(b_idx, [], self.chain.tip, "RECOVERY", pub_bytes)
                recovery_block.sign_block(self.node.private_key)
                await self.commit_block(recovery_block)
                self._clear_cache(b_idx)
        finally:
            self.recovering_hashes.discard(b_hash)

    def _clear_cache(self, height):
        self.height_candidates.pop(height, None)
        self.vote_tasks.pop(height, None)
        self.consensus.clear_votes()
        keys_to_del = [k for k, b in self.best_proposals.items() if b.idx <= height]
        for k in keys_to_del: self.best_proposals.pop(k, None)

    async def commit_block(self, block, broadcast=True):
        if self.chain.add_block(block):
            tx_hashes = [TraceTransaction.unpack(t).h for t in block.txs]
            print(f"[BENCHMARK] BLOCK_COMMIT|{len(block.txs)}|{','.join(tx_hashes)}|{time.time()}")
            self.tx_pool.remove_txs(tx_hashes)
            self.consensus.current_proposal = None
            if broadcast:
                await self.server.broadcast("COMMIT_BLOCK", block.pack())
            return True
        return False

    async def run_consensus_logic(self):
        """ 主共识循环 """
        while True:
            await asyncio.sleep(2.5)
            # 基础检查：是否同步中，池中是否有交易
            if len(self.tx_pool.pool) < 1 or self.syncing: continue

            # 高度同步检查
            max_peer_height = max([p["height"] for p in self.active_shard_peers.values()], default=0)
            if self.chain.height < max_peer_height: continue

            tx_batch = [t.pack_signed() for t in self.tx_pool.get_batch(50)]
            pub_bytes = self.node.public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

            if self.mode == "PBFT":
                all_node_ids = list(self.active_shard_peers.keys()) + [self.node.Id]
                if self.consensus.is_pbft_leader(self.chain.height, all_node_ids):
                    new_idx = self.chain.height + 1
                    prop = Block(new_idx, tx_batch, self.chain.tip, self.node.Id, pub_bytes)
                    prop.sign_block(self.node.private_key)
                    print(f"[PROPOSE_STAT] PBFT Leader {self.node.Id} proposing height {new_idx}")
                    await self._handle_proposal(prop.pack())
                    await self.server.broadcast("PROPOSAL", prop.pack())

            elif self.mode == "TW_BFT":
                win, beta, pi = self.consensus.check_lottery(self.chain.tip)
                if win:
                    new_idx = self.chain.height + 1
                    prop = Block(new_idx, tx_batch, self.chain.tip, self.node.Id, pub_bytes, pi, beta)
                    prop.sign_block(self.node.private_key)
                    print(f"[PROPOSE_STAT] TW_BFT Winner {self.node.Id} proposing height {new_idx}")
                    await self._handle_proposal(prop.pack())
                    await self.server.broadcast("PROPOSAL", prop.pack())

            # 模式 3: PoW (算力)
            elif self.mode == "PoW":
                prop = Block(self.chain.height + 1, tx_batch, self.chain.tip, self.node.Id, pub_bytes)
                if self.consensus.run_pow_mining(prop):
                    prop.sign_block(self.node.private_key)
                    await self.server.broadcast("PROPOSAL", prop.pack())
                    await self.commit_block(prop)

    async def heartbeat_loop(self):
        while True:
            hb_data = {
                "id": self.node.Id,
                "shard_id": self.shard_id,
                "ip": "127.0.0.1",
                "port": self.node.port,
                "height": self.chain.height
            }
            await self.server.broadcast("HEARTBEAT", msgpack.packb(hb_data))
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

    async def monitor_peers_loop(self):
        while True:
            await asyncio.sleep(5)
            now = time.time()
            dead_nodes = [nid for nid, data in self.active_shard_peers.items() if
                          now - data["last_seen"] > self.OFFLINE_THRESHOLD]
            for nid in dead_nodes:
                del self.active_shard_peers[nid]

    async def handle_rep_requests(self, req):
        req_type = req.get("type")
        if req_type == "JOIN":
            return {"status": "OK", "height": self.chain.height}
        elif req_type == "GET_BLOCK":
            idx = req.get("idx")
            raw_data = self.chain.get_block_by_idx(idx)
            return {"status": "OK", "data": list(raw_data) if raw_data else None}
        return {"status": "ERR"}