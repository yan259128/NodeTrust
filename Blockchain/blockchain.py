import plyvel
import Blockchain.block as block
from Blockchain.transaction import TraceTransaction
from collections import OrderedDict


class Blockchain:
    def __init__(self, node_id):
        # 1. 基础账本数据库 (Block H -> Block Data)
        self.db = plyvel.DB(f"./db_chain_{node_id}", create_if_missing=True)
        # 2. 状态数据库索引 (Product ID -> Latest TX Hash)
        self.state_db = plyvel.DB(f"./db_state_{node_id}", create_if_missing=True)
        # 3. 防伪存证统计 (Product ID -> Scan Count)
        self.anti_fake_db = plyvel.DB(f"./db_antifake_{node_id}", create_if_missing=True)
        # 4. 交易快速查询索引 (TX Hash -> TX Data)
        self.tx_index = plyvel.DB(f"./db_txs_{node_id}", create_if_missing=True)

        # --- 极致查询优化：双级内存索引 ---
        # 路径结果缓存池 (LRU 策略)
        self.query_cache = OrderedDict()
        self.CACHE_CAPACITY = 10000

        # 扫码计数器内存索引 (加速防伪 QPS)
        self.counter_memory_index = {}

        self.tip, self.height = None, -1
        self._check_genesis()
        self._preload_indexes()  # 启动时预热内存

    def _preload_indexes(self):
        """ 启动预热：将所有扫码计数值加载至内存，实现查询 0 磁盘 I/O """
        try:
            for key, value in self.anti_fake_db.iterator(prefix=b'cnt_'):
                p_id = key.decode()[4:]
                self.counter_memory_index[p_id] = int(value.decode())
        except:
            pass

    def _check_genesis(self):
        """ 创世逻辑保留 """
        last_h = self.db.get(b'l')
        if not last_h:
            gen = block.Block(0, [], "0" * 64, "GENESIS", b"0" * 32)
            gen.h = "0" * 64
            self.save_block(gen)
        else:
            self.tip = last_h.decode()
            raw = self.db.get(self.tip.encode())
            self.height = block.Block.unpack(raw).idx

    def save_block(self, b_obj):
        """ 写入区块并同步更新所有索引与缓存 """
        self.db.put(b_obj.h.encode(), b_obj.pack())
        self.db.put(f"idx_{b_obj.idx}".encode(), b_obj.h.encode())
        self.db.put(b'l', b_obj.h.encode())

        for tx_bin in b_obj.txs:
            tx = TraceTransaction.unpack(tx_bin)
            self.tx_index.put(tx.h.encode(), tx_bin)
            self.state_db.put(tx.p_id.encode(), tx.h.encode())

            if tx.op == TraceTransaction.OP_SCAN:
                # 扫码存证更新
                count_key = f"cnt_{tx.p_id}".encode()
                cur_cnt = self.counter_memory_index.get(tx.p_id, 0)
                new_cnt = cur_cnt + 1
                self.anti_fake_db.put(count_key, str(new_cnt).encode())

                # 同步内存计数器
                self.counter_memory_index[tx.p_id] = new_cnt
                # 更新缓存中的对象
                if tx.p_id in self.query_cache:
                    self.query_cache[tx.p_id]["scan_count"] = new_cnt
            else:
                # PRODUCE/TRANS 发生，轨迹链变动，必须失效路径缓存
                if tx.p_id in self.query_cache:
                    del self.query_cache[tx.p_id]

        self.tip = b_obj.h
        self.height = b_obj.idx

    def add_block(self, b_obj):
        if b_obj.prev_h != self.tip: return False
        self.save_block(b_obj)
        return True

    def trace_back(self, product_id):
        """
        核心算法：LRU 缓存 -> O(k) 磁盘回溯
        """
        # 1. 优先命中路径缓存
        if product_id in self.query_cache:
            self.query_cache.move_to_end(product_id)
            res = self.query_cache[product_id]
            return res["path"], res["scan_count"]

        # 2. 缓存未命中，执行磁盘级指针回溯
        trace_path = []
        latest_tx_h = self.state_db.get(product_id.encode())
        if not latest_tx_h: return None, 0

        curr_hash = latest_tx_h.decode()
        while curr_hash:
            tx_data = self.tx_index.get(curr_hash.encode())
            if not tx_data: break
            tx = TraceTransaction.unpack(tx_data)
            # 仅提取核心字段减少内存占用
            trace_path.append({
                "stage": tx.stg, "op": tx.op, "time": tx.ts, "loc": tx.loc
            })
            curr_hash = tx.ph
            if not curr_hash or curr_hash == "": break

        # 3. 获取内存计数值
        scan_count = self.counter_memory_index.get(product_id, 0)

        # 4. 更新 LRU 缓存池
        if len(self.query_cache) >= self.CACHE_CAPACITY:
            self.query_cache.popitem(last=False)
        self.query_cache[product_id] = {"path": trace_path, "scan_count": scan_count}

        return trace_path, scan_count

    def get_block_by_idx(self, idx):
        h = self.db.get(f"idx_{idx}".encode())
        return self.db.get(h) if h else None