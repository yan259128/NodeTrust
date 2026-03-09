import plyvel
import Blockchain.block as block
from Blockchain.transaction import TraceTransaction


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

        self.tip, self.height = None, -1
        self._check_genesis()

    def _check_genesis(self):
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
        """ 写入区块并同步更新状态索引与防伪计数 """
        self.db.put(b_obj.h.encode(), b_obj.pack())
        self.db.put(f"idx_{b_obj.idx}".encode(), b_obj.h.encode())
        self.db.put(b'l', b_obj.h.encode())

        # 更新状态索引：遍历区块内所有交易
        for tx_bin in b_obj.txs:
            tx = TraceTransaction.unpack(tx_bin)
            # a. 存储交易明细以便回溯
            self.tx_index.put(tx.h.encode(), tx_bin)
            # b. 更新该产品的最新交易哈希指向
            self.state_db.put(tx.p_id.encode(), tx.h.encode())
            # c. 如果是扫码存证，累加频次
            if tx.op == TraceTransaction.OP_SCAN:
                count_key = f"cnt_{tx.p_id}".encode()
                cur_cnt = int(self.anti_fake_db.get(count_key) or b'0')
                self.anti_fake_db.put(count_key, str(cur_cnt + 1).encode())

        self.tip = b_obj.h
        self.height = b_obj.idx

    def add_block(self, b_obj):
        if b_obj.prev_h != self.tip: return False
        self.save_block(b_obj)
        return True

    def get_block_by_idx(self, idx):
        h = self.db.get(f"idx_{idx}".encode())
        return self.db.get(h) if h else None

    def trace_back(self, product_id):
        """
        核心算法：基于反向指针的高速溯源回溯
        复杂度：O(k)，k为该产品的流转环节数，与账本总量无关
        """
        trace_path = []
        # 1. 从状态库获取该产品的最后一次交易哈希
        latest_tx_h = self.state_db.get(product_id.encode())
        if not latest_tx_h: return []

        curr_hash = latest_tx_h.decode()
        # 2. 沿着 ph (prev_hash) 指针逆向回溯
        while curr_hash:
            tx_data = self.tx_index.get(curr_hash.encode())
            if not tx_data: break

            tx = TraceTransaction.unpack(tx_data)
            trace_path.append(tx)

            # 移动到前一个环节
            curr_hash = tx.ph
            if not curr_hash or curr_hash == "": break

        return trace_path