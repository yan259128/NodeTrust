import plyvel
import Blockchain.block as block


class Blockchain:
    def __init__(self, node_id):
        self.db = plyvel.DB(f"./db_chain_{node_id}", create_if_missing=True)
        self.tip, self.height = None, -1
        self._check_genesis()

    def _check_genesis(self):
        last_h = self.db.get(b'l')
        if not last_h:
            gen = block.Block(0, [], "0" * 64, "GENESIS", b"0" * 32, b"0" * 64, b"0" * 64)
            gen.h = "0" * 64
            self.save_block(gen)
        else:
            self.tip = last_h.decode()
            raw = self.db.get(self.tip.encode())
            self.height = block.Block.unpack(raw).idx

    def save_block(self, b_obj):
        # 写入数据库
        self.db.put(b_obj.h.encode(), b_obj.pack())
        self.db.put(f"idx_{b_obj.idx}".encode(), b_obj.h.encode())
        self.db.put(b'l', b_obj.h.encode())
        # 更新内存索引
        self.tip = b_obj.h
        self.height = b_obj.idx

    def add_block(self, b_obj):
        """ 严格的前序校验 """
        if b_obj.prev_h != self.tip:
            print(f"[!] Block Reject: Prev Hash {b_obj.prev_h[:8]} != Tip {self.tip[:8]}")
            return False

        # 校验高度是否冲突
        if b_obj.idx != self.height + 1:
            print(f"[!] Block Reject: Height {b_obj.idx} != Expected {self.height + 1}")
            return False

        self.save_block(b_obj)
        return True

    def get_block_by_idx(self, idx):
        h = self.db.get(f"idx_{idx}".encode())
        return self.db.get(h) if h else None