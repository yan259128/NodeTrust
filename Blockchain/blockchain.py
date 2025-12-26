import plyvel
from block import Block

class Blockchain:
    """ 账本管理器：管理磁盘持久化、创世块、高度索引及防伪统计 """
    def __init__(self, node_id):
        self.db = plyvel.DB(f"./db_chain_{node_id}", create_if_missing=True)
        self.tip, self.height = None, -1
        self._check_genesis()

    def _check_genesis(self):
        """ 检查本地数据库，若为空则强制初始化创世区块 """
        last_h = self.db.get(b'l')
        if not last_h:
            gen = Block(0, [], "0"*64, "GENESIS", b"0"*32, b"0"*64, b"0"*64)
            gen.h = "0" * 64
            self.save_block(gen)
        else:
            self.tip = last_h.decode()
            self.height = Block.unpack(self.db.get(last_h)).idx

    def save_block(self, block):
        """ 将区块写入磁盘并更新高度和最新哈希索引 """
        self.db.put(block.h.encode(), block.pack())
        self.db.put(f"idx_{block.idx}".encode(), block.h.encode())
        self.db.put(b'l', block.h.encode())
        self.tip, self.height = block.h, block.idx

    def add_block(self, block):
        """ 上链新区块，需检查前序哈希一致性 """
        if block.prev_h != self.tip: return False
        self.save_block(block)
        return True

    def get_block_by_idx(self, idx):
        """ 获取指定高度的区块原始数据 (用于节点同步) """
        h = self.db.get(f"idx_{idx}".encode())
        return self.db.get(h) if h else None

    def update_scan_stats(self, p_id):
        """ 防伪统计：持久化记录溯源码的扫码查询次数 """
        key = f"scan_{p_id}".encode()
        curr = self.db.get(key)
        new_val = int(curr.decode()) + 1 if curr else 1
        self.db.put(key, str(new_val).encode())
        return new_val

    def get_scan_count(self, p_id):
        """ 查询溯源码已被扫码的次数 """
        val = self.db.get(f"scan_{p_id}".encode())
        return int(val.decode()) if val else 0