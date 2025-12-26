import threading


class TransactionPool:
    """ 交易池：维护各分片内待上链的有序交易集合 """

    def __init__(self):
        self.pool = []
        self.lock = threading.Lock()

    def add_tx(self, tx):
        """ 验证签名并按时间戳排序存入池中 """
        if not tx.verify_tx(): return False
        with self.lock:
            # 去重判断
            if any(t.h == tx.h for t in self.pool): return False
            self.pool.append(tx)
            # 核心：基于业务时间戳排序，确保溯源环节逻辑顺序
            self.pool.sort(key=lambda x: x.ts)
        return True

    def get_batch(self, limit=10):
        """ 获取一批最早产生的交易用于区块打包 """
        with self.lock: return self.pool[:limit]

    def remove_txs(self, tx_hashes):
        """ 从池中移除已在区块中上链的交易 """
        with self.lock:
            self.pool = [t for t in self.pool if t.h not in tx_hashes]
