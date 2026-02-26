import threading


class TransactionPool:
    def __init__(self):
        self.pool = []
        self.lock = threading.Lock()

    def add_tx(self, tx):
        """ 存入池中。如果是重复交易返回 False，新交易返回 True """
        if not tx.verify_tx():
            return False

        with self.lock:
            # 去重判断：如果哈希已存在，则不处理
            if any(t.h == tx.h for t in self.pool):
                return False

            self.pool.append(tx)
            # 按业务时间戳排序
            self.pool.sort(key=lambda x: x.ts)
            return True

    def get_batch(self, limit=20):
        with self.lock:
            return self.pool[:limit]

    def remove_txs(self, tx_hashes):
        with self.lock:
            self.pool = [t for t in self.pool if t.h not in tx_hashes]