import threading


class TransactionPool:
    def __init__(self):
        self.pool = []
        # 新增：记录已见过的交易哈希，防止重复处理和广播风暴
        self.seen_hashes = set()
        self.lock = threading.Lock()

    def is_new(self, tx_hash):
        """ 线程安全地检查并记录新哈希 """
        with self.lock:
            if tx_hash in self.seen_hashes:
                return False
            self.seen_hashes.add(tx_hash)

            # 可选：防止内存溢出，限制记录最近的 10000 条
            if len(self.seen_hashes) > 10000:
                # 简单处理：清空（实际系统中应使用 LRU 缓存）
                self.seen_hashes.clear()
            return True

    def add_tx(self, tx):
        """ 存入池中。调用此方法前应先通过 is_new 过滤 """
        if not tx.verify_tx():
            return False

        with self.lock:
            # 去重：如果池中已存在则不处理
            if any(t.h == tx.h for t in self.pool):
                return False
            self.pool.append(tx)
            self.pool.sort(key=lambda x: x.ts)
            return True

    def get_batch(self, limit=20):
        with self.lock:
            return self.pool[:limit]

    def remove_txs(self, tx_hashes):
        with self.lock:
            self.pool = [t for t in self.pool if t.h not in tx_hashes]