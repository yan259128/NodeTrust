import msgpack
import hashlib


class Block:
    # 增加 nonce 用于 PoW 计算
    __slots__ = ['idx', 'txs', 'prev_h', 'miner', 'pub', 'pi', 'beta', 'sig', 'h', 'votes', 'nonce']

    def __init__(self, idx, txs, prev_h, miner, pub, pi=b"", beta=b""):
        self.idx, self.txs, self.prev_h = idx, txs, prev_h
        self.miner, self.pub, self.pi, self.beta = miner, pub, pi, beta
        self.sig, self.h, self.votes = b"", "", []
        self.nonce = 0  # 初始随机数

    def calculate_header_hash(self):
        """ 计算区块哈希，包含 nonce 以适配 PoW """
        data = msgpack.packb([self.idx, self.txs, self.prev_h, self.miner, self.beta, self.nonce])
        return hashlib.blake2b(data, digest_size=32).hexdigest()

    def sign_block(self, priv_key):
        self.h = self.calculate_header_hash()
        self.sig = priv_key.sign(self.h.encode())

    def pack(self):
        return msgpack.packb([self.idx, self.txs, self.prev_h, self.miner, self.pub,
                              self.pi, self.beta, self.sig, self.h, self.votes, self.nonce])

    @classmethod
    def unpack(cls, raw):
        d = msgpack.unpackb(raw)
        b = cls(d[0], d[1], d[2], d[3], d[4], d[5], d[6])
        b.sig, b.h, b.votes, b.nonce = d[7], d[8], d[9], d[10]
        return b