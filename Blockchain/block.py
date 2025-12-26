import hashlib
import msgpack


class Block:
    """ 区块结构：包含 VRF 共识参数、交易集合及多方投票列表 """
    __slots__ = ['idx', 'txs', 'prev_h', 'miner', 'pub', 'pi', 'beta', 'sig', 'h', 'votes']
    def __init__(self, idx, txs, prev_h, miner, pub, pi, beta):
        self.idx, self.txs, self.prev_h = idx, txs, prev_h
        self.miner, self.pub, self.pi, self.beta = miner, pub, pi, beta
        self.sig, self.h, self.votes = b"", "", [] # votes 存储验证者的签名

    def sign_block(self, priv_key):
        """ 提议者对区块头进行签名 """
        header = msgpack.packb([self.idx, self.txs, self.prev_h, self.miner, self.beta])
        self.h = hashlib.blake2b(header, digest_size=32).hexdigest()
        self.sig = priv_key.sign(self.h.encode())

    def pack(self):
        """ 区块二进制打包 """
        return msgpack.packb([self.idx, self.txs, self.prev_h, self.miner, self.pub, self.pi, self.beta, self.sig, self.h, self.votes])

    @classmethod
    def unpack(cls, raw):
        """ 从原始字节恢复区块对象 """
        d = msgpack.unpackb(raw)
        b = cls(d[0], d[1], d[2], d[3], d[4], d[5], d[6])
        b.sig, b.h, b.votes = d[7], d[8], d[9]
        return b