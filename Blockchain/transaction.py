import hashlib
import msgpack
import time
import uuid
from cryptography.hazmat.primitives.asymmetric import ed25519


class TraceTransaction:
    """ 溯源交易：包含地理路由标识、防伪溯源码、指针链与数字签名 """
    __slots__ = ['p_id', 'stg', 'op', 'loc', 'shard_id', 'ph', 'iot', 'ts', 'pub', 'sig', 'h']

    def __init__(self, p_id, stg, op, loc, pub, ph="", iot=None, ts=None):
        self.p_id, self.stg, self.op, self.loc = p_id, stg, op, loc
        self.shard_id = f"Shard_{str(loc)[:2]}"  # 自动计算路由分片
        self.pub, self.ph = pub, ph
        self.iot = iot or {}
        self.ts = ts or int(time.time() * 1000)
        self.sig, self.h = b"", ""

    @staticmethod
    def generate_trace_code(batch_id):
        """ 为产品批次生成唯一的溯源码 """
        return hashlib.sha256(f"{batch_id}-{uuid.uuid4().hex}".encode()).hexdigest()[:12].upper()

    def get_raw_data(self):
        """ 获取待签名原始字节 """
        return msgpack.packb(
            [self.p_id, self.stg, self.op, self.loc, self.shard_id, self.ph, self.iot, self.ts, self.pub])

    def sign_tx(self, priv_key):
        """ 对交易执行私钥签名并计算唯一哈希 """
        raw_data = self.get_raw_data()
        self.sig = priv_key.sign(raw_data)
        # 计算包含签名的完整数据哈希作为交易ID
        full_data = msgpack.packb(
            [self.p_id, self.stg, self.op, self.loc, self.shard_id, self.ph, self.iot, self.ts, self.pub, self.sig])
        self.h = hashlib.blake2b(full_data, digest_size=20).hexdigest()

    def verify_tx(self):
        """ 验证交易签名合规性 """
        try:
            pk = ed25519.Ed25519PublicKey.from_public_bytes(self.pub)
            pk.verify(self.sig, self.get_raw_data())
            return True
        except:
            return False

    def pack_signed(self):
        """ 打包带签名的交易 """
        return msgpack.packb(
            [self.p_id, self.stg, self.op, self.loc, self.shard_id, self.ph, self.iot, self.ts, self.pub, self.sig])

    @classmethod
    def unpack(cls, bin_data):
        """ 解包二进制交易 """
        d = msgpack.unpackb(bin_data)
        obj = cls(d[0], d[1], d[2], d[3], d[8], d[5], d[6], d[7])
        obj.shard_id, obj.sig = d[4], d[9]
        obj.h = hashlib.blake2b(bin_data, digest_size=20).hexdigest()
        return obj