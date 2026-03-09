import hashlib
import msgpack
import time
import uuid
from cryptography.hazmat.primitives.asymmetric import ed25519
from util.parameter import ENABLE_SHARDING,SHARD_COUNT


class TraceTransaction:
    """
    溯源交易类：记录农产品全生命周期环节

    变量解释：
    p_id      (product_id):      产品的唯一溯源码或批次ID。
    stg       (stage):           业务阶段（如：采摘、加工、冷链、零售）。
    op        (operation):       操作类型（PRODUCE:生产, TRANS:运输, SCAN:扫码存证）。
    loc       (location):        地理位置编码（如 110101），前两位用于分片路由。
    shard_id  (shard_id):        根据地理位置计算出的分片ID。
    ph        (prev_hash):       反向指针，指向该产品上一个环节的交易哈希。实现 O(k) 回溯的核心。
    iot       (iot_data):        关联的IoT设备数据（如温度、湿度、经纬度）。
    ts        (timestamp):       交易生成的毫秒级时间戳。
    pub       (public_key):      操作员或节点的公钥。
    sig       (signature):       数字签名，保证数据不可篡改。
    h         (hash):            当前交易的唯一哈希标识。
    """

    # 定义操作常量
    OP_PRODUCE = "PRODUCE"
    OP_TRANSPORT = "TRANS"
    OP_SCAN = "SCAN"  # 查询即存证类型

    __slots__ = ['p_id', 'stg', 'op', 'loc', 'shard_id', 'ph', 'iot', 'ts', 'pub', 'sig', 'h']

    def __init__(self, p_id, stg, op, loc, pub, ph="", iot=None, ts=None):
        self.p_id = p_id
        self.stg = stg
        self.op = op
        self.loc = loc
        self.shard_id = f"Shard_{str(loc)[:2]}"
        self.pub = pub
        self.ph = ph  # 关键：指向前序交易哈希
        self.iot = iot or {}
        self.ts = ts or int(time.time() * 1000)
        self.sig, self.h = b"", ""
        self.ts = ts if ts else time.time()

        # 修改点：根据开关决定分片 ID
        if ENABLE_SHARDING:
            # self.shard_id = f"Shard_{str(loc)[:2]}"
            shard_index = loc % SHARD_COUNT
            self.shard_id = f"Shard_{shard_index}"
        else:
            self.shard_id = "GLOBAL_SHARD"

    def get_raw_data(self):
        """ 获取待签名数据序列 """
        return msgpack.packb([
            self.p_id, self.stg, self.op, self.loc,
            self.shard_id, self.ph, self.iot, self.ts, self.pub
        ])

    def sign_tx(self, priv_key):
        """ 执行私钥签名并生成交易哈希 """
        raw_data = self.get_raw_data()
        self.sig = priv_key.sign(raw_data)
        full_data = msgpack.packb([
            self.p_id, self.stg, self.op, self.loc,
            self.shard_id, self.ph, self.iot, self.ts,
            self.pub, self.sig
        ])
        self.h = hashlib.blake2b(full_data, digest_size=20).hexdigest()

    def verify_tx(self):
        """ 验证签名有效性 """
        try:
            pk = ed25519.Ed25519PublicKey.from_public_bytes(self.pub)
            pk.verify(self.sig, self.get_raw_data())
            return True
        except:
            return False

    def pack_signed(self):
        return msgpack.packb([
            self.p_id, self.stg, self.op, self.loc,
            self.shard_id, self.ph, self.iot, self.ts,
            self.pub, self.sig
        ])

    @classmethod
    def unpack(cls, bin_data):
        d = msgpack.unpackb(bin_data)
        obj = cls(d[0], d[1], d[2], d[3], d[8], d[5], d[6], d[7])
        obj.shard_id, obj.sig = d[4], d[9]
        obj.h = hashlib.blake2b(bin_data, digest_size=20).hexdigest()
        return obj