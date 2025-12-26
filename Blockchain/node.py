from Node.node import Node
from cryptography.hazmat.primitives.asymmetric import ed25519


class BlockchainNode(Node):
    """ 区块链扩展节点：集成地理分片属性、密钥管理与核心组件指针 """

    def __init__(self, node_id, port, location_code):
        super().__init__(node_id, port)
        self.location = location_code
        # 地理分片逻辑：根据位置前两位划分为片 S_i
        self.shard_id = f"Shard_{str(location_code)[:2]}"

        # 加密身份
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

        # 挂载组件
        self.chain_manager = None
        self.tx_pool = None
        self.consensus_engine = None