import asyncio
import sys
import random
import os

# --- 1. 修复 Windows 环境下的 ZMQ 异步警告与策略 ---
if sys.platform == 'win32':
    # ZMQ 在 Windows 上需要 Selector 事件策略才能正常处理文件描述符
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- 2. 导入项目自定义模块 ---
from Blockchain.node import BlockchainNode
from Blockchain.blockchain import Blockchain
from Blockchain.txpool import TransactionPool
from Blockchain.consensus import ConsensusEngine
from Blockchain.transaction import TraceTransaction

from Communication.server import ZMQServer
from Communication.client import ZMQClient
from Communication.service import TraceabilityService

from cryptography.hazmat.primitives import serialization


class AgroBlockchainApp:

    def __init__(self, node_id, port, loc_code, seed_ip=None, seed_port=None, mode="TW_BFT"):
        self.node_id = node_id
        self.port = port
        self.mode = mode  # 共识模式切换器

        # A. 初始化核心业务节点 (继承自 Node/node.py)
        self.node = BlockchainNode(node_id, port, loc_code)

        # B. 初始化区块链数据管理组件
        self.node.chain_manager = Blockchain(node_id)
        self.node.tx_pool = TransactionPool()

        # C. 初始化多算法共识引擎
        self.node.consensus_engine = ConsensusEngine(self.node, mode=mode)

        # D. 初始化网络通信层 (ZMQ)
        self.server = ZMQServer(port)
        self.client = ZMQClient()

        # E. 初始化业务逻辑中枢服务
        self.service = TraceabilityService(self.node, self.server, self.client)

        # 种子节点信息
        self.seed_ip = seed_ip
        self.seed_port = seed_port

    async def start(self):
        """ 启动节点所有并行异步任务 """
        print(f"==================================================")
        print(f"[*] 节点 ID: {self.node_id} | 分片: {self.node.shard_id}")
        print(f"[*] 监听端口: {self.port} | 共识模式: {self.mode}")
        print(f"[*] 地理编码: {self.node.location} | 信任分: {self.node.trust_total}")
        print(f"==================================================")

        # 1. 注册基础通信任务
        tasks = [
            # 监听全网广播 (TX, BLOCK, VOTE, HEARTBEAT)
            asyncio.create_task(self.client.listen(self.service.handle_incoming)),
            # 响应点对点同步请求 (JOIN, GET_BLOCK)
            asyncio.create_task(self.server.start_rep_handler(self.service.handle_rep_requests)),
            # 定期广播心跳及分片信息
            asyncio.create_task(self.service.heartbeat_loop()),
            # 运行共识决策循环 (出块提议)
            asyncio.create_task(self.service.run_consensus_logic()),
            # 运行邻居存活监控任务
            asyncio.create_task(self.service.monitor_peers_loop()),
            # 模拟现实中的农产品传感器数据输入
            asyncio.create_task(self.simulate_iot_data_flow()),
            # 并行运行信任度指标动态演变
            asyncio.create_task(self.run_trust_simulation_loop())
        ]

        # 2. 如果存在种子节点，执行初始握手与历史块同步
        if self.seed_ip and self.seed_port:
            self.client.add_peer(self.seed_ip, self.seed_port)
            # 异步执行块同步，不阻塞上述任务
            asyncio.create_task(self.service.sync_blockchain(self.seed_ip, self.seed_port))

        # 等待所有任务运行 (直到进程被终止)
        await asyncio.gather(*tasks)

    async def simulate_iot_data_flow(self):
        """ 模拟农业现场持续产生溯源交易 """
        # 生成一个模拟的溯源码
        t_code = TraceTransaction.generate_trace_code(f"BATCH-{self.node_id}")

        while True:
            # 模拟每 8-15 秒产生一笔农事记录（采收/包装/施肥等）
            await asyncio.sleep(random.randint(8, 15))

            # 只有处于非同步状态才产生交易
            if not self.service.syncing:
                pub_bytes = self.node.public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )

                # 构造溯源交易，自动关联本分片 ID
                tx = TraceTransaction(
                    p_id=t_code,
                    stg="PROCESSING",
                    op=self.node_id,
                    loc=self.node.location,
                    pub=pub_bytes,
                    ph=self.service.chain.tip,  # 指向当前链尖哈希
                    iot={"temp": round(random.uniform(15, 25), 2), "humidity": 65}
                )

                # 签名并广播
                tx.sign_tx(self.node.private_key)
                await self.server.broadcast("TX", tx.pack_signed())

                # 同时在本机模拟一次扫码动作
                self.service.chain.update_scan_stats(t_code)

    async def run_trust_simulation_loop(self):
        """ 后台运行信任指标演化逻辑，影响 TW_BFT 共识权重 """
        while True:
            self.node.simulate_dynamic_change()
            # 定期打印信任分方便观察
            if random.random() < 0.2:
                print(f"[Trust-Node] {self.node_id} Current Trust: {self.node.trust_total:.4f}")
            await asyncio.sleep(10)


# --- 3. 命令行入口解析 ---
if __name__ == "__main__":
    import sys

    # 预期参数: python main.py <NodeID> <Port> <LocCode> [SeedIP SeedPort Mode]
    if len(sys.argv) < 4:
        print("Usage: python main.py <ID> <Port> <LocationCode> [SeedIP SeedPort Mode]")
        sys.exit(1)

    # 基础参数
    arg_id = sys.argv[1]
    arg_port = int(sys.argv[2])
    arg_loc = sys.argv[3]

    # 扩展参数处理 (适配 Benchmark 脚本的 None 传递)
    arg_seed_ip = sys.argv[4] if len(sys.argv) > 4 else None
    if arg_seed_ip == "None": arg_seed_ip = None

    arg_seed_port = int(sys.argv[5]) if len(sys.argv) > 5 else None

    # 共识模式 (TW_BFT, PBFT, PoW)
    arg_mode = sys.argv[6] if len(sys.argv) > 6 else "TW_BFT"

    # 实例化并运行应用
    app = AgroBlockchainApp(
        node_id=arg_id,
        port=arg_port,
        loc_code=arg_loc,
        seed_ip=arg_seed_ip,
        seed_port=arg_seed_port,
        mode=arg_mode
    )

    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        print(f"\n[!] 节点 {arg_id} 正在安全关闭...")
    except Exception as e:
        print(f"\n[X] 运行时异常: {e}")