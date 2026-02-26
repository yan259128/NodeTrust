import asyncio
import sys
import zmq.asyncio
import msgpack

# 指定导入路径
from Blockchain.blockchain import Blockchain
from Blockchain.txpool import TransactionPool
from Blockchain.consensus import ConsensusEngine
from Blockchain.node import BlockchainNode
from Communication.server import ZMQServer
from Communication.client import ZMQClient
from Communication.service import TraceabilityService

Mode = "TW_BFT"
# Mode = "PoW"
# Mode = "PBFT"

class Logger:
    def __init__(self, node_id):
        self.terminal = sys.stdout
        self.log = open(f"node_{node_id}.log", "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


async def tx_receiver(port, service):
    """ 接收来自压测脚本的交易 PUSH """
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(f"tcp://*:{port + 50}")
    while True:
        payload = await sock.recv()
        # 直接调用 service 的交易处理逻辑（内含广播）
        await service._handle_transaction(payload)


async def run_node(node_id, port, location_code):
    # 1. 核心初始化
    node = BlockchainNode(node_id, port, location_code)
    node.chain_manager = Blockchain(node_id)
    node.tx_pool = TransactionPool()
    node.consensus_engine = ConsensusEngine(node, mode=Mode)

    # 2. 通讯初始化
    server = ZMQServer(port)
    client = ZMQClient()
    service = TraceabilityService(node, server, client)

    sys.stdout = Logger(node_id)

    # 3. 主动发现邻居 (Bootstrap)
    # 假设测试环境端口为 3300-3303
    for p in range(3300, 3304):
        if p != port:
            client.add_peer("127.0.0.1", p)

    # 4. 启动后台任务
    asyncio.create_task(server.start_rep_handler(service.handle_rep_requests))
    asyncio.create_task(client.listen(service.handle_incoming))
    asyncio.create_task(service.heartbeat_loop())
    asyncio.create_task(service.monitor_peers_loop())
    asyncio.create_task(service.run_consensus_logic())
    asyncio.create_task(tx_receiver(port, service))

    print(f"[*] Node {node_id} Online | Port: {port} | TX_Inlet: {port + 50}", flush=True)

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Windows 异步兼容修复
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if len(sys.argv) < 4:
        print("Usage: python main.py <id> <port> <loc_code>")
        sys.exit(1)

    n_id, n_port, n_loc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    try:
        asyncio.run(run_node(n_id, n_port, n_loc))
    except KeyboardInterrupt:
        pass