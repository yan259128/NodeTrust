import asyncio, sys, random
from Blockchain.node import BlockchainNode
from Blockchain.blockchain import Blockchain
from Blockchain.txpool import TransactionPool
from Blockchain.consensus import VRF_BFT_Consensus
from Blockchain.transaction import TraceTransaction
from Communication.server import ZMQServer
from Communication.client import ZMQClient
from Communication.service import TraceabilityService
from cryptography.hazmat.primitives import serialization

# 解决 Windows ZMQ 异步策略兼容性问题
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main(n_id, port, location_code, s_ip=None, s_port=None):
    # 1. 初始化核心节点
    node = BlockchainNode(n_id, port, location_code)
    node.chain_manager = Blockchain(n_id)
    node.tx_pool = TransactionPool()
    node.consensus_engine = VRF_BFT_Consensus(node)

    # 2. 通信组件实例化
    server = ZMQServer(port)
    client = ZMQClient()
    service = TraceabilityService(node, server, client)

    # 3. 注册并行异步后台任务
    tasks = [
        asyncio.create_task(client.listen(service.handle_incoming)),  # 全网广播监听
        asyncio.create_task(server.start_rep_handler(service.handle_rep_requests)),  # RPC响应
        asyncio.create_task(service.heartbeat_loop()),  # 周期心跳
        asyncio.create_task(service.run_consensus_logic()),  # 共识抽奖逻辑
        asyncio.create_task(simulate_agri_trace(service, node))  # 模拟业务
    ]

    # 4. 执行节点同步逻辑
    if s_ip:
        client.add_peer(s_ip, s_port)
        await service.sync_blockchain(s_ip, s_port)

    print(f"[*] Industrial Node {n_id} (Shard: {node.shard_id}) started.")
    await asyncio.gather(*tasks)


async def simulate_agri_trace(service, node):
    """ 模拟农产品溯源业务：环节存证与扫码防伪 """
    # 生成防伪溯源码
    t_code = TraceTransaction.generate_trace_code("LOT2025-AX")
    pub = node.public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    while True:
        await asyncio.sleep(random.randint(15, 20))
        # 1. 发起一笔溯源交易 (例如：采收环节)
        tx = TraceTransaction(t_code, "HARVEST", node.Id, node.location, pub)
        tx.sign_tx(node.private_key)
        await service.server.broadcast("TX", tx.pack_signed())

        # 2. 模拟终端扫码查询行为：更新并查询扫码统计
        sc_cnt = service.chain.update_scan_stats(t_code)
        print(f"[IOT] Product {t_code} recorded & scanned. Total Scan Count: {sc_cnt}")


if __name__ == "__main__":
    # 启动命令：python main.py NodeA 6000 37
    if len(sys.argv) < 4:
        print("Usage: python main.py <ID> <Port> <LocationCode> [SeedIP SeedPort]")
    else:
        asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3],
                         sys.argv[4] if len(sys.argv) > 4 else None,
                         int(sys.argv[5]) if len(sys.argv) > 5 else None))
