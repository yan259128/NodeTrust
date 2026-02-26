import time
import psutil
import zmq
import os
import glob
from Blockchain.transaction import TraceTransaction
from cryptography.hazmat.primitives.asymmetric import ed25519

# 配置
NODE_COUNT = 4
TX_INLET_PORT = 3350  # Node 0 的交易入口
TEST_DURATION = 30


def get_performance_metrics():
    tx_entries = {}
    commit_counts = 0
    latencies = []

    # 读取所有节点的日志文件
    log_files = glob.glob("node_*.log")
    for f_path in log_files:
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                if "[BENCHMARK]" in line:
                    parts = line.strip().split('|')
                    if "TX_ENTRY" in parts[0]:
                        tx_entries[parts[1]] = float(parts[2])
                    elif "BLOCK_COMMIT" in parts[0]:
                        tx_hashes = parts[2].split(',')
                        c_time = float(parts[3])
                        commit_counts += len(tx_hashes)
                        for h in tx_hashes:
                            if h in tx_entries:
                                latencies.append(c_time - tx_entries[h])
    return commit_counts, latencies


def run_test():
    # # 1. 清理旧日志
    # for f in glob.glob("node_*.log"):
    #     os.remove(f)

    print(f"[*] 请确保 {NODE_COUNT} 个节点已手动启动...")
    print(f"[*] 准备注入交易至端口 {TX_INLET_PORT}...")
    time.sleep(2)

    # 2. 准备 ZMQ 发送
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://127.0.0.1:{TX_INLET_PORT}")

    priv_key = ed25519.Ed25519PrivateKey.generate()
    pub_bytes = priv_key.public_key().public_bytes_raw()

    # 3. 开始压测
    cpu_usage = []
    start_t = time.time()
    print(f"[*] 正在运行测试 ({TEST_DURATION}s)...")

    while time.time() - start_t < TEST_DURATION:
        # 记录 CPU
        cpu_usage.append(psutil.cpu_percent())

        # 发送 1 笔交易
        tx = TraceTransaction("TEST_PROD", "STEP1", "ADD", 1001, pub_bytes)
        tx.sign_tx(priv_key)
        sock.send(tx.pack_signed())

        time.sleep(0.05)  # 约 20 TPS

    print("[*] 测试完成，正在分析日志...")
    time.sleep(2)  # 等待最后的区块入库

    # 4. 统计
    total_tx, latencies = get_performance_metrics()
    avg_tps = total_tx / TEST_DURATION
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

    print("\n" + "=" * 40)
    print("      Win10 区块链手动测试报告")
    print("=" * 40)
    print(f"上链总数:    {total_tx}")
    print(f"平均 TPS:    {avg_tps:.2f} tx/s")
    print(f"平均延迟:    {avg_lat:.4f} s")
    print(f"系统 CPU:    {avg_cpu:.1f} %")
    print("=" * 40)


if __name__ == "__main__":
    run_test()