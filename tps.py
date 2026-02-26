import time
import psutil
import zmq
import os
import glob
import threading

import main
from Blockchain.transaction import TraceTransaction
from cryptography.hazmat.primitives.asymmetric import ed25519

NODE_COUNT = 4
TX_INLET_PORT = 3350  # Node 0 的交易入口
TEST_DURATION = 10  # 测试时长 (秒)
THREAD_COUNT = 4  # 注入线程数 (建议根据 CPU 核心数调整)
TX_PER_THREAD_SLEEP = 0.01  # 每个线程两次发送间的间隔 (0.01s 约等于单线程 100TPS)

Mode = main.Mode

class BenchmarkStats:
    def __init__(self):
        self.cpu_usage = []
        self.stop_event = threading.Event()


stats = BenchmarkStats()


def get_performance_metrics():
    """ 解析日志统计性能 """
    tx_entries = {}
    commit_counts = 0
    latencies = []

    # 获取当前目录下所有节点日志
    log_files = glob.glob("node_*.log")
    for f_path in log_files:
        # 使用 errors='ignore' 防止读取冲突
        with open(f_path, "r", encoding="utf-8", errors='ignore') as f:
            for line in f:
                if "[BENCHMARK]" in line:
                    parts = line.strip().split('|')
                    try:
                        if "TX_ENTRY" in parts[0]:
                            tx_entries[parts[1]] = float(parts[2])
                        elif "BLOCK_COMMIT" in parts[0]:
                            tx_hashes = parts[2].split(',')
                            c_time = float(parts[3])
                            commit_counts += len(tx_hashes)
                            for h in tx_hashes:
                                if h in tx_entries:
                                    latencies.append(c_time - tx_entries[h])
                    except (IndexError, ValueError):
                        continue
    return commit_counts, latencies


def tx_injector_worker(thread_id):
    """ 交易注入子线程 """
    # 每个线程创建独立的 ZMQ 上下文和 Socket 提高吞吐量
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://127.0.0.1:{TX_INLET_PORT}")

    # 每个线程使用独立的签名私钥 (模拟不同用户)
    priv_key = ed25519.Ed25519PrivateKey.generate()
    pub_bytes = priv_key.public_key().public_bytes_raw()

    count = 0
    while not stats.stop_event.is_set():
        # 构造、签名、发送
        tx = TraceTransaction(f"BATCH_{thread_id}", "PROD", "STEP_N", 1001, pub_bytes)
        tx.sign_tx(priv_key)
        sock.send(tx.pack_signed())

        count += 1
        if TX_PER_THREAD_SLEEP > 0:
            time.sleep(TX_PER_THREAD_SLEEP)

    sock.close()
    ctx.term()
    print(f"[Thread-{thread_id}] Finished. Total injected: {count}")


def monitor_system():
    """ 监控系统 CPU 占用 """
    while not stats.stop_event.is_set():
        stats.cpu_usage.append(psutil.cpu_percent(interval=1))


def run_test():
    print(f"[*] 节点总数: {NODE_COUNT}")
    print(f"[*] 注入线程: {THREAD_COUNT}")
    print(f"[*] 预计持续: {TEST_DURATION}s")
    print(f"[*] 目标端口: {TX_INLET_PORT}")
    time.sleep(2)

    # 1. 启动 CPU 监控线程
    monitor_thread = threading.Thread(target=monitor_system)
    monitor_thread.start()

    # 2. 启动多个交易注入线程
    injectors = []
    for i in range(THREAD_COUNT):
        t = threading.Thread(target=tx_injector_worker, args=(i,))
        injectors.append(t)
        t.start()

    print(f"[*] 测试正在运行...")

    # 3. 等待测试时长
    try:
        time.sleep(TEST_DURATION)
    except KeyboardInterrupt:
        print("[!] 收到中断指令，提前结束测试...")
    finally:
        # 4. 停止所有线程
        stats.stop_event.set()
        for t in injectors:
            t.join()
        monitor_thread.join()

    print("[*] 注入完成，等待 5s 待最后区块入库...")
    time.sleep(5)

    # 5. 分析日志统计结果
    total_tx, latencies = get_performance_metrics()

    avg_tps = total_tx / TEST_DURATION
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    avg_cpu = sum(stats.cpu_usage) / len(stats.cpu_usage) if stats.cpu_usage else 0

    print("\n" + "=" * 45)
    print(f"Win10 区块链手动测试报告 (多线程版){Mode}")
    print("=" * 45)
    print(f"上链总数:       {total_tx} 笔")
    print(f"实测平均 TPS:   {avg_tps:.2f} tx/s")
    print(f"平均确认延迟:   {avg_lat:.4f} s")
    print(f"系统平均 CPU:   {avg_cpu:.1f} %")
    print(f"注入线程数:     {THREAD_COUNT}")
    print("=" * 45)


if __name__ == "__main__":
    # 运行前确保没有旧日志干扰，或者 main.py 启动时会自动覆盖
    run_test()