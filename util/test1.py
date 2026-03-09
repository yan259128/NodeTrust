import time
import psutil
import zmq
import glob
import threading

import main
from Blockchain.transaction import TraceTransaction
from cryptography.hazmat.primitives.asymmetric import ed25519
import random
from util.parameter import BASE_PORT, NODE_COUNT,ENABLE_SHARDING

TX_INLET_PORT = 3350  # Node 0 的交易入口
TEST_DURATION = 10  # 测试时长 (秒)
THREAD_COUNT = 10  # 注入线程数 (建议根据 CPU 核心数调整)
TX_PER_THREAD_SLEEP = 0.01  # 每个线程两次发送间的间隔 (0.01s 约等于单线程 100TPS)

Mode = main.Mode

class BenchmarkStats:
    def __init__(self):
        self.cpu_usage = []
        self.stop_event = threading.Event()


stats = BenchmarkStats()


def get_performance_metrics():
    tx_birth_map = {}
    tx_commit_map = {}

    # 获取当前目录下所有日志
    log_files = glob.glob("node_*.log")
    if not log_files:
        print("[!] 未找到任何 node_*.log 文件")
        return 0, []

    for f_path in log_files:
        try:
            with open(f_path, "r", encoding="utf-8", errors='ignore') as f:
                lines = f.readlines()

            for line in lines:
                if "[BENCHMARK]" not in line:
                    continue

                parts = [p.strip() for p in line.split('|')]

                # --- 解析 TX_ENTRY ---
                # 格式: [BENCHMARK] TX_ENTRY | hash | birth_ts | receive_ts
                if "TX_ENTRY" in parts[0] and len(parts) >= 3:
                    tx_hash = parts[1]
                    try:
                        birth_ts = float(parts[2])
                        if tx_hash not in tx_birth_map:
                            tx_birth_map[tx_hash] = birth_ts
                    except ValueError:
                        continue

                # --- 解析 BLOCK_COMMIT ---
                # 格式: [BENCHMARK] BLOCK_COMMIT | index | tx_hashes | commit_ts
                elif "BLOCK_COMMIT" in parts[0] and len(parts) >= 4:
                    tx_hashes_str = parts[2]
                    try:
                        commit_ts = float(parts[3])
                        tx_list = tx_hashes_str.split(',')
                        for tx_hash in tx_list:
                            if not tx_hash: continue
                            # 记录最早见到的提交时间
                            if tx_hash not in tx_commit_map:
                                tx_commit_map[tx_hash] = commit_ts
                    except ValueError:
                        continue

        except Exception as e:
            print(f"[*] 解析文件 {f_path} 时出错: {e}")

    # 计算最终统计
    latencies = []
    for tx_hash, c_time in tx_commit_map.items():
        if tx_hash in tx_birth_map:
            delay = c_time - tx_birth_map[tx_hash]
            if delay >= 0:
                latencies.append(delay)

    total_committed = len(tx_commit_map)
    print(f"[DEBUG] 解析完成: 捕获出生记录 {len(tx_birth_map)} 条, 提交记录 {total_committed} 条")

    return total_committed, latencies


def tx_injector_worker(thread_id):
    """ 交易注入子线程 """
    # 每个线程创建独立的 ZMQ 上下文和 Socket 提高吞吐量
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    target_node = random.randint(0, NODE_COUNT - 1)
    target_port = (BASE_PORT + target_node) + 50
    sock.connect(f"tcp://127.0.0.1:{target_port}")

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

    return {
        "mode": Mode,
        "total_tx": total_tx,
        "avg_tps": avg_tps,
        "avg_lat": avg_lat,
        "avg_cpu": avg_cpu,
        "duration": TEST_DURATION,
        "threads": THREAD_COUNT
    }


if __name__ == "__main__":
    # 运行前确保没有旧日志干扰，或者 main.py 启动时会自动覆盖
    run_test()