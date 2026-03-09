import time
import psutil
import zmq
import os
import glob
import threading
import pandas as pd
import numpy as np
from Blockchain.transaction import TraceTransaction
from cryptography.hazmat.primitives.asymmetric import ed25519
from util.parameter import NODE_COUNT, ENABLE_SHARDING

# 配置
DATASET_PATH = os.path.join("Dataset", "china_agri_traceability_v10_final.csv")
TX_INLET_PORT = 3350
TEST_DURATION = 15
THREAD_COUNT = 2  # 模拟数据处理较重，建议减小线程数提高质量
TX_PER_THREAD_SLEEP = 0.05


class BenchmarkStats:
    def __init__(self):
        self.cpu_usage = []
        self.stop_event = threading.Event()
        self.dataset = None


stats = BenchmarkStats()


def load_dataset():
    """ 加载并预处理数据集 """
    if not os.path.exists(DATASET_PATH):
        print(f"[!] 找不到数据集: {DATASET_PATH}，请检查路径")
        return None
    # 读取 CSV (处理 UTF-8 BOM)
    df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
    return df


def get_performance_metrics():
    """ 解析日志统计性能 (增加异常处理防止 Windows 文件锁) """
    tx_entries = {}
    commit_counts = 0
    latencies = []
    log_files = glob.glob("node_*.log")

    for f_path in log_files:
        try:
            with open(f_path, "r", encoding="utf-8", errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    if "[BENCHMARK]" in line:
                        parts = line.strip().split('|')
                        if "TX_ENTRY" in parts[0]:
                            tx_entries[parts[1]] = float(parts[2])
                        elif "BLOCK_COMMIT" in parts[0]:
                            if len(parts) < 4: continue
                            tx_hashes = parts[2].split(',')
                            c_time = float(parts[3])
                            commit_counts += len(tx_hashes)
                            for h in tx_hashes:
                                if h in tx_entries:
                                    latencies.append(c_time - tx_entries[h])
        except:
            continue
    return commit_counts, latencies


def tx_injector_worker(thread_id, df_chunk):
    """ 交易注入子线程：使用真实数据构造交易链 """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://127.0.0.1:{TX_INLET_PORT}")

    priv_key = ed25519.Ed25519PrivateKey.generate()
    pub_bytes = priv_key.public_key().public_bytes_raw()

    idx = 0
    row_count = len(df_chunk)

    while not stats.stop_event.is_set() and idx < row_count:
        row = df_chunk.iloc[idx]
        p_id = row['溯源批次码']

        # 确定 loc_code 以匹配分片 (100, 101, 102...)
        # 循环分配给不同分片
        loc_code = 100 + (idx % NODE_COUNT)

        # --- 环节 1: 生产 (PRODUCE) ---
        tx1 = TraceTransaction(p_id, "产地环节", TraceTransaction.OP_PRODUCE, loc_code, pub_bytes)
        tx1.iot = {"crop": row['作物名称'], "weight": row['采摘重量(kg)'], "prov": row['生产省份']}
        tx1.sign_tx(priv_key)
        sock.send(tx1.pack_signed())

        # --- 环节 2: 加工 (TRANS) ---
        tx2 = TraceTransaction(p_id, "加工环节", TraceTransaction.OP_TRANSPORT, loc_code, pub_bytes, ph=tx1.h)
        tx2.iot = {"factory": row['加工中心'], "proc": row['加工具体工序']}
        tx2.sign_tx(priv_key)
        sock.send(tx2.pack_signed())

        # --- 环节 3: 零售 (SCAN) ---
        tx3 = TraceTransaction(p_id, "销售环节", TraceTransaction.OP_SCAN, loc_code, pub_bytes, ph=tx2.h)
        tx3.iot = {"city": row['销售城市'], "store": row['销售区县网点']}
        tx3.sign_tx(priv_key)
        sock.send(tx3.pack_signed())

        idx += 1
        if TX_PER_THREAD_SLEEP > 0:
            time.sleep(TX_PER_THREAD_SLEEP)

    sock.close()
    ctx.term()


def run_test():
    df = load_dataset()
    if df is None: return

    print(f"[*] 已加载数据集，共 {len(df)} 条原始记录")
    # 分割数据给不同线程
    chunks = np.array_split(df, THREAD_COUNT)

    monitor_thread = threading.Thread(target=lambda: [stats.cpu_usage.append(psutil.cpu_percent(interval=1)) for _ in
                                                      iter(int, stats.stop_event.is_set())])
    monitor_thread.daemon = True
    monitor_thread.start()

    injectors = []
    start_time = time.time()
    for i in range(THREAD_COUNT):
        t = threading.Thread(target=tx_injector_worker, args=(i, chunks[i]))
        injectors.append(t)
        t.start()

    try:
        time.sleep(TEST_DURATION)
    finally:
        stats.stop_event.set()
        for t in injectors: t.join()

    print("[*] 注入完成，等待共识落盘...")
    time.sleep(5)

    total_committed, latencies = get_performance_metrics()
    avg_tps = total_committed / (time.time() - start_time - 5)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "=" * 45)
    print(f"基于真实数据集的测试报告")
    print(f"分片机制: {'开启' if ENABLE_SHARDING else '关闭'}")
    print(f"总上链交易数: {total_committed}")
    print(f"平均 TPS:      {avg_tps:.2f} tx/s")
    print(f"平均延迟:      {avg_lat:.4f} s")
    print("=" * 45)

    return {"avg_tps": avg_tps, "avg_lat": avg_lat, "total_tx": total_committed,
            "avg_cpu": np.mean(stats.cpu_usage) if stats.cpu_usage else 0, "mode": "TW_BFT", "threads": THREAD_COUNT,
            "duration": TEST_DURATION}


if __name__ == "__main__":
    run_test()