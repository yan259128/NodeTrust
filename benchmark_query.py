import subprocess
import time
import os
import sys
import psutil
import pandas as pd
import numpy as np
import zmq
import zmq.asyncio
import asyncio
import random
import uuid
import traceback
import csv
import zlib
from datetime import datetime
from Blockchain.transaction import TraceTransaction
from cryptography.hazmat.primitives.asymmetric import ed25519
from util.parameter import BASE_PORT, NODE_COUNT, SHARD_COUNT, ENABLE_SHARDING

# --- 极致性能配置 ---
CONCURRENT_SESSIONS_PER_NODE = 50  # 每个节点分配的长连接数
TOTAL_TEST_QUERIES = 10000  # 总压测请求数

# --- 路径与配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
DATASET_PATH = os.path.join(BASE_DIR, "china_agri_traceability_v10_final.csv")
QUERY_RESULT_FILE = os.path.join(BASE_DIR, "query_benchmark_results.csv")
INIT_FLAG_FILE = os.path.join(BASE_DIR, ".blockchain_initialized")


def get_p_id_shard(p_id):
    """ 计算数据属于哪个分片索引 (0 到 SHARD_COUNT-1) """
    return zlib.adler32(p_id.encode('utf-8')) % SHARD_COUNT


class TraceabilityBenchmark:
    def __init__(self, total_queries=TOTAL_TEST_QUERIES):
        self.total_queries = total_queries
        self.df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
        self.priv_key = ed25519.Ed25519PrivateKey.generate()
        self.pub_bytes = self.priv_key.public_key().public_bytes_raw()

        # 有效 ID 池
        self.valid_pids = self.df['溯源批次码'].head(4000).tolist()
        # 选定一个 ID 专门用于模拟“克隆扫码攻击”
        self.attack_target_id = self.valid_pids[0]

        self.query_latencies = []
        self.correct_responses = 0
        self.alerts_detected = 0  # 成功预警次数
        self.node_processes = []

        self.ctx = zmq.asyncio.Context()
        # 资源池：{shard_idx: asyncio.Queue[(node_id, socket)]}
        self.socket_pools = {i: asyncio.Queue() for i in range(SHARD_COUNT)}

    def is_first_run(self):
        return not os.path.exists(INIT_FLAG_FILE)

    def start_nodes(self):
        print(f"[*] 启动 {NODE_COUNT} 个区块链节点...")
        for i in range(NODE_COUNT):
            port = BASE_PORT + i
            loc_code = 100 + i
            p = subprocess.Popen(
                [PYTHON_EXE, MAIN_SCRIPT, str(i), str(port), str(loc_code)],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
                cwd=BASE_DIR
            )
            self.node_processes.append(p)
        print(f"[*] 节点启动中，等待 12s 初始化数据库与网络...")
        time.sleep(12)

    async def init_pools(self):
        print(f"[*] 正在为每个节点初始化 {CONCURRENT_SESSIONS_PER_NODE} 个长连接...")
        for i in range(NODE_COUNT):
            # 对齐 main.py 中的分片逻辑: (100 + i) % SHARD_COUNT
            shard_idx = (100 + i) % SHARD_COUNT
            port = BASE_PORT + i + 100

            for _ in range(CONCURRENT_SESSIONS_PER_NODE):
                sock = self.ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.RCVTIMEO, 5000)
                sock.setsockopt(zmq.SNDTIMEO, 5000)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(f"tcp://127.0.0.1:{port}")
                self.socket_pools[shard_idx].put_nowait((i, sock))
        print(f"[*] 连接池就绪。")

    async def inject_all_data(self):
        """ 阶段 1: 生产与加工双环节布置 """
        print(f"[*] 首次启动：注入 8000 笔具有 O(k) 指针的链式轨迹数据...")
        sock = self.ctx.socket(zmq.PUSH)
        sock.connect(f"tcp://127.0.0.1:{BASE_PORT + 50}")

        for i, p_id in enumerate(self.valid_pids):
            shard_idx = get_p_id_shard(p_id)
            loc = 100 + shard_idx  # 确保注入分片归属正确

            # 生产环节
            tx1 = TraceTransaction(p_id, "生产环节", "PRODUCE", loc, self.pub_bytes)
            tx1.sign_tx(self.priv_key)
            await sock.send(tx1.pack_signed())

            # 加工环节 (ph 指向 tx1)
            tx2 = TraceTransaction(p_id, "加工环节", "TRANS", loc, self.pub_bytes, ph=tx1.h)
            tx2.sign_tx(self.priv_key)
            await sock.send(tx2.pack_signed())

            if (i + 1) % 500 == 0: print(f"    - 已发送 {i + 1} / 4000 组...")

        sock.close()
        print("[*] 注入完成，等待 90s 待分片共识落盘...")
        await asyncio.sleep(90)
        with open(INIT_FLAG_FILE, "w") as f:
            f.write("done")

    async def check_data_ready(self):
        """ 寻址预校验 """
        print("[*] 正在验证分片数据寻址一致性...")
        test_pid = self.valid_pids[1]  # 用第二个 ID 校验
        shard_idx = get_p_id_shard(test_pid)
        node_id, sock = await self.socket_pools[shard_idx].get()
        try:
            await sock.send_json({"type": "QUERY_TRACE", "p_id": test_pid})
            res = await sock.recv_json()
            if res.get("status") == "OK":
                print(f"[✔] 校验成功！分片 {shard_idx} 数据寻址 100% 对齐。")
                return True
        except Exception as e:
            print(f"[!] 校验失败: {e}")
        finally:
            self.socket_pools[shard_idx].put_nowait((node_id, sock))
        return False

    async def perform_stress_queries(self):
        """ 阶段 2: 极限压力查询与防伪逻辑验证 """
        print(f"[*] 开始高并发压测：查询总量 {self.total_queries}...")
        start_time = time.perf_counter()

        tasks = []
        # 信号量控制并发窗口，保护本地 CPU
        sem = asyncio.Semaphore(1500)

        for i in range(self.total_queries):
            rand = random.random()
            if rand < 0.4:
                # 场景 1: 伪造码查询 (10%)
                p_id, expected = f"FAKE_CODE_{uuid.uuid4().hex[:6]}", "NOT_FOUND"
            elif rand < 0.8:
                # 场景 2: 克隆攻击/重复扫码 (10%)
                p_id, expected = self.attack_target_id, "ALERT_TEST"
            else:
                # 场景 3: 正常溯源回溯 (80%)
                p_id, expected = random.choice(self.valid_pids), "OK"

            tasks.append(self.single_query_optimized(p_id, expected, sem))

        await asyncio.gather(*tasks)

        duration = time.perf_counter() - start_time
        res = self.calculate_results(duration)
        self.save_results_to_csv(res)
        self.print_report(res)

    async def single_query_optimized(self, p_id, expected, sem):
        async with sem:
            shard_idx = get_p_id_shard(p_id)
            pool = self.socket_pools[shard_idx]
            node_id, sock = await pool.get()

            q_start = time.perf_counter()
            try:
                await sock.send_json({"type": "QUERY_TRACE", "p_id": p_id, "loc": 888})
                res = await sock.recv_json()
                self.query_latencies.append(time.perf_counter() - q_start)

                # --- 核心验证逻辑 ---
                status = res.get("status")

                if expected == "NOT_FOUND":
                    if status == "NOT_FOUND": self.correct_responses += 1

                elif expected == "ALERT_TEST":
                    if status == "OK":
                        self.correct_responses += 1
                        if res.get("is_alert"): self.alerts_detected += 1

                elif expected == "OK":
                    # 正常溯源必须返回 OK 且包含轨迹数据（生产和加工）
                    if status == "OK" and len(res.get("data", [])) >= 2:
                        self.correct_responses += 1

                pool.put_nowait((node_id, sock))

            except Exception:
                # Socket 异常自愈
                sock.close()
                new_sock = self.ctx.socket(zmq.REQ)
                new_sock.setsockopt(zmq.RCVTIMEO, 5000)
                new_sock.connect(f"tcp://127.0.0.1:{BASE_PORT + node_id + 100}")
                pool.put_nowait((node_id, new_sock))

    def calculate_results(self, duration):
        avg_lat = np.mean(self.query_latencies) * 1000 if self.query_latencies else 0
        qps = self.total_queries / duration
        accuracy = (self.correct_responses / self.total_queries) * 100
        return {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Sharding": "ON" if ENABLE_SHARDING else "OFF",
            "Shards": SHARD_COUNT,
            "Nodes": NODE_COUNT,
            "QPS": round(qps, 2),
            "AvgLat_ms": round(avg_lat, 2),
            "Accuracy": f"{round(accuracy, 2)}%",
            "AlertsFound": self.alerts_detected
        }

    def save_results_to_csv(self, data):
        file_exists = os.path.isfile(QUERY_RESULT_FILE)
        with open(QUERY_RESULT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(data)

    def print_report(self, res):
        print("\n" + "=" * 50)
        print("高速溯源与动态防伪体系压测最终结论")
        print("-" * 50)
        print(f"吞吐量 QPS:   {res['QPS']}")
        print(f"平均延迟:      {res['AvgLat_ms']} ms")
        print(f"执行准确率:    {res['Accuracy']} (拦截/回溯成功率)")
        print(f"预警触发次数:  {res['AlertsFound']} (识别克隆扫码攻击)")
        print(f"结果已写入:    {QUERY_RESULT_FILE}")
        print("=" * 50 + "\n")

    def cleanup(self):
        print("[*] 正在清理进程资源...")
        for pool in self.socket_pools.values():
            while not pool.empty():
                _, sock = pool.get_nowait()
                sock.close()
        self.ctx.term()
        for p in self.node_processes:
            try:
                parent = psutil.Process(p.pid)
                for child in parent.children(recursive=True): child.kill()
                parent.kill()
            except:
                pass


async def main():
    benchmark = TraceabilityBenchmark(total_queries=10000)
    try:
        benchmark.start_nodes()
        await benchmark.init_pools()

        if benchmark.is_first_run():
            await benchmark.inject_all_data()
        else:
            print("[*] 检测到持久化数据，跳过注入，直接开始高速压测...")

        if await benchmark.check_data_ready():
            await benchmark.perform_stress_queries()
        else:
            print("[✘] 致命错误：分片路由验证失败。")
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())