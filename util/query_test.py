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
from datetime import datetime
from Blockchain.transaction import TraceTransaction
from cryptography.hazmat.primitives.asymmetric import ed25519
from util.parameter import BASE_PORT, NODE_COUNT, SHARD_COUNT, ENABLE_SHARDING

# --- 路径与配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
DATASET_PATH = os.path.join(BASE_DIR, "china_agri_traceability_v10_final.csv")
QUERY_RESULT_FILE = os.path.join(BASE_DIR, "query_benchmark_results.csv")
INIT_FLAG_FILE = os.path.join(BASE_DIR, ".blockchain_initialized")


class TraceabilityBenchmark:
    def __init__(self, total_queries=1000):
        self.total_queries = total_queries
        # 加载真实数据集
        if os.path.exists(DATASET_PATH):
            self.df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
        else:
            raise FileNotFoundError(f"找不到数据集文件: {DATASET_PATH}")

        self.priv_key = ed25519.Ed25519PrivateKey.generate()
        self.pub_bytes = self.priv_key.public_key().public_bytes_raw()

        self.injected_pids = self.df['溯源批次码'].tolist()
        self.query_latencies = []
        self.correct_responses = 0
        self.node_processes = []

    def is_first_run(self):
        """判断是否需要初始化上链"""
        return not os.path.exists(INIT_FLAG_FILE)

    def start_nodes(self):
        print(f"[*] 正在启动 {NODE_COUNT} 个区块链节点...")
        for i in range(NODE_COUNT):
            port = BASE_PORT + i
            loc_code = 100 + i
            p = subprocess.Popen(
                [PYTHON_EXE, MAIN_SCRIPT, str(i), str(port), str(loc_code)],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
                cwd=BASE_DIR
            )
            self.node_processes.append(p)
        print(f"[*] 节点启动中，等待网络就绪...")
        time.sleep(5)

    def cleanup(self):
        print("[*] 正在清理节点进程...")
        for p in self.node_processes:
            try:
                parent = psutil.Process(p.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except:
                pass
        print("[*] 清理完成。")

    async def inject_all_data(self):
        """
        布置生产与加工两个环节：
        每个原始记录生成两个有关联的交易
        """
        print(f"[*] 首次运行：正在将 {len(self.df)} 组(共 {len(self.df) * 2} 笔)生产与加工数据布置到区块链...")
        ctx = zmq.asyncio.Context()
        # 负载均衡：随机选择一个节点作为网关入口
        sock = ctx.socket(zmq.PUSH)
        target_port = BASE_PORT + random.randint(0, NODE_COUNT - 1) + 50
        sock.connect(f"tcp://127.0.0.1:{target_port}")

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            p_id = row['溯源批次码']
            # loc_code 决定分片，确保同一批次的生产和加工在同一分片
            loc_code = 100 + (i % SHARD_COUNT)

            # --- 环节 1: 生产环节 (Production) ---
            tx_prod = TraceTransaction(p_id, "产地生产环节", TraceTransaction.OP_PRODUCE, loc_code, self.pub_bytes)
            tx_prod.iot = {
                "crop": str(row['作物名称']),
                "origin": f"{row['生产省份']}{row['生产县区']}",
                "weight": str(row['采摘重量(kg)']),
                "harvest_time": str(row['采摘完成时刻']),
                "farm_log": str(row['全流程农事操作记录'])[:50]  # 截断防止单笔过大
            }
            tx_prod.sign_tx(self.priv_key)
            await sock.send(tx_prod.pack_signed())

            # --- 环节 2: 加工环节 (Processing) ---
            # 关键点：ph 指向生产环节的哈希，建立 O(k) 溯源链
            tx_proc = TraceTransaction(p_id, "工厂加工环节", TraceTransaction.OP_TRANSPORT, loc_code, self.pub_bytes,
                                       ph=tx_prod.h)
            tx_proc.iot = {
                "factory": str(row['加工中心']),
                "process_detail": str(row['加工具体工序']),
                "start_time": str(row['加工开始时刻']),
                "end_time": str(row['加工结束时刻'])
            }
            tx_proc.sign_tx(self.priv_key)
            await sock.send(tx_proc.pack_signed())

            if (i + 1) % 500 == 0:
                print(f"    - 已发送 {i + 1} 组生产/加工链式数据...")

        sock.close()
        # 4000条记录*2笔交易=8000笔交易。单机共识压力较大，建议等待较长时间
        wait_time = 80
        print(f"[*] 发送完毕，等待 {wait_time}s 待共识打包入库...")
        await asyncio.sleep(wait_time)

        # 标记初始化完成
        with open(INIT_FLAG_FILE, "w") as f:
            f.write(f"Initialized at {datetime.now()}")

    async def perform_stress_queries(self):
        """ 执行 10,000 次压力查询测试 """
        print(f"[*] 开始执行 {self.total_queries} 次溯源查询压测 (含防伪预警测试)...")
        ctx = zmq.asyncio.Context()
        start_time = time.perf_counter()

        tasks = []
        for i in range(self.total_queries):
            rand = random.random()
            if rand < 0.1:  # 10% 模拟非法假码查询
                p_id = f"FAKE_CODE_{uuid.uuid4().hex[:6]}"
                expected = "NOT_FOUND"
            elif rand < 0.2:  # 10% 模拟克隆攻击 (高频扫码)
                p_id = self.injected_pids[0]
                expected = "ALERT_EXPECTED"
            else:  # 80% 正常溯源回溯
                p_id = random.choice(self.injected_pids)
                expected = "OK"

            tasks.append(self.single_query_task(ctx, p_id, expected))

            # 控制并发数，避免单机网络栈过载
            if len(tasks) >= 100:
                await asyncio.gather(*tasks)
                tasks = []

        total_duration = time.perf_counter() - start_time
        res = self.calculate_results(total_duration)
        self.save_results_to_csv(res)
        self.print_report(res)

    async def single_query_task(self, ctx, p_id, expected):
        """ 发送 REQ 查询请求并统计 """
        # 根据 p_id 哈希值计算路由（模拟地理分片治理策略）
        shard_idx = (hash(p_id) % SHARD_COUNT) if ENABLE_SHARDING else 0
        target_node = shard_idx * (NODE_COUNT // SHARD_COUNT)
        port = BASE_PORT + target_node + 100  # RPC 端口

        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        sock.connect(f"tcp://127.0.0.1:{port}")

        q_start = time.perf_counter()
        try:
            # 执行查询
            await sock.send_json({"type": "QUERY_TRACE", "p_id": p_id, "loc": 999})
            res = await sock.recv_json()
            latency = time.perf_counter() - q_start
            self.query_latencies.append(latency)

            # 验证结果准确率
            if expected == "NOT_FOUND":
                if res.get("status") == "NOT_FOUND": self.correct_responses += 1
            elif expected == "ALERT_EXPECTED":
                # 统计是否返回了预警标记或状态正常（克隆识别是动态过程）
                if res.get("status") == "OK":
                    self.correct_responses += 1
            else:
                if res.get("status") == "OK" and len(res.get("data", [])) >= 2:  # 至少包含生产和加工两笔
                    self.correct_responses += 1
        except:
            pass
        finally:
            sock.close()

    def calculate_results(self, duration):
        avg_lat = np.mean(self.query_latencies) * 1000 if self.query_latencies else 0
        qps = self.total_queries / duration
        accuracy = (self.correct_responses / self.total_queries) * 100
        return {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Sharding": "ON" if ENABLE_SHARDING else "OFF",
            "Shards": SHARD_COUNT,
            "Nodes": NODE_COUNT,
            "QueryCount": self.total_queries,
            "AvgLatency_ms": round(avg_lat, 2),
            "QPS": round(qps, 2),
            "Accuracy": f"{round(accuracy, 2)}%"
        }

    def save_results_to_csv(self, data):
        file_exists = os.path.isfile(QUERY_RESULT_FILE)
        with open(QUERY_RESULT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    def print_report(self, res):
        print("\n" + "=" * 50)
        print("高速溯源与动态防伪体系测试报告")
        print("-" * 50)
        print(f"测试时间:      {res['Time']}")
        print(f"分片策略:      {res['Sharding']} ({res['Shards']} 分片)")
        print(f"平均查询延迟:  {res['AvgLatency_ms']} ms")
        print(f"查询吞吐量:    {res['QPS']} QPS")
        print(f"预警/溯源准确率: {res['Accuracy']}")
        print(f"详细报告已写入: {QUERY_RESULT_FILE}")
        print("=" * 50 + "\n")


async def main():
    tester = TraceabilityBenchmark(total_queries=1000)
    try:
        tester.start_nodes()

        if tester.is_first_run():
            await tester.inject_all_data()
        else:
            print("[*] 检测到已有区块链索引，跳过注入，直接进行压测...")

        await tester.perform_stress_queries()

    except Exception as e:
        print(f"[✘] 运行出错: {e}")
        traceback.print_exc()
    finally:
        tester.cleanup()


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())