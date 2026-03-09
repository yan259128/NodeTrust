import subprocess
import time
import os
import sys
import psutil
import csv
import traceback  # 用于打印详细错误堆栈
from datetime import datetime
import tps
from util.parameter import BASE_PORT,NODE_COUNT

# 获取当前脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(BASE_DIR, "benchmark_results.csv")

PYTHON_EXE = sys.executable
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
STARTUP_WAIT = 5  # 稍微加长等待时间，确保节点完全启动


def start_all_nodes():
    processes = []
    print(f"[*] 正在从目录启动节点: {BASE_DIR}")
    for i in range(NODE_COUNT):
        port = BASE_PORT + i
        loc_code = 100 + i
        # 确保在 Windows 下弹出窗口，方便观察节点是否真的动了
        p = subprocess.Popen(
            [PYTHON_EXE, MAIN_SCRIPT, str(i), str(port), str(loc_code)],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
            cwd=BASE_DIR  # 强制指定工作目录
        )
        processes.append(p)
    return processes


def save_results_to_csv(data):
    """将结果写入 CSV 文件"""
    print(f"[*] 准备写入文件: {RESULT_FILE}")
    file_exists = os.path.isfile(RESULT_FILE)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Time": timestamp,
        "Mode": data['mode'],
        "NodeSum":NODE_COUNT,
        "TPS": round(data['avg_tps'], 2),
        "Latency(s)": round(data['avg_lat'], 4),
        "CPU(%)": round(data['avg_cpu'], 1),
        "TotalTX": data['total_tx'],
        "Duration(s)": data['duration'],
        "Threads": data['threads']
    }

    try:
        with open(RESULT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"[✔] 写入成功！当前文件大小: {os.path.getsize(RESULT_FILE)} bytes")
    except Exception as e:
        print(f"[✘] 写入文件失败: {e}")


def main():
    # 1. 清理旧日志
    for f in os.listdir(BASE_DIR):
        if f.startswith("node_") and f.endswith(".log"):
            try:
                os.remove(os.path.join(BASE_DIR, f))
            except:
                pass

    procs = []
    try:
        # 2. 启动
        procs = start_all_nodes()
        time.sleep(STARTUP_WAIT)

        # 3. 运行测试
        print("[*] 开始运行 tps.run_test()...")
        results = tps.run_test()

        # 4. 保存
        if results:
            save_results_to_csv(results)
        else:
            print("[!] 警告: tps.run_test() 返回了空结果")

    except Exception:
        print("[!] 脚本执行过程中崩溃了:")
        traceback.print_exc()  # 打印具体的报错位置
    finally:
        # 5. 关闭节点
        print("[*] 正在清理进程...")
        for p in procs:
            try:
                parent = psutil.Process(p.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except:
                pass
        print("[*] 测试流程结束。")


if __name__ == "__main__":
    main()