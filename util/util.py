# 此文件用于定义辅助函数

import socket
import requests
import re


def get_ip(is_public: bool):
    # 获取公网或私网的ip
    if is_public:
        res = requests.get("https://myip.ipip.net", timeout=5).text
        return re.findall(r'(\d+\.\d+\.\d+\.\d+)', res)[0]
    else:
        return socket.gethostbyname(socket.gethostname())



def round_add(round_time):
    # 进行次数的自增
    return round_time + 1

