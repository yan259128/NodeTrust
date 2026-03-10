# server.py

import zmq
import zmq.asyncio
import asyncio
import random
from util.parameter import NODE_COUNT


class ZMQServer:
    """ 网络服务端：异步消息广播与 RPC 响应请求 """

    def __init__(self, port, inproc_addr=None, is_worker=False):
        self.ctx = zmq.asyncio.Context()

        # 广播 Socket 只在主进程或非工作模式下启动
        if not is_worker:
            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.bind(f"tcp://*:{port}")

        # RPC 响应 Socket
        self.rep = self.ctx.socket(zmq.REP)
        if is_worker:
            # 工作进程连接到 DEALER
            self.rep.connect(inproc_addr)
        else:
            # 传统模式或 Broker 的前端（虽然代理会处理，但保留逻辑）
            self.rep.bind(f"tcp://*:{port + 100}")

    async def broadcast(self, topic, payload):
        """ 向全网发布消息 """
        if hasattr(self, 'pub'):  # 确保只有主进程能广播
            low = 0.001 * NODE_COUNT
            await asyncio.sleep(random.uniform(low, 0.01))
            await self.pub.send_multipart([topic.encode(), payload])

    async def start_rep_handler(self, callback):
        """ 持续监听并响应请求 """
        while True:
            try:
                msg = await self.rep.recv_json()
                res = await callback(msg)
                await self.rep.send_json(res)
            except Exception as e:
                # 在高并发下，可能会有状态错误，忽略并继续
                await asyncio.sleep(0.001)