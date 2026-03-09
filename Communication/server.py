import zmq
import zmq.asyncio
import asyncio
import random
from util.parameter import NODE_COUNT


class ZMQServer:
    """ 网络服务端：异步消息广播与 RPC 响应请求 """
    def __init__(self, port):
        self.ctx = zmq.asyncio.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{port}")
        # REP 用于节点握手与历史块同步
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind(f"tcp://*:{port + 100}")

    async def broadcast(self, topic, payload):
        """ 向全网发布消息 """
        low = 0.001 * NODE_COUNT
        await asyncio.sleep(random.uniform(low, 0.2))
        await self.pub.send_multipart([topic.encode(), payload])

    async def start_rep_handler(self, callback):
        """ 持续监听并响应点对点同步请求 """
        while True:
            msg = await self.rep.recv_json()
            res = await callback(msg)
            await self.rep.send_json(res)