import zmq
import zmq.asyncio
import asyncio

class ZMQClient:
    def __init__(self):
        self.ctx = zmq.asyncio.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.peers = set()

    def add_peer(self, ip, port):
        """ 主动连接对等节点 """
        addr = f"tcp://{ip}:{port}"
        if addr not in self.peers:
            self.sub.connect(addr)
            self.peers.add(addr)

    async def fetch_peer(self, ip, port, req_data):
        """ 主动请求数据 (REQ) """
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(f"tcp://{ip}:{port + 100}")
        try:
            await sock.send_json(req_data)
            res = await sock.recv_json()
            return res
        except:
            return None
        finally:
            sock.close()

    async def listen(self, callback):
        """ 持续监听广播 """
        while True:
            try:
                # 接收 [topic, payload]
                msg = await self.sub.recv_multipart()
                if len(msg) >= 2:
                    topic = msg[0].decode()
                    payload = msg[1]
                    # 回调给 service.handle_incoming
                    await callback(topic, payload)
            except Exception as e:
                await asyncio.sleep(0.1)