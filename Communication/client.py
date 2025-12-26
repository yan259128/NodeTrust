import zmq
import zmq.asyncio


class ZMQClient:
    """ 网络客户端：异步接收订阅广播与主动拉取数据 """

    def __init__(self):
        self.ctx = zmq.asyncio.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.peers = {}

    def add_peer(self, ip, port):
        """ 加入已知对等节点 """
        addr = f"tcp://{ip}:{port}"
        if addr not in self.peers:
            self.sub.connect(addr)
            self.peers[addr] = port

    async def fetch_peer(self, ip, port, req_data):
        """ 主动向指定节点请求数据 (REQ) """
        sock = self.ctx.socket(zmq.REQ)
        sock.connect(f"tcp://{ip}:{port + 100}")
        await sock.send_json(req_data)
        if await sock.poll(3000):  # 3秒超时
            res = await sock.recv_json()
            sock.close();
            return res
        sock.close();
        return None

    async def listen(self, callback):
        """ 后台持续监听入站广播主题 """
        while True:
            topic, payload = await self.sub.recv_multipart()
            await callback(topic.decode(), payload)
