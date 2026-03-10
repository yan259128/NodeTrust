# 此文件用于定义常用的参数
Port = 3300

NODE_COUNT = 12  # 在这里统一修改节点总数
BASE_PORT = 3300

# 是否开启地理分片机制
# True:  根据位置编码划分不同的分片，节点并行处理。
# False: 所有节点处于同一个全局分片，共同竞争/协作处理所有交易（传统区块链模式）。
ENABLE_SHARDING = True

# 当 ENABLE_SHARDING 为 True 时，交易和节点将根据 location_code 分配到 0 到 SHARD_COUNT-1 个分片中
SHARD_COUNT = 3
# 每个节点开启的并行查询进程数
NUM_WORKERS = 3