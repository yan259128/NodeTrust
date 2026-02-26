import tps
TEST_DURATION = 30

total_tx, latencies = tps.get_performance_metrics()
avg_tps = total_tx / TEST_DURATION
avg_lat = sum(latencies) / len(latencies) if latencies else 0

print(avg_tps)
print(avg_lat)