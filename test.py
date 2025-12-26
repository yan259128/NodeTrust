t = int(input())

for _ in range(t):
    a, b, c = map(int, input().split())

    # 检查所有可能的组合
    if a + b >= 10 or a + c >= 10 or b + c >= 10:
        print("YES")
    else:
        print("NO")