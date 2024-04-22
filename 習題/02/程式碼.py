import numpy as np

# 定義城市座標
citys = [
    (0, 3), (0, 0),
    (0, 2), (0, 1),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 3),
    (3, 1), (3, 2)
]

# 計算城市之間的距離
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# 計算路徑的長度
def pathLength(p, citys):
    dist = 0
    plen = len(p)
    for i in range(plen):
        dist += distance(citys[p[i]], citys[p[(i + 1) % plen]])
    return dist

# 定義動態規劃算法函數
def dynamicProgramming(citys):
    n = len(citys)
    dp = np.full((1 << n, n), float('inf'))
    dp[1][0] = 0

    for mask in range(1, 1 << n):
        for i in range(n):
            if mask & (1 << i):
                for j in range(n):
                    if i != j and mask & (1 << j):
                        dp[mask][i] = min(dp[mask][i], dp[mask ^ (1 << i)][j] + distance(citys[j], citys[i]))

    min_dist = min([dp[(1 << n) - 1][i] + distance(citys[i], citys[0]) for i in range(1, n)])
    return min_dist

# 執行動態規劃算法
min_distance = dynamicProgramming(citys)

# 輸出結果
print("Minimized Distance:", min_distance)
