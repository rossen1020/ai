import numpy as np

citys = [
    (0, 3), (0, 0),
    (0, 2), (0, 1),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 3),
    (3, 1), (3, 2)
]

def dis(p1, p2):
    x1, y1= p1
    x2, y2= p2
    return((x2 -x1)** 2 +(y2 - y1)** 2)** 0.5

def pL(p, citys):
    dist= 0
    plen= len(p)
    for i in range(plen):
        dist +=dis(citys[p[i]], citys[p[(i + 1)% plen ]])
    return dist

def Pro(citys):
    n= len(citys)
    dp= np.full((1<< n, n), float('inf'))
    dp[1][0]=  0
    parent= np.full((1 << n, n), -1)

    for m in range(1, 1<< n):
        for i in range(n):
            if m &(1<< i):
                for j in range(n):
                    if i !=j and m&(1 << j):
                        if dp[m ^(1<< i)][j] +dis(citys[j], citys[i])< dp[m][i]:
                            dp[m][i]= dp[m^ (1 << i)][j]+ dis(citys[j], citys[i])
                            parent[m][i]= j

    m= (1<< n)- 1
    i= np.argmin([dp[m][i]+ dis(citys[i], citys[0]) for i in range(n)])
    min_dist= dp[m][i]+ dis(citys[i], citys[0])

    p= []
    while i != -1:
        p.append(i)
        n= parent[m][i]
        m ^= (1<< i)
        i= n

    p.reverse()
    return min_dist, p

min_dis, p= Pro(citys)
print("最短距離:", min_dis)
print("最佳路徑:", p)
print("長度:", pL(p, citys))
