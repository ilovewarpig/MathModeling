import numpy as np
import networkx as nx


G = nx.Graph()


def dijkstra(G, s, t):
    # G为邻接矩阵
    num = G.shape[0]
    Parents = [[False for i in range(num)] for j in range(num)]
    final = [False for i in range(num)]
    distance = [G[s][j] for j in range(num)]
    path = []
    for v, item in enumerate(distance):
        if item < np.inf:
            Parents[v][s] = True
            Parents[v][v] = True
    distance[s] = 0
    final[s] = True

    # 主循环
    for i in range(1, num):
        min = np.inf
        # scan
        for j in range(num):
            if not final[j]:
                if distance[j] < min:
                    v = j
                    min = distance[j]
        # add
        final[v] = True
        # update
        for w in range(num):
            if (not final[w]) & ((min + G[v][w]) < distance[w]):
                distance[w] = min + G[v][w]
                Parents[w] = Parents[v].copy()
                Parents[w][w] = True

    print('最短路径：', distance[t])
    for index, value in enumerate(Parents[t]):
        if value:
            path.append(index)
    print(path)

    return