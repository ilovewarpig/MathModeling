import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# 图论工具箱，测试
a = np.array([0, 50, np.inf, 40, 25, 10, 50, 0, 15, 20, np.inf, 25, np.inf, 15, 0, 10, 20, np.inf, 40, 20, 10, 0, 10, 25, 25, np.inf, 20, 10, 0, 55, 10, 25, np.inf, 25, 55, 0]).reshape((6, 6))
G = nx.from_numpy_matrix(a, create_using=nx.Graph)
# [nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, source=0, target=i, weight='weight') for i in range(1, 6)]
# 输出g1到其他节点的最短路径和cost
for item in nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, source=0, weight='weight'):
    print(item)

# 求最小生成树
G = nx.Graph()
e = [('1', '2', 50), ('1', '3', 60), ('2', '4', 65), ('2', '5', 40), ('3', '4', 52), ('3', '7', 45), ('4', '5', 46), ('4', '6', 30), ('4', '7', 42), ('5', '6', 70)]
G.add_weighted_edges_from(e)
T = nx.minimum_spanning_tree(G, weight='weight')
print(sorted(T.edges(data=True)))

# 课后习题 4.3 仓库-市场最大流问题
G = nx.DiGraph()
G.add_edge('s', 'a', capacity=20)
G.add_edge('s', 'b', capacity=20)
G.add_edge('s', 'c', capacity=100)
G.add_edge('a', '1', capacity=30)
G.add_edge('a', '2', capacity=10)
G.add_edge('a', '4', capacity=40)
G.add_edge('b', '3', capacity=10)
G.add_edge('b', '4', capacity=50)
G.add_edge('c', '1', capacity=20)
G.add_edge('c', '2', capacity=10)
G.add_edge('c', '3', capacity=40)
G.add_edge('c', '4', capacity=5)
G.add_edge('1', 't', capacity=20)
G.add_edge('2', 't', capacity=20)
G.add_edge('3', 't', capacity=60)
G.add_edge('4', 't', capacity=20)
flow_value, flow_dict = nx.maximum_flow(G, 's', 't')
print(flow_value)
print(flow_dict)

# 最小费用最大流 习题4.5
G = nx.DiGraph()
G.add_edges_from([('s', 'A', {'capacity': 8, 'weight': 0}),
                  ('s', 'B', {'capacity': 7, 'weight': 0}),
                  ('A', '1', {'capacity': 8, 'weight': 20}),
                  ('A', '2', {'capacity': 8, 'weight': 24}),
                  ('A', '3', {'capacity': 8, 'weight': 5}),
                  ('B', '1', {'capacity': 7, 'weight': 30}),
                  ('B', '2', {'capacity': 7, 'weight': 22}),
                  ('B', '3', {'capacity': 7, 'weight': 20}),
                  ('1', 't', {'capacity': 4, 'weight': 0}),
                  ('2', 't', {'capacity': 5, 'weight': 0}),
                  ('3', 't', {'capacity': 6, 'weight': 0}),
                  ])
mincostFlow = nx.max_flow_min_cost(G, 's', 't')
print(mincostFlow)
mincost = nx.cost_of_flow(G, mincostFlow)
print('最小费用:', mincost)
mincostFlowValue = (sum((mincostFlow[u]['t'] for u in G.predecessors('t'))) - sum((mincostFlow['t'][v] for v in G.successors('t'))))
print('最大流：', mincostFlowValue)
