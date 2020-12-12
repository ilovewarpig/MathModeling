import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import time


df = pd.read_csv('data1.csv')
'''
adj = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        adj[i][j] = round(np.sqrt((df['x'][i]-df['x'][j])**2 + (df['y'][i]-df['y'][j])**2 + (df['z'][i]-df['z'][j])**2),3)
'''
adj = np.array(pd.read_csv('adj_data.csv'))
a1 = 25
a2 = 15
b1 = 20
b2 = 25
theta = 30
delta = 0.001

num = adj.shape[0] # 城市数
tau = [[1 for i in range(613)] for j in range(613)]  # ij路上的信息素
Q = 100
tau_delta = [[0 for i in range(613)] for j in range(613)]

def fitness(path):
    total = 0
    for item in range(1, len(path)):
        total += adj[path[item - 1]][path[item]]
    return total

class Ant:
    def __init__(self):
        # 初始化
        self.tabu = [1] # 禁忌表
        self.alpha = 1 # 信息素的重要程度
        self.beta = 2 # 启发因子
        self.fit = 9999999 # 适应值
        self.level = 0 # 动态水平误差
        self.vert = 0 # 动态垂直误差
        self.unvisited = list(range(1, 613)) # 未访问节点
        self.connected_visited = [] # 候选节点
        self.connected_visited_index = [] # 候选节点索引
        self.path = [0] # 可行路径
        self.currentCity = 0 # 当前节点
        self.isError = False
        self.q0 = 0.1 # 随机扰动系数 10%


    def findPath(self):
        # 寻路循环
        for current_index in range(0, num-1):
            r = self.path[current_index] # 未考虑最终节点
            self.connected_visited = []
            wrong_direction = []
            indexC = 1 # 暂无用处
            # 终点判断
            if delta*adj[r][612] + self.level <= theta and delta*adj[r][612] + self.vert <= theta:
                self.path.append(612)
                print('节点%g到节点612有可达路径，寻路结束' % r)
                break
            # 终点判断 获取所有r为始点的可行点
            for index in range(len(self.unvisited)):
                if df['type'][self.unvisited[index]]=='1':
                    if delta*adj[r][self.unvisited[index]] + self.level <= a2 and delta*adj[r][self.unvisited[index]] + self.vert <= a1:
                        self.connected_visited.append(self.unvisited[index])
                        self.connected_visited_index.append(index)
                        indexC += 1
                        # print('节点%g-->节点%g满足垂直误差纠正约束'%(r, self.unvisited[index]))
                        continue
                elif df['type'][self.unvisited[index]]=='0':
                    if delta*adj[r][self.unvisited[index]] + self.level <= b2 and delta*adj[r][self.unvisited[index]] + self.vert <= b1:
                        self.connected_visited.append(self.unvisited[index])
                        self.connected_visited_index.append(index)
                        # print('节点%g-->节点%g满足水平误差纠正约束' % (r, self.unvisited[index]))
                        indexC += 1
            # 终点导向约束：不走回头路
            for item in self.connected_visited:
                # 当前点、下一节点、终点
                point_r = np.array(df[['x', 'y', 'z']].iloc[r])
                point_s = np.array(df[['x', 'y', 'z']].iloc[item])
                point_t = np.array(df[['x', 'y', 'z']].iloc[612])
                sr = point_r - point_s
                st = point_t - point_s
                # 计算向量夹角，若夹角小于90度则无人机朝向起点移动
                cosin_angle = np.dot(sr, st) / (np.linalg.norm(sr) * np.linalg.norm(st))
                degree = np.degrees(np.arccos(cosin_angle))
                if degree < 90:
                    wrong_direction.append(item)
            self.connected_visited = [item for item in self.connected_visited if item not in wrong_direction].copy()

            if len(self.connected_visited) == 0:
                self.isError = 1
                break
            # 状态转移概率
            s = self.nextCity(r)
            # print(self.path)
            # 更新
            self.update(r, s)
            # print('水平误差：', self.level, '垂直误差：', self.vert)
        if not self.isError:
            # 计算适应值
            self.fit = fitness(self.path)
            # 散布信息素，保存best fit
            # for item in range(1, len(self.path)):
                # tau_delta[self.path[item - 1]][self.path[item]] += Q / self.fit
        else:
            self.reset()
            self.findPath()


    def nextCity(self, r):
        # 状态转移
        prob = np.array([0 for ci in range(len(self.connected_visited))], dtype=np.float64)
        if r == 0:
            bottom = sum(
                [math.pow(tau[r][pj], self.alpha) * math.pow(adj[0][pj] / 25000, self.beta)
                 for pj in self.connected_visited])
            for index2, pj in enumerate(self.connected_visited):
                prob[index2] = (math.pow(tau[r][pj], self.alpha) * math.pow(adj[0][pj] / 25000,self.beta)) / bottom
            P = np.cumsum(prob)
            rnd = random.random()
            next_city = self.connected_visited[np.where(P >= rnd)[0][0]]
            return next_city

        o = np.array(df[['x', 'y', 'z']].iloc[0], dtype=np.float64)
        t = np.array(df[['x', 'y', 'z']].iloc[612], dtype=np.float64)
        r2 = np.array(df[['x', 'y', 'z']].iloc[r], dtype=np.float64)
        o_t = t - o
        o_r = r2 - o
        cosin_angle1 = np.dot(o_t, o_r) / (np.linalg.norm(o_t) * np.linalg.norm(o_r))
        # print('r: ', r, 'o_t: ', o_t, 'o_r: ', o_r, 'cos1: ', cosin_angle1)
        dis = []
        for item in self.connected_visited:
            s = np.array(df[['x', 'y', 'z']].iloc[item])
            o_s = s - o
            # print('s: ', s, 'o_s: ', o_s)
            cosin_angle2 = np.dot(o_t, o_s) / (np.linalg.norm(o_t) * np.linalg.norm(o_s))
            # print('cos2: ', cosin_angle2)
            # print('item: ', item, 'delta: ', cosin_angle2*adj[0][item] - cosin_angle1*adj[0][r])
            dis.append(cosin_angle2*adj[0][item] - cosin_angle1*adj[0][r])

        bottom = sum([math.pow(tau[r][self.connected_visited[index]], self.alpha)*math.pow(delt/(25000*adj[r][self.connected_visited[index]]), self.beta)
                      for index, delt in enumerate(dis)])
        # print('bottom: ', bottom)
        # print('dis: ', dis)

        for index2, pj in enumerate(dis):
            prob[index2] = (math.pow(tau[r][self.connected_visited[index2]], self.alpha)*math.pow(pj/(25000*adj[r][self.connected_visited[index2]]), self.beta)) / bottom
        # print('prob:')
        # print(prob)
        P = np.cumsum(prob)
        rnd = random.random()
        next_city = self.connected_visited[np.where(P >= rnd)[0][0]]

        return next_city

    def update(self, r, s):
        # 更新信息素
        # 更新path
        # 更新水平、垂直误差
        self.level += delta*adj[r][s]
        self.vert += delta*adj[r][s]
        if df['type'][s] == '1':
            self.vert = 0
        elif df['type'][s] == '0':
            self.level = 0
        self.path.append(s)
        self.unvisited.remove(s)
        '''
        # 删除淘汰节点, 加速收敛
        for item in self.connected_visited:
            if item == s:
                continue
            self.unvisited.remove(item)
        '''

    def reset(self):
        # 重置蚂蚁(清空禁忌表和误差)
        self.tabu = [1] # 禁忌表
        self.fit = 9999999 # 适应值
        self.level = 0 # 动态水平误差
        self.vert = 0 # 动态垂直误差
        self.unvisited = list(range(1, 613)) # 未访问节点
        self.connected_visited = [] # 候选节点
        self.connected_visited_index = [] # 候选节点索引
        self.path = [0] # 可行路径
        self.currentCity = 0 # 当前节点
        self.isError = False



class ACO:
    def __init__(self):
        # 生成蚂蚁
        self.initN = 40
        self.bestTour = []
        self.bestFit = 99999999
        self.local_best = []
        self.local_path = []
        self.QA = 15 # 取前20蚂蚁
        self.rho = 0.3 # 信息素挥发因子
        # self.initAnt()

    def optimize(self, n):
        # 优化函数
        ant = Ant()
        Gen = []
        dist = []
        for i in range(n):
            print('第%g/%g代'%(i, n))
            for k in range(self.initN):
                print('第%g/%g只蚂蚁'%(k, self.initN))
                ant.findPath()
                self.local_best.append(ant.fit)
                self.local_path.append(ant.path)
                if ant.fit < self.bestFit:
                    self.bestFit = ant.fit
                    self.bestTour = ant.path
                if ant.isError:
                    print('死胡同')
                else:
                    print('当前可行解：', ant.path, '路程长度', ant.fit)
                ant.reset()

            print('全局最优解: ', self.bestTour, ': ', self.bestFit, )
            Gen.append(i)
            dist.append(self.bestFit)
            tau_delta = np.array([[0 for i in range(613)] for j in range(613)], dtype=np.float64)
            # 排序
            best20 = np.argsort(self.local_best)[:self.QA]
            # print(best20)
            for index, item in enumerate(best20):
                path = self.local_path[item]
                # print('best20: ', best20, 'item', item, 'path', print(path), 'local_best', self.local_best, self.local_path)
                # print(path)
                for item2 in range(1, len(path)):
                    # print(path[item2-1], path[item2])
                    tau_delta[path[item2 - 1]][path[item2]] += (self.QA-index)*(Q / self.local_best[item])
                    # print('done')
            # 精英加成
            for item in range(1, len(self.bestTour)):
                tau_delta[self.bestTour[item - 1]][self.bestTour[item]] += self.QA*(Q / self.bestFit)


            for item1 in range(adj.shape[0]):
                for item2 in range(adj.shape[0]):
                    tau[item1][item2] = (1-self.rho)*tau[item1][item2]
                    tau[item1][item2] += tau_delta[item1][item2]
                    tau_delta[item1][item2] = 0
            self.local_path = []
            self.local_best = []
            # print('tau')
            # print(tau)


        plt.plot(Gen, dist, '-r')
        plt.show()


if __name__ == '__main__':
    start_time = time.time()
    aco = ACO()
    aco.optimize(30)
    end_time = time.time()
    print('耗时：', end_time-start_time)



