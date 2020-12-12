import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import time


df = pd.read_csv('pr264.txt', sep=' ')
adj = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        adj[i][j] = round(np.sqrt((df['x'][i]-df['x'][j])**2 + (df['y'][i]-df['y'][j])**2),3)

num = adj.shape[0] # 城市数
tau = [[1 for i in range(num)] for j in range(num)]  # ij路上的信息素
Q = 1
tau_delta = [[0 for i in range(num)] for j in range(num)]

def fitness(path):
    total = 0
    for item in range(1, len(path)):
        total += adj[path[item - 1]][path[item]]
    return total

class Ant:
    def __init__(self):
        # 初始化
        self.alpha = 1 # 信息素的重要程度
        self.beta = 2 # 启发因子
        self.fit = 9999999 # 适应值
        self.unvisited = list(range(1, num)) # 未访问节点
        self.connected_visited = [] # 候选节点
        self.path = [0] # 可行路径
        self.currentCity = 0 # 当前节点
        self.isError = False


    def findPath(self):
        # 寻路循环
        for current_index in range(0, num-1):
            r = self.path[current_index] # 未考虑最终节点
            self.connected_visited = self.unvisited.copy()
            # 终点判断
            if len(self.unvisited) == 0:
                print('全部走完，寻路结束')
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
            for item in range(1, len(self.path)):
                tau_delta[self.path[item - 1]][self.path[item]] += Q / self.fit
                tau_delta[self.path[item]][self.path[item] - 1] += Q / self.fit
        else:
            self.reset()
            self.findPath()


    def nextCity(self, r):
        # 状态转移
        prob = np.array([0 for ci in range(len(self.connected_visited))], dtype=np.float64)

        bottom = sum([math.pow(tau[r][pj], self.alpha)*math.pow(1/adj[r][pj], self.beta) for pj in self.connected_visited])
        for index, pj in enumerate(self.connected_visited):
            prob[index] = (math.pow(tau[r][pj], self.alpha)*math.pow(1/adj[r][pj], self.beta)) / bottom

        # adj[r][pj]

        # big_shoot = np.where(prob == np.max(prob))[0][0]
        P = np.cumsum(prob)
        r = random.random()
        # print('index', np.where(P >= r)[0][0], 'con', self.connected_visited[0], self.connected_visited, self.path)
        next_city = self.connected_visited[np.where(P >= r)[0][0]]
        # print('ok', next_city)

        return next_city

    def update(self, r, s):
        self.path.append(s)
        self.unvisited.remove(s)

    def reset(self):
        # 重置蚂蚁(清空禁忌表和误差)
        self.fit = 9999999 # 适应值
        self.unvisited = list(range(1, num)) # 未访问节点
        self.connected_visited = [] # 候选节点
        self.path = [0] # 可行路径
        self.currentCity = 0 # 当前节点
        self.isError = False



class ACO:
    def __init__(self):
        # 生成蚂蚁
        self.initN = 50
        self.bestTour = []
        self.bestFit = 99999999
        self.rho = 0.3 # 信息素挥发因子
        self.local_best = []

        # self.init_ant()

    def init_ant(self):
        initAnt = Ant()
        tau_delta = [[0 for i in range(num)] for j in range(num)]
        for i in range(self.initN):
            initTour = [j for j in range(num)]
            random.shuffle(initTour)
            initAnt.path = initTour.copy()
            initAnt.fit = fitness(initAnt.path)
            for item in range(1, len(initAnt.path)):
                tau_delta[initAnt.path[item - 1]][initAnt.path[item]] += Q / initAnt.fit
                tau_delta[initAnt.path[item]][initAnt.path[item]-1] += Q / initAnt.fit
            for item1 in range(adj.shape[0]):
                for item2 in range(adj.shape[0]):
                    tau[item1][item2] = (1-self.rho)*tau[item1][item2]
                    tau[item1][item2] += tau_delta[item1][item2]
                    tau_delta[item1][item2] = 0
            tau_delta = [[0 for i in range(num)] for j in range(num)]
            if initAnt.fit < self.bestFit:
                self.bestFit = initAnt.fit
                self.bestTour = initAnt.path
            initAnt.reset()
            print('init fit', self.bestFit)


    def optimize(self, n):
        # 优化函数
        ant = Ant()
        Gen = []
        dist = []
        for i in range(n):
            print('第%g/%g代'%(i, n))
            for k in range(self.initN):
                # print('第%g/%g只蚂蚁'%(k, self.initN))
                ant.findPath()
                self.local_best.append(ant.fit)
                if ant.fit < self.bestFit:
                    self.bestFit = ant.fit
                    self.bestTour = ant.path
                # print('当前可行解：', ant.path, '路程长度', ant.fit)
                ant.reset()

            print('全局最优解: ', self.bestFit, 'tour: ',self.bestTour)
            Gen.append(i)
            dist.append(self.bestFit)
            for item1 in range(adj.shape[0]):
                for item2 in range(adj.shape[0]):
                    tau[item1][item2] = (1-self.rho)*tau[item1][item2]
                    tau[item1][item2] += tau_delta[item1][item2]
                    tau_delta[item1][item2] = 0


        plt.plot(Gen, dist, '-r')
        plt.show()


if __name__ == '__main__':
    start_time = time.time()
    aco = ACO()
    aco.optimize(50)
    end_time = time.time()
    print('耗时：', end_time-start_time)



