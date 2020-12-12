import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import time


df = pd.read_csv('full.csv')
df['delay'] = df['delay'] / 1000
adj = np.zeros((len(df), len(df)))
# adj不用距离而是使用飞行时长（包含绕过障碍物、转弯等时间）
for i in range(len(df)):
    for j in range(len(df)):
        adj[i][j] = round(np.sqrt((df['x'][i]-df['x'][j])**2 + (df['y'][i]-df['y'][j])**2 + (df['z'][i]-df['z'][j])**2),3)

num = adj.shape[0] # 城市数

tau = np.array([[1 for i in range(num)] for j in range(num)], dtype=np.float64)  # ij路上的信息素
Q = 1
tau_delta = np.array([[0 for i in range(num)] for j in range(num)], dtype=np.float64)
speed = 100000

total_distant2 = 0
for item in range(1, len(df)):
    total_distant2 += np.sqrt((df['x'][0]-df['x'][i])**2 + (df['y'][0]-df['y'][i])**2 + (df['z'][0]-df['z'][i])**2)

def distand_d(path):
    distant2 = 0
    for item in range(1, len(path)):
        distant2 += np.sqrt((df['x'][0] - df['x'][i]) ** 2 + (df['y'][0] - df['y'][i]) ** 2 + (df['z'][0] - df['z'][i]) ** 2)
    return distant2/total_distant2

def fitness(path):
    # 修改为三段路径累加
    total = 0
    for item in range(1, len(path)):
        total += adj[path[item - 1]][path[item]] / speed
        total += df['delay'][item]
    return total

def total_fit(path1, path2, path3):
    if len(path1) < 2 or len(path2) < 2 or len(path3) < 3:
        return np.inf
    total = max(fitness(path1), fitness(path2), fitness(path3))
    return total

class Ant:
    def __init__(self):
        # 初始化
        self.tabu = [1] # 禁忌表
        self.alpha = 1 # 信息素的重要程度
        self.beta = 2.5 # 启发因子
        self.fit = np.inf # 适应值
        self.unvisited = list(range(1, num))  # 未访问节点
        self.connected_visited = []  # 候选节点
        self.path = [0] # 可行路径
        self.path1 = []
        self.path2 = []
        self.path3 = []
        self.currentCity = 0 # 当前节点
        self.single_max = 4
        self.drone_count = 0
        self.isError = False
        self.num = adj.shape[0]


    def findPath(self):
        # 寻路循环
        current_index = 0
        while current_index < self.num:
            r = self.path[current_index] # 未考虑最终节点
            self.connected_visited = self.unvisited.copy()
            # 终点判断
            # 增加无人机计数，当不足3辆时计算单机最大长度且仍有未探索区域，重置当前点为起始点。第三量时不计算约束条件
            if len(self.unvisited) == 0:
                # print('全部走完，寻路结束')
                self.path3 = self.path.copy()
                break
            # 状态转移概率
            if self.drone_count < 2:
                if len(self.path) > 1:
                    if distand_d(self.path) >= 1/3:
                    # if fitness(self.path) >= self.single_max:
                    # if len(self.path) >= adj.shape[0]/3:
                        if self.drone_count == 0:
                            self.path1 = self.path.copy()
                            self.num -= len(self.path1)
                            self.num += 1
                        elif self.drone_count == 1:
                            self.path2 = self.path.copy()
                            self.num -= len(self.path2)
                            self.num += 1
                        self.drone_count += 1
                        current_index = 0
                        self.path=[0]
                        r = self.path[current_index]

            s = self.nextCity(r)
            # print(self.path)
            # 更新
            self.update(r, s)
            # print('水平误差：', self.level, '垂直误差：', self.vert)
            current_index += 1
        if not self.isError:
            # 计算适应值
            self.fit = total_fit(self.path1, self.path2, self.path3)
            # 散布信息素，保存best fit
        else:
            self.reset()
            self.findPath()


    def nextCity(self, r):
        # 状态转移
        prob = np.array([0 for ci in range(len(self.connected_visited))], dtype=np.float64)

        bottom = sum([math.pow(tau[r][pj], self.alpha)*math.pow(1/(adj[r][pj]/speed + df['delay'][pj]), self.beta) for pj in self.connected_visited])
        for index, pj in enumerate(self.connected_visited):
            prob[index] = (math.pow(tau[r][pj], self.alpha)*math.pow(1/(adj[r][pj]/speed + df['delay'][pj]), self.beta)) / bottom

        P = np.cumsum(prob)
        r = random.random()
        # print('index', np.where(P >= r)[0][0], 'con', self.connected_visited[0], self.connected_visited, self.path)
        next_city = self.connected_visited[np.where(P >= r)[0][0]]
        # print('ok', next_city)

        return next_city

    def update(self, r, s):
        # 更新信息素
        # 更新path
        # 更新水平、垂直误差
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
        self.fit = np.inf # 适应值
        self.unvisited = list(range(1, num))  # 未访问节点
        self.connected_visited = []  # 候选节点
        self.path = [0] # 可行路径
        self.currentCity = 0 # 当前节点
        self.isError = False
        self.path1 = []
        self.path2 = []
        self.path3 = []
        self.num = adj.shape[0]



class ACO:
    def __init__(self):
        # 生成蚂蚁
        self.initN = 50
        self.bestTour = []
        self.bestFit = np.inf
        self.local_best = np.inf
        self.local_path = []
        self.rho = 0.3 # 信息素挥发因子
        self.tau_max = 1
        self.tau_min = 0.001
        self.pbest = 0.005
        self.change = 5
        self.stuck_count = 0
        self.tour1 = []
        self.tour2 = []
        self.tour3 = []
        self.local_tou1 = []
        self.local_tou2 = []
        self.local_tou3 = []

        # self.initAnt()

    def optimize(self, iter_max):
        # 优化函数
        global tau
        ant = Ant()
        Gen = []
        dist = []
        temp_fit = self.bestFit
        for myiter in range(iter_max):
            local_results = []
            print('第%g/%g代'%(myiter, iter_max))
            self.local_best = np.inf
            self.local_path = []
            avg_count = 0
            for k in range(self.initN):
                # print('第%g/%g只蚂蚁'%(k, self.initN))
                ant.findPath()
                if ant.fit < self.local_best:
                    self.local_best = ant.fit
                    self.local_path = ant.path
                    self.local_tou1 = ant.path1
                    self.local_tou2 = ant.path2
                    self.local_tou3 = ant.path3
                if ant.fit < self.bestFit:
                    self.bestFit = ant.fit
                    self.bestTour = ant.path
                    self.tour1 = ant.path1
                    self.tour2 = ant.path2
                    self.tour3 = ant.path3

                local_results.append(ant.fit)
                ant.reset()
            print('全局最优解: ', self.bestFit)
            print(self.tour1, fitness(self.tour1), self.tour2, fitness(self.tour2), self.tour3, fitness(self.tour3))
            Gen.append(myiter)
            dist.append(self.bestFit)
            if temp_fit == self.bestFit:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
                temp_fit = self.bestFit
            if self.stuck_count >= 5:
                tau += 0.5 * (self.tau_max-tau)
            if np.array(local_results).std()==0 and self.bestFit == self.local_best:
                break

            tau_delta = np.array([[0 for i in range(num)] for j in range(num)], dtype=np.float64)

            # 精英加成
            if (myiter % self.change == 0) and (myiter >= 10):
                print('精英')
                for item in range(1, len(self.tour1)):
                    # tau_delta[self.bestTour[item - 1]][self.bestTour[item]] += (Q / self.bestFit)
                    tau_delta[self.tour1[item - 1]][self.tour1[item]] += self.rho*self.tau_max
                for item in range(1, len(self.tour2)):
                    # tau_delta[self.bestTour[item - 1]][self.bestTour[item]] += (Q / self.bestFit)
                    tau_delta[self.tour2[item - 1]][self.tour2[item]] += self.rho*self.tau_max
                for item in range(1, len(self.tour3)):
                    # tau_delta[self.bestTour[item - 1]][self.bestTour[item]] += (Q / self.bestFit)
                    tau_delta[self.tour3[item - 1]][self.tour3[item]] += self.rho*self.tau_max
            else:
                for item in range(1, len(self.local_tou1)):
                    # tau_delta[self.local_path[item - 1]][self.local_path[item]] += (Q / self.local_best)
                    tau_delta[self.local_tou1[item - 1]][self.local_tou1[item]] += self.rho*self.tau_max
                for item in range(1, len(self.local_tou2)):
                    # tau_delta[self.local_path[item - 1]][self.local_path[item]] += (Q / self.local_best)
                    tau_delta[self.local_tou2[item - 1]][self.local_tou2[item]] += self.rho*self.tau_max
                for item in range(1, len(self.local_tou3)):
                    # tau_delta[self.local_path[item - 1]][self.local_path[item]] += (Q / self.local_best)
                    tau_delta[self.local_tou3[item - 1]][self.local_tou3[item]] += self.rho*self.tau_max
            # 更新信息素
            for item1 in range(adj.shape[0]):
                for item2 in range(adj.shape[0]):
                    tau[item1][item2] = (1-self.rho)*tau[item1][item2]
                    tau[item1][item2] += tau_delta[item1][item2]
                    if tau[item1][item2] > self.tau_max:
                        tau[item1][item2] = self.tau_max
                    elif tau[item1][item2] < self.tau_min:
                        tau[item1][item2] = self.tau_min
                    tau_delta[item1][item2] = 0



        plt.plot(Gen, dist, '-r')
        plt.show()


if __name__ == '__main__':
    start_time = time.time()
    aco = ACO()
    aco.optimize(iter_max=30)
    end_time = time.time()
    print('耗时：', end_time-start_time)



