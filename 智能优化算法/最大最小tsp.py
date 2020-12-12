import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import time


df = pd.read_csv('position.csv')
adj = np.array(pd.read_csv('adj.csv'))
car = {'C1':[50, 30], 'C2':[50, 30],'C3':[50, 30],'C4':[50, 30],'C5':[50, 30],'C6':[50, 30],'C7':[50, 30],'C8':[50, 30],
       'C9':[50, 30],'C10':[50, 30],'C11':[50, 30],'C12':[50, 30],'B1':[60, 35],'B2':[60, 35],'B3':[60, 35],'B4':[60, 35],
       'B5':[60, 35],'B6':[60, 35],'A1':[70, 45],'A2':[70, 45],'A3':[70, 45],'A4':[70, 45],'A5':[70, 45],'A6':[70, 45]}
start_point = [65,    64,    50,    38,    40,    39,    31,    32,    10 ,    9,     8,    33,    41,    42,    48,
               53,    54,    52,    49,    47,44, 51, 35, 34, 0]
main_street = {}
for item in range(1, 12):
    if item < 10:
        c2 = 'J0%g'%(item)
    else:
        c2 = 'J%g'%(item)
    if item-1 < 10:
        c1 = 'J0%g'%(item-1)
    else:
        c1 = 'J%g' % (item-1)
    main_street[(c1, c2)] = 0
    main_street[(c2, c1)] = 0

for item in range(12, 21):
    if item < 10:
        c2 = 'J0%g'%(item)
    else:
        c2 = 'J%g'%(item)
    if item-1 < 10:
        c1 = 'J0%g'%(item-1)
    else:
        c1 = 'J%g' % (item-1)
    main_street[(c1, c2)] = 0
    main_street[(c2, c1)] = 0

main_pair = list(main_street.keys())
# adj不用距离而是使用飞行时长（包含绕过障碍物、转弯等时间）
'''
for i in range(len(df)):
    for j in range(len(df)):
        adj[i][j] = round(np.sqrt((df['x'][i]-df['x'][j])**2 + (df['y'][i]-df['y'][j])**2 + (df['z'][i]-df['z'][j])**2),3)
'''
num = adj.shape[0] # 城市数

tau = np.array([[1 for i in range(num)] for j in range(num)], dtype=np.float64)  # ij路上的信息素
Q = 1
tau_delta = np.array([[0 for i in range(num)] for j in range(num)], dtype=np.float64)

def dist_2d(r, t):
    x1, y1 = df[['x', 'y']].iloc[r]
    x2, y2 = df[['x', 'y']].iloc[t]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def fitness(path, wait):
    # 修改为三段路径累加
    total = 0
    car_count = [0, 0]
    count = 0
    # 路段、车型
    for item in range(1, len(path)-2):
        if path[item] in start_point:
        # if df['code'][path[item]] == 'D1' or df['code'][path[item]] == 'D2':
            total += car_count[1]
            car_count[0] += 1
            car_count[1] = 0
            count += 1
            # 下一辆
            continue

        c1 = df['code'][path[item-1]]
        c2 = df['code'][path[item]]
        # 判断是否主路
        if (c1, c2) in main_pair:
            # 判断车型
            if car_count[0] < 12:
                dis = adj[path[item - 1]][path[item]] / car['C1'][0]
            elif car_count[0] < 18:
                dis = adj[path[item - 1]][path[item]] / car['B1'][0]
            elif car_count[0] < 24:
                dis = adj[path[item - 1]][path[item]] / car['A1'][0]
        else:
            # 判断车型
            if car_count[0] < 12:
                dis = adj[path[item - 1]][path[item]] / car['C1'][1]
            elif car_count[0] < 18:
                dis = adj[path[item - 1]][path[item]] / car['B1'][1]
            else:
                dis = adj[path[item - 1]][path[item]] / car['A1'][1]
        car_count[1] += dis
        # print('%g->%g'%(), dis)
    total += car_count[1]
    for item in wait:
        total += item
    print('共%g辆'%count)
    return total

class Ant:
    def __init__(self):
        # 初始化
        self.tabu = [68, 0, 1] # 禁忌表

        self.alpha = 1 # 信息素的重要程度
        self.beta = 2 # 启发因子
        self.fit = np.inf # 适应值
        self.unvisited = list(range(8, 68))  # 未访问火力节点
        self.unvisited = list(set(self.unvisited).difference(start_point))
        self.unvisited_Z = list(range(2,8))  # 未访问转载节点
        self.connected_visited = []  # 候选节点
        self.path = [start_point[0]] # 可行路径
        self.currentCity = 0 # 当前节点
        self.isError = False

        self.car_count = 0
        self.single_count = 1 # 判断单车任务全部完成
        self.work_table = [[[] for i in range(num) ] for j in range(num)]  # 各车经过时刻
        self.wait_time = [0 for i in range(24)]  # 车辆等待时间
        self.is_stop = False
        self.current_mission = 'Z'
        self.current_time = 0
        self.restart = False # 是否回到起点发第二辆车

    def findPath(self):
        # 寻路循环
        for current_index in range(0, adj.shape[0]*30):
            wait = []
            stuck = []
            r = self.path[current_index] # 未考虑最终节点
            # print('current: ',r)
            # 终点判断
            if self.is_stop:
                print('全部走完，寻路结束')
                break

            # 确定车型
            if self.car_count < 12:
                my_speed = car['C1'].copy()
            elif self.car_count < 18:
                my_speed = car['B1'].copy()
            else:
                my_speed = car['A1'].copy()
            # 确定目标为F的候选节点
            self.connected_visited = []
            if self.current_mission == 'F':
                scane = []
                r_distance = []
                dropped = []
                dropped_index = []
                for fire_spot in self.unvisited:
                    r_distance.append(dist_2d(r, fire_spot))
                    scane.append(fire_spot)
                nearest_F = scane[r_distance.index(min(r_distance))]
                nearest_dis = min(r_distance)
                stupid_point = []
                t = list(list(np.where(adj[r]<99999))[0])
                # print('t:', t, 'tabu:', self.tabu)
                for tbi in t:
                    # print('r:', r, 'tbi:', tbi)
                    if tbi in self.tabu:
                        stupid_point.append(tbi)
                        continue
                    if 'F' not in df['code'][tbi]:
                        temp_dis = dist_2d(tbi, nearest_F)
                        # print('tbi:', tbi, 'temp_dis:', temp_dis, 'nearest_dis:', nearest_dis, 'F:', nearest_F)
                        if temp_dis > nearest_dis:
                            # print('nearst_F:', nearest_F, 'remove %g->%g'%(r, tbi))
                            stupid_point.append(tbi)
                            dropped.append(temp_dis)
                            dropped_index.append(tbi)

                t = list(set(t).difference(set(stupid_point)))
                # print('_______________', t)
                for ti in t:
                    if ('F' in df['code'][ti]) and (ti in self.unvisited):
                        self.connected_visited.append(ti)
                if len(self.connected_visited) == 0:
                    for ti in t:
                        if ('Z' in df['code'][ti]) or ('J' in df['code'][ti]) and (ti not in self.tabu):
                            if len(self.work_table[r][ti]) > 0:
                                if (r, ti) in main_pair:
                                    vie_count = 0
                                    for scadul in self.work_table[r][ti]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    for scadul in self.work_table[ti][r]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    if vie_count >= 2:
                                        continue
                                else:
                                    vie_count = 0
                                    for scadul in self.work_table[r][ti]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    for scadul in self.work_table[ti][r]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    if vie_count >= 1:
                                        continue
                                self.connected_visited.append(ti)
                            elif ti not in self.tabu:
                                self.connected_visited.append(ti)
            # 确定目标为Z的候选节点
            elif self.current_mission == 'Z':
                scane = []
                r_distance = []
                dropped_index = []
                dropped = []
                stupid_point = []
                for z_spot in self.unvisited_Z:
                    r_distance.append(dist_2d(r, z_spot))
                    scane.append(z_spot)
                nearest_Z = scane[r_distance.index(min(r_distance))]
                nearest_dis = min(r_distance)

                t = list(list(np.where(adj[r] < 99999))[0])
                # print('Z删除前', t, 'tabu:', self.tabu)
                for tbi in t:
                    if (tbi in self.tabu) or ('F' in df['code'][tbi]):
                        stupid_point.append(tbi)
                        continue
                    if 'Z' not in df['code'][tbi]:
                        temp_dis = dist_2d(tbi, nearest_Z)
                        # print('tbi:', tbi, 'temp_dis:', temp_dis, 'nearest_dis:', nearest_dis, 'Z:', nearest_Z)
                        if temp_dis > nearest_dis:
                            # print('nearst_Z:', nearest_Z, 'remove %g->%g' % (r, tbi))
                            stupid_point.append(tbi)
                            dropped.append(temp_dis)
                            dropped_index.append(tbi)
                t = list(set(t).difference(set(stupid_point)))
                # print('Z删除后', t, 'tabu:', self.tabu)
                for ti in t:
                    if 'Z' in df['code'][ti]:
                        self.connected_visited.append(ti)
                if len(self.connected_visited) == 0:
                    for ti in t:
                        if ('J' in df['code'][ti]) and (ti not in self.tabu):
                            if len(self.work_table[r][ti]) > 0:
                                if (r, ti) in main_pair:
                                    vie_count = 0
                                    for scadul in self.work_table[r][ti]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            # print('stuck1', r, ti, t)
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    for scadul in self.work_table[ti][r]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            # print('stuck2', r, ti, t)
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    if vie_count >= 2:
                                        continue
                                else:
                                    vie_count = 0
                                    for scadul in self.work_table[r][ti]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            # print('stuck3', r, ti, t)
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    for scadul in self.work_table[ti][r]:
                                        if (scadul[0] < self.current_time) and (self.current_time < scadul[1]):
                                            vie_count += 1
                                            # print('stuck4', r, ti, t)
                                            stuck.append(ti)
                                            wait.append(scadul[1] - self.current_time)
                                    if vie_count >= 1:
                                        continue
                                self.connected_visited.append(ti)
                            elif ti not in self.tabu:
                                self.connected_visited.append(ti)
            # print(self.path)
            if len(self.connected_visited) == 0:
                if len(wait) > 0:
                    inde = wait.index(min(wait))
                    self.connected_visited.append(stuck[inde])
                    self.wait_time[self.car_count] += min(wait)
                else:
                    self.connected_visited.append(dropped_index[dropped.index(min(dropped))])
            # print(r, t, self.connected_visited)

            s = self.nextCity(r, my_speed)
            # print(self.path)
            # 更新
            self.update(r, s, my_speed)
            # print('水平误差：', self.level, '垂直误差：', self.vert)
            current_index += 1
        if not self.isError:
            # 计算适应值
            self.fit = fitness(self.path, self.wait_time)
            # 散布信息素，保存best fit
        else:
            self.reset()
            self.findPath()

    def nextCity(self, r, myspeed):
        # 状态转移
        if self.restart:
            '''
            if self.car_count in [6, 7, 8, 9, 10, 11, 15, 16, 17, 21, 22, 23]:
                return 1  # D2
            else:
                return 0  # D1
            '''
            return start_point[self.car_count]

        prob = np.array([0 for ci in range(len(self.connected_visited))], dtype=np.float64)

        # bottom = sum([math.pow(tau[r][pj], self.alpha)*math.pow(1/(adj[r][pj]/speed + df['delay'][pj]), self.beta) for pj in self.connected_visited])
        bottom = 0
        for pj in self.connected_visited:
            if (r, pj) in main_pair:
                speed = myspeed[0]
            else:
                speed = myspeed[1]
            bottom += math.pow(tau[r][pj], self.alpha)*math.pow(1/(adj[r][pj]/speed), self.beta)

        for index, pj in enumerate(self.connected_visited):
            if (r, pj) in main_pair:
                speed = myspeed[0]
            else:
                speed = myspeed[1]
            prob[index] = (math.pow(tau[r][pj], self.alpha) * math.pow(1 / (adj[r][pj] / speed),self.beta)) / bottom

        # for index, pj in enumerate(self.connected_visited):
            # prob[index] = (math.pow(tau[r][pj], self.alpha)*math.pow(1/(adj[r][pj]/speed + df['delay'][pj]), self.beta)) / bottom

        P = np.cumsum(prob)
        r = random.random()
        # print(self.connected_visited)
        # print('index', np.where(P >= r)[0][0], 'con', self.connected_visited[0], self.connected_visited, self.path)
        # print(self.path)
        next_city = self.connected_visited[np.where(P >= r)[0][0]]
        # print('ok', next_city)

        return next_city

    def update(self, r, s, myspeed):
        # 更新信息素
        # 更新path
        # 更新水平、垂直误差
        self.path.append(s)
        self.tabu.append(s)
        if self.restart:
            self.restart = False
            return
        if (r, s) in main_pair:
            speed = myspeed[0]
        else:
            speed = myspeed[1]
        arrive_time = self.current_time
        leave_time = self.current_time + adj[r][s] / speed
        if self.car_count == 24:
            self.is_stop = True
            return
        if 'F' in df['code'][s]:
            self.unvisited.remove(s)
        if ('J' in df['code'][s]) or ('Z' in df['code'][s]):
            self.work_table[r][s].append([arrive_time, leave_time, self.car_count]) # 可以改为字典 'A1':[a, b]
        self.current_time = leave_time
        '''
        if (self.single_count == 0) and (self.current_mission == 'F') and 'F' in df['code'][s]:
            self.single_count += 1
            self.tabu = [68]
            self.current_mission = 'Z'
        '''
        if (self.single_count == 1) and (self.current_mission == 'Z') and 'Z' in df['code'][s]:
            self.single_count += 1
            self.tabu = [68, 0, 1, s]
            self.current_mission = 'F'

        if (self.single_count == 2) and (self.current_mission == 'F') and 'F' in df['code'][s]:
            # 该车完成所有任务
            self.single_count = 1
            self.tabu = [68, 0, 1]
            self.current_mission = 'Z'
            self.current_time = 0
            self.car_count += 1
            self.restart = True

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
        self.unvisited = list(range(8, 68))  # 未访问火力节点
        self.unvisited = list(set(self.unvisited).difference(start_point))
        self.unvisited_Z = list(range(2, 8))  # 未访问转载节点
        self.connected_visited = []  # 候选节点
        self.path = [start_point[0]] # 可行路径
        self.currentCity = 0 # 当前节点
        self.isError = False

        self.car_count = 0
        self.single_count = 1 # 判断单车任务全部完成
        self.work_table = [[[] for i in range(num) ] for j in range(num)]  # 各车经过时刻
        self.wait_time = [0 for i in range(24)]  # 车辆等待时间
        self.is_stop = False
        self.current_mission = 'Z'
        self.current_time = 0
        self.restart = False # 是否回到起点发第二辆车
        self.tabu = [68, 0, 1]


class ACO:
    def __init__(self):
        # 生成蚂蚁
        self.initN = 30
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

                if ant.fit < self.bestFit:
                    self.bestFit = ant.fit
                    self.bestTour = ant.path
                    work_table = pd.DataFrame(ant.work_table)

                local_results.append(ant.fit)
                ant.reset()
            print('全局最优解: ', self.bestFit, self.bestTour)
            # print(self.tour1, fitness(self.tour1), self.tour2, fitness(self.tour2), self.tour3, fitness(self.tour3))
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
                for item in range(1, len(self.bestTour)):
                    tau_delta[self.bestTour[item - 1]][self.bestTour[item]] += (Q / self.bestFit)

            else:
                for item in range(1, len(self.local_path)):
                    tau_delta[self.local_path[item - 1]][self.local_path[item]] += (Q / self.local_best)

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
        work_table.to_csv('work_table.csv', index=False)



if __name__ == '__main__':
    start_time = time.time()
    aco = ACO()
    aco.optimize(iter_max=30)
    end_time = time.time()
    print('耗时：', end_time-start_time)



