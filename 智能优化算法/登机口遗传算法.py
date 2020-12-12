# -*- coding: utf-8 -*-
import numpy as np
import math
import random
import pandas as pd
import geatpy as ea
import geatpy as ea # import geatpy
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
# from MyProblem import MyProblem
import sys
sys.path.append('C:\\Users\\Leo-Li\\.PyCharmCE2018.3\\config\\scratches')
from my_ga import my_ga
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool


# 读取数据
df_plane = pd.read_csv('flight_303.csv')
df_port = pd.read_csv('InputData.csv')

# 端口对应约束
def is_flight_valid(flight, port):
    arrive = df_plane['arrive_type'][flight] in df_port['arrive_type'][port]
    depart = df_plane['depart_type'][flight] in df_port['depart_type'][port]
    return arrive and depart

def is_type_valid(plane, port):
    type1 = df_plane['type'][plane]
    type2 = df_port['type'][port]
    return type1 == type2

def flight_port():
    table = [[0 for i in range(69)] for j in range(303)]
    for flight in range(303):
        for port in range(69):
            boo1 = is_flight_valid(flight, port)
            boo2 = is_type_valid(flight, port)
            if not (boo1 and boo2):
                table[flight][port] = 1
    return table

# 时间冲突约束
def is_time_valid():
    table = [[0 for i in range(303)] for j in range(303)]
    for flight in range(303):
        flight_arrive_str = df_plane['arrive_date'][flight].strip()+df_plane['arrive_time'][flight].strip()
        flight_arrive = datetime.strptime(flight_arrive_str, '%d-%m-%y%H:%M')
        flight_depart_str = df_plane['depart_date'][flight].strip() + df_plane['depart_time'][flight].strip()
        flight_depart = datetime.strptime(flight_depart_str, '%d-%m-%y%H:%M') + timedelta(minutes=45)
        for item in range(303):
            if flight == item:
                continue
            item_arrive_str = df_plane['arrive_date'][item].strip()+df_plane['arrive_time'][item].strip()
            item_arrive = datetime.strptime(item_arrive_str, '%d-%m-%y%H:%M')
            item_depart_str = df_plane['depart_date'][item].strip()+df_plane['depart_time'][item].strip()
            item_depart = datetime.strptime(item_depart_str, '%d-%m-%y%H:%M') + timedelta(minutes=45)
            if (flight_arrive <= item_arrive and flight_depart > item_depart) or (flight_arrive >= item_arrive and
            flight_arrive <= item_depart) or (flight_arrive <= item_arrive and flight_depart<= item_depart and flight_depart >= item_arrive)or (flight_arrive >= item_arrive and flight_depart >= item_depart and item_depart >= flight_arrive):
                table[flight][item] = 1
    return table

# 登机口是否一致
def is_equal(port1, port2):
    arrive = len(set(df_port['arrive_type'][port1].split(',')).intersection(set(df_port['arrive_type'][port2].split(',')))) >= 1
    depart = len(set(df_port['depart_type'][port1].split(',')).intersection(set(df_port['depart_type'][port2].split(',')))) >= 1
    p_type = df_port['type'][port1] == df_port['type'][port2]
    return arrive and depart and p_type
# 约束矩阵
time_table = is_time_valid()
port_table = flight_port()

# 航班一致检验
def is_equal_flight(flight1, flight2):
    arrive = df_plane['arrive_type'][flight1]==df_plane['arrive_type'][flight2]
    depart = df_plane['depart_type'][flight1]==df_plane['depart_type'][flight2]
    p_type = df_plane['type'][flight1] == df_plane['type'][flight2]
    return arrive and depart and p_type

def get_x0():
    x0 = [-1 for index in range(303)]
    first = list(range(69))
    random.shuffle(first)
    for i in range(303):
        for j in first:
            if port_table[i][j] == 0:
                col_count = 0
                collid = [flight for flight,port in enumerate(x0) if port==j].copy()
                # print('collid:', collid, j)
                if len(collid) == 0:
                    x0[i] = j
                    # print(i, j, x0[i])
                    break
                for c_flight in collid:
                    if time_table[i][c_flight] == 1:
                        col_count += 1
                        break
                if col_count == 0:
                    x0[i] = j
                    # print(i, j, x0[i])
                    break
            # print(i, j, x0[i])
    return x0

def is_valid(x):
    for flight,port in enumerate(x):
        if port == -1:
            continue
        flights = [my_flight for my_flight,my_port in enumerate(x) if my_port==port]
        if not is_flight_valid(flight, port):
            return False
        if not is_type_valid(flight, port):
            return False
        for item in flights:
            if flight == item:
                continue
            if time_table[flight][item] == 1:
                return False
    return True

# 遗传算法解TSP问题
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 303  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [-1] * Dim  # 决策变量下界
        ub = [68] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen  # 得到决策变量矩阵

        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, x)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, x)
            result.wait()
            pop.ObjV = np.array(result.get())
        '''
        ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(x.shape[0]):
            count = 0
            for item in x[i]:
                if item != -1:
                    count += 1
            ObjV.append(count)
        pop.ObjV = np.array([ObjV]).T  # 把求得的目标函数值赋值给种群pop的ObjV
        # 找到违反约束条件的个体在种群中的索引，保存在向量exIdx中（如：若0、2、4号个体违反约束条件，则编程找出他们来）
        pop.CV = np.zeros((pop.sizes, 1))
        for index,item in enumerate(x):
            if not is_valid(item):
                pop.CV[index] = 1

        # exIdx1 = np.where(np.where(x == 3)[1] - np.where(x == 6)[1] < 0)[0]
        # exIdx2 = np.where(np.where(x == 4)[1] - np.where(x == 5)[1] < 0)[0]
        # exIdx = np.unique(np.hstack([exIdx1, exIdx2]))
        # pop.CV = np.zeros((pop.sizes, 1))
        # pop.CV[exIdx] = 1  # 把求得的违反约束程度矩阵赋值给种群pop的CV
        '''
def subAimFunc(x):
    count = 0
    for item in x:
        if item != -1:
            count += 1
    return [count]

if __name__ == '__main__':
    PoolType = 'Process'
    """================================实例化问题对象============================"""
    problem = MyProblem(PoolType) # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'        # 编码方式，采用排列编码
    NIND = 50             # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = my_ga(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 200 # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.5 # 变异概率
    myAlgorithm.drawing = 1 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """========================先验种群=========================================="""
    prophetChrom = []
    for item in range(NIND):
        prophetChrom.append(get_x0())
    prophetChrom = np.array(prophetChrom)
    prophetPop = ea.Population(Encoding, Field, NIND, prophetChrom)  # 实例化种群对象
    myAlgorithm.call_aimFunc(prophetPop)  # 计算先知种群的目标函数值及约束
    """===========================调用算法模板进行种群进化======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run(prophetPop) # 执行算法模板
    population.save() # 把最后一代种群的信息保存到文件中
    """===============================输出结果及绘图============================"""
    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
    best_ObjV = np.min(obj_trace[:, 1])
    print('最短路程为：%s'%(best_ObjV))
    print('最佳路线为：')
    # spots = pd.read_csv('spotsF.csv')
    best_journey = var_trace[best_gen, :]
    print(list(best_journey))
    # for i in best_journey:
        # print(spots.name[i])
    print()
    print('有效进化代数：%s'%(obj_trace.shape[0]))
    print('最优的一代是第 %s 代'%(best_gen + 1))
    print('评价次数：%s'%(myAlgorithm.evalsNum))
    print('时间已过 %s 秒'%(myAlgorithm.passTime))
    '''
    # 绘图
    plt.figure()
    plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], c = 'black')
    plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], 'o', c = 'black')
    for i in range(len(best_journey)):
        plt.text(problem.places[int(best_journey[i]), 0], problem.places[int(best_journey[i]), 1], chr(int(best_journey[i]) + 65), fontsize=20)
    plt.grid(True)
    plt.xlabel('x坐标')
    plt.ylabel('y坐标')
    plt.savefig('roadmap.svg', dpi=600, bbox_inches='tight')
    '''