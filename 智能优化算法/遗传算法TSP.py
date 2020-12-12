# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geatpy as ea
import geatpy as ea # import geatpy
import sys
sys.path.append('C:\\Users\\Leo-Li\\.PyCharmCE2018.3\\config\\scratches')
from my_ga import my_ga
import matplotlib.pyplot as plt
# from MyProblem import MyProblem

# 遗传算法解TSP问题
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 15  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1] * Dim  # 决策变量下界
        ub = [15] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 新增一个属性存储旅行地坐标
        self.places = np.array([[0.4, 0.4439],
                                [0.2439, 0.1463],
                                [0.1707, 0.2293],
                                [0.2293, 0.761],
                                [0.5171, 0.9414],
                                [0.8732, 0.6536],
                                [0.6878, 0.5219],
                                [0.8488, 0.3609],
                                [0.6683, 0.2536],
                                [0.6195, 0.2634]])
        df = pd.read_csv('mix_time.csv')
        df = df[:16]
        del df['q']
        self.cost = np.array(df)

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen  # 得到决策变量矩阵
        # 添加从0地出发且最后回到出发地
        X = np.hstack([np.zeros((x.shape[0], 1)), x, np.zeros((x.shape[0], 1))]).astype(int)

        ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(X.shape[0]):
            distance = 0
            for current in range(len(X[i])):
                if current == len(X[i])-1:
                    break
                next = current + 1
                if current == len(X[i])-2:
                    next = 0
                # 根据邻接矩阵计算距离和
                distance += self.cost[X[i][current]][X[i][next]]
            # journey = self.places[X[i], :]  # 按既定顺序到达的地点坐标
            # distance = np.sum(np.sqrt(np.sum(np.diff(journey.T) ** 2, 0)))  # 计算总路程

            ObjV.append(distance)
        pop.ObjV = np.array([ObjV]).T  # 把求得的目标函数值赋值给种群pop的ObjV
        # 找到违反约束条件的个体在种群中的索引，保存在向量exIdx中（如：若0、2、4号个体违反约束条件，则编程找出他们来）
        # exIdx1 = np.where(np.where(x == 3)[1] - np.where(x == 6)[1] < 0)[0]
        # exIdx2 = np.where(np.where(x == 4)[1] - np.where(x == 5)[1] < 0)[0]
        # exIdx = np.unique(np.hstack([exIdx1, exIdx2]))
        pop.CV = np.zeros((pop.sizes, 1))
        # pop.CV[exIdx] = 1  # 把求得的违反约束程度矩阵赋值给种群pop的CV
'''
class my_ga(ea.soea_SEGA_templet):
    def __init__(self, problem, population):
        ea.soea_SEGA_templet.__init__(self, problem, population)
'''

if __name__ == '__main__':
    """================================实例化问题对象============================"""
    problem = MyProblem() # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'P'        # 编码方式，采用排列编码
    NIND = 50             # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 200 # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.5 # 变异概率
    myAlgorithm.drawing = 1 # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
    population.save() # 把最后一代种群的信息保存到文件中
    """===============================输出结果及绘图============================"""
    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
    best_ObjV = np.min(obj_trace[:, 1])
    print('最短路程为：%s'%(best_ObjV))
    print('最佳路线为：')
    spots = pd.read_csv('spotsF.csv')
    best_journey = np.hstack([0, var_trace[best_gen, :], 0])
    print(list(best_journey))
    for i in best_journey:
        print(spots.name[i])
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