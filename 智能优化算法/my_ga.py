import geatpy as ea
import numpy as np
import math
import random
import pandas as pd
from datetime import timedelta
from datetime import datetime

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

class my_mut(ea.Mutation):
    def __init__(self):
        pass

    def do(self, Encoding, OldChrom, FieldDR, *args):
        NewChrom = OldChrom.copy()
        if Encoding != 'RI':
            raise RuntimeError('编码方式必须为RI')
        for item in NewChrom:
            if not is_valid(item):
                item = get_x0().copy()
        return NewChrom

class my_ga(ea.soea_SEGA_templet):
    def __init__(self, problem, population):
        ea.SoeaAlgorithm.__init__(self, problem, population)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'SEGA'
        self.selFunc = 'tour'  # 锦标赛选择算子
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=0.7)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=0.5)  # 生成逆转变异算子对象
        else:
            self.recOper = ea.Xovdp(XOVR=0.7)  # 生成两点交叉算子对象
            if population.Encoding == 'BG':
                self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
            elif population.Encoding == 'RI':
                self.mutOper = my_mut()  # 生成breeder GA变异算子对象
            else:
                raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')

    def run(self, prophetPop=None):
        # 初始化参数
        population = self.population
        NIND = population.sizes
        self.initialization()
        # 初始化染色体
        population.initChrom(NIND)
        self.call_aimFunc(population)
        # 先验知识
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)
        # 开始进化
        while self.terminated(population) == False:
            # 选择
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            self.call_aimFunc(offspring)  # 计算目标函数值
            population = population + offspring  # 父子合并
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
            # 得到新一代种群
            population = population[ea.selecting('dup', population.FitnV, NIND)]  # 采用基于适应度排序的直接复制选择生成新一代种群

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果