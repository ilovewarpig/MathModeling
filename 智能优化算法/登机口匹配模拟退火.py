import numpy as np
import math
import random
import matplotlib.pyplot as plt
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

# 获取一个新解
def get_new_one(x0):
    delta = []
    new_x = x0.copy()
    target = random.randint(0, 69)
    while len([flight for flight,port in enumerate(x0) if port==target]) == 0:
        target = random.randint(0, 69)
    # print(target)
    outside = [flight for flight,port in enumerate(x0) if port==-1]
    flights = [flight for flight,port in enumerate(x0) if port==target]
    point1 = random.randint(0, len(flights))
    point2 = random.randint(0, len(flights))
    if point1 == point2:
        drop_out = [random.choice(flights)]
    else:
        drop_out = flights[min(point1, point2):max(point1, point2)]
    for i in drop_out:
        flights.remove(i)
    for de in drop_out:
            outside.append(de)
            new_x[de] = -1
    random.shuffle(outside)
    for item in outside:
        col_count = 0
        if is_equal_flight(drop_out[0], item):
            for item2 in flights:
                if time_table[item][item2] == 1:
                    col_count += 1
                    break
            if col_count == 0:
                delta.append(item)
                flights.append(item)
    if len(delta) > 0:
        for item in delta:
            new_x[item] = target
    else:
        print('no change')
        for de in drop_out:
            new_x[de] = target
        # new_x[drop_out] = target
    # print(drop_out)
    # print(delta)
    return new_x

nvar = 303  # 自变量维度
T0 = 100  # 初始温度
max_iter = 100  # 外循环最大迭代次数
Lk = 300  # 内循环最大迭代次数
alpha = 0.95  # 温度衰退系数
# lb = [-3, 4.1]
# ub = [12.1, 5.8]

def obj_fun(x):
    # x1, x2 = x
    # y = 21.5 + x1*math.sin(4*math.pi*x1) + x2 * math.sin(20*math.pi*x2)
    # count = 0
    # for item in x:
        # if item != -1:
            # count += 1
    lamb = 0.02
    beta = 3
    sum = 0
    for item in x:
        if item != -1:
            sum += 1
    # z = math.exp(sum * lamb) * 10 ** beta

    return sum

def get_new_x(old_x, T):
    # 产生新解
    # 登机口间、登机口外交换
    new = old_x.copy()
    for count in range(20):
        new_x = get_new_one(new)
    # y = np.random.randn(len(old_x))
    # z = y / np.sqrt(np.sum(y**2))
    # new_x = old_x + z*T
    return new_x

def temperature_down(t):
    # 降温操作
    return alpha * t

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

def simulate_annealing(nvar=nvar, T0=T0, max_iter=max_iter, Lk=Lk):
    # 初始化
    T = T0
    r = 0
    current_iter = 0
    # 初始解
    # x0 = get_x0()
    x0 = [6.0, 48.0, 20.0, 47.0, 21.0, 19.0, 10.0, 40.0, 39.0, -1.0, -1.0, 11.0, -1.0, 18.0, 52.0, 8.0, 7.0, -1.0, -1.0, 17.0, 43.0, 0.0, 38.0, 10.0, 55.0, 21.0, 30.0, 24.0, 20.0, 52.0, 43.0, 18.0, 8.0, 19.0, 6.0, 48.0, -1.0, 7.0, 47.0, 40.0, 11.0, 10.0, -1.0, -1.0, 55.0, 34.0, 9.0, 37.0, 21.0, 39.0, 0.0, 60.0, 12.0, 45.0, 29.0, 1.0, 36.0, 18.0, 8.0, 43.0, 20.0, -1.0, 13.0, 59.0, 6.0, -1.0, 19.0, 48.0, -1.0, -1.0, 28.0, -1.0, 50.0, 10.0, -1.0, 4.0, 7.0, -1.0, 22.0, -1.0, 3.0, 40.0, 11.0, 63.0, -1.0, 62.0, 43.0, 60.0, -1.0, 18.0, 6.0, 0.0, 52.0, 30.0, 66.0, 34.0, 26.0, 20.0, 21.0, 24.0, 8.0, 7.0, 48.0, 47.0, 55.0, 40.0, -1.0, 11.0, -1.0, 43.0, 61.0, 10.0, 19.0, -1.0, 17.0, -1.0, 9.0, 27.0, 38.0, -1.0, 58.0, 5.0, -1.0, 37.0, 68.0, -1.0, -1.0, 12.0, 45.0, 2.0, 29.0, -1.0, 36.0, 0.0, 13.0, 28.0, 50.0, 34.0, 39.0, 48.0, 8.0, 54.0, 49.0, 20.0, 21.0, 11.0, 43.0, 59.0, 31.0, 19.0, 53.0, -1.0, -1.0, 15.0, 41.0, 46.0, 40.0, 35.0, 1.0, 33.0, 30.0, 44.0, 52.0, 10.0, 37.0, 12.0, 36.0, -1.0, -1.0, 47.0, 17.0, 9.0, 45.0, 20.0, 7.0, 29.0, 28.0, 11.0, 43.0, 19.0, 49.0, -1.0, 31.0, -1.0, 53.0, 52.0, 30.0, 15.0, 54.0, 6.0, 48.0, 21.0, 22.0, 47.0, 10.0, 11.0, 60.0, 40.0, -1.0, 4.0, 1.0, 59.0, 39.0, 3.0, 67.0, 25.0, 8.0, 63.0, 62.0, -1.0, 66.0, 26.0, 0.0, 61.0, 7.0, 27.0, -1.0, 38.0, 58.0, -1.0, 10.0, -1.0, -1.0, 18.0, 52.0, 17.0, 43.0, 51.0, -1.0, 55.0, 4.0, 30.0, -1.0, 34.0, 24.0, 9.0, 37.0, 12.0, 45.0, -1.0, 29.0, 36.0, -1.0, 13.0, 28.0, 50.0, -1.0, 49.0, 31.0, 53.0, 15.0, 68.0, 2.0, 41.0, 60.0, 24.0, 1.0, 59.0, -1.0, 18.0, 4.0, 3.0, 30.0, -1.0, 40.0, 46.0, 34.0, -1.0, 39.0, 52.0, 5.0, 13.0, 20.0, 64.0, 19.0, 35.0, 55.0, 9.0, 17.0, 33.0, 43.0, -1.0, -1.0, 37.0, 12.0, 36.0, 45.0, -1.0, 49.0, -1.0, -1.0, 29.0, 28.0, -1.0, -1.0, 50.0, 31.0, 44.0, 16.0, 32.0, 14.0, 42.0, 65.0]
    y0 = obj_fun(x0)
    best_fit = y0
    best_vals = x0
    gen = []
    best_score = []

    # 开始迭代
    while current_iter < max_iter:
        for count in range(Lk):
            x_new = get_new_x(x0, T)
            y_new = obj_fun(x_new)
            if y_new > best_fit:
                y0 = y_new
                x0 = x_new
            else:
                p = math.exp((y_new-y0)/T)
                if random.random() < p:
                    x0 = x_new
                    y0 = y_new
            if y0 > best_fit:
                best_fit = y0
                best_vals = x0
        print('当前代最优值:', y0)
        current_iter += 1
        T = temperature_down(T)
        gen.append(current_iter)
        best_score.append(best_fit)
    plt.plot(gen, best_score, '-r')
    plt.show()
    print('最优解为: ', best_fit, '各变量值为: ', best_vals)


if __name__ == '__main__':
    simulate_annealing()