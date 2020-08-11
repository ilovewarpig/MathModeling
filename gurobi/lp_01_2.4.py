import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# 运动员指派问题，目标函数0-1变量 司守奎 数学建模算法与应用 第2版 习题2.4第1小问
# gurobi不支持两个以上决策变量连乘，要么创建新的变量(y1=x1*x2, y2=x3*x4, y1*y2)要么转化为线性组合
# 非gurobi可以使用以下语法:sum([np.prod([a[i][j] for i in range(4)]) for j in range(10)])

c = np.array([9.25,9.6,9,9.1,9.25,9.7,9.8,9,9.25,9.4,9,9,9.1,9.1,9.4,9.1,9,9.8,9.2,9.1,9.5,9,9.25,9.5,8.9,8.9,8.9,9.1,9,
              9.2,9.1,9.3,9.8,9,9.7,9.25,9.2,9.3,9.7,9.5]).reshape((4, 10))

# 创建模型
m = gp.Model("ex2.4.1")
# 创建变量
x = np.array([m.addVar(vtype=GRB.BINARY, name='c%s'%item) for item in range(40)])
x = x.reshape((4, 10))
# 是否参加全能赛，用以将连乘的约束(Πx[i][i])转化为求和(4*y[i]<=Σx[i][j]<=3+y[i]&Σy[i]==4)
y = m.addVars(10, vtype=GRB.BINARY)
# 目标函数
m.setObjective(gp.quicksum([c[i][j]*x[i][j] for i in range(4) for j in range(10)]), GRB.MAXIMIZE)
# 约束条件
m.addConstrs(gp.quicksum([x[i][j] for j in range(10)]) == 6 for i in range(4))
m.addConstrs(gp.quicksum(x[i][j] for i in range(4)) >= 4*y[j] for j in range(10))
m.addConstrs(gp.quicksum(x[i][j] for i in range(4)) <= 3+y[j] for j in range(10))
m.addConstr(gp.quicksum(y[i] for i in range(10))==4)

# 模型求解
m.optimize()

# 打印结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))
