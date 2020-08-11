import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# S到D的最短路径问题

w = np.ones((7, 7))*999
w[0][1]=2
w[0][2]=4
w[1][3]=3
w[1][4]=3
w[1][5]=1
w[2][3]=2
w[2][4]=3
w[2][5]=1
w[3][6]=1
w[4][6]=3
w[5][6]=4
for i in range(7):
    for j in range(7):
        if i==j:
            w[i][j]=0

# 创建模型
m = gp.Model("exa4.2")
# 创建变量
x = np.array([m.addVar(vtype=gp.GRB.BINARY, name='c%s'%item) for item in range(49)])
x = x.reshape((7, 7))
# 目标函数
m.setObjective(gp.quicksum(w[i][j]*x[i][j] for i in range(7) for j in range(7) if w[i][j] < 10), gp.GRB.MINIMIZE)
# 约束条件
m.addConstr(gp.quicksum(x[0][j] for j in range(7) if w[0][j] < 10) - gp.quicksum(x[i][0] for i in range(7) if w[i][0] < 10) == 1)
m.addConstr(gp.quicksum(x[6][j] for j in range(7) if w[6][j] < 10) - gp.quicksum(x[i][6] for i in range(7) if w[i][6] < 10) == -1)
for i in range(1, 6):
    m.addConstr(gp.quicksum(x[i][j] for j in range(7) if w[i][j]<10) - gp.quicksum(x[j][i] for j in range(7) if w[j][i] < 10) == 0)

# 模型求解
m.optimize()

# 打印结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))
