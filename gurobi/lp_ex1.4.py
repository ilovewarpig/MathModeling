import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# 货机装载问题，目标函数和约束条件涉及大量Σ 司守奎 数学建模算法与应用 第2版 习题1.4
# 能不用addMVar就不用，很坑

a = np.array([18, 15, 23, 12])
b = np.array([480, 650, 580, 390])
c = np.array([3100, 3800, 3500, 2850])
w = np.array([10, 16, 8])
v = np.array([6800, 8700, 5300])

# 创建模型
m = gp.Model("ex1.4")
# 创建变量
x = np.array([m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='c%s'%item) for item in range(12)])
x = x.reshape((4, 3))
# 目标函数
m.setObjective(gp.quicksum(c[i]*x[i][j] for i in range(4) for j in range(3)), GRB.MAXIMIZE)

# 约束条件
m.addConstrs(gp.quicksum(x[i][j] for j in range(3)) <= a[i] for i in range(4))
m.addConstrs(gp.quicksum(x[i][j] for i in range(4)) <= w[j] for j in range(3))
m.addConstrs(gp.quicksum(b[i]*x[i][j] for i in range(4)) <= v[j] for j in range(3))
m.addConstr(gp.quicksum(x[i][0] for i in range(4))/10 == gp.quicksum(x[j][1] for j in range(4))/16)
m.addConstr(gp.quicksum(x[j][1] for j in range(4))/16 == gp.quicksum(x[k][2] for k in range(4))/8)

# 模型求解
m.optimize()

# 打印结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))
