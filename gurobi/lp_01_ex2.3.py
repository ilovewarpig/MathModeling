import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# 设备分配问题，目标函数0-1变量 司守奎 数学建模算法与应用 第2版 习题2.3

c = np.array([4, 2, 3, 4, 6, 4, 5, 5, 7, 6, 7, 6, 7, 8, 8, 6, 7, 9, 8, 6, 7, 10, 8, 6]).reshape((6, 4))

# 创建模型
m = gp.Model("ex2.3")
# 创建变量
x = np.array([m.addVar(vtype=GRB.BINARY, name='c%s'%item) for item in range(24)])
x = x.reshape((6, 4))
# 目标函数
m.setObjective(gp.quicksum([c[i][j]*x[i][j] for i in range(6) for j in range(4)]), GRB.MAXIMIZE)

# 约束条件
m.addConstrs(gp.quicksum([x[i][j] for j in range(4)]) == 1 for i in range(6))
m.addConstrs(gp.quicksum([x[i][j] for i in range(6)]) >= 1 for j in range(4))

# 模型求解
m.optimize()

# 打印结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))
