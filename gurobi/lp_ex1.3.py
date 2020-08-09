import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# 工厂生产问题 司守奎 数学建模算法与应用 第2版 习题1.3
# 创建模型
m = gp.Model("ex1.3")

# 创建变量
x = m.addVars(9, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER)

# 设置目标函数
m.setObjective((1.25-0.25)*(x[0]+x[1]) + (2-0.35)*x[7] +(2.8-0.5)*x[8] - 300/6000 *(5*x[0]+10*x[5]) - 321/10000*(7*x[3]
        + 9*x[6] + 12*x[8]) - 250/4000*(6*x[2] + 8*x[7]) - 783/7000*(4*x[3]+11*x[8]-200/4000*7*x[4]), GRB.MAXIMIZE)
# 设置约束
m.addConstr(5*x[0]+10*x[5] <= 6000)
m.addConstr(7*x[1]+9*x[6]+12*x[8] <= 10000)
m.addConstr(6*x[2] + 8*x[7] <= 7000)
m.addConstr(7*x[4]<=4000)
m.addConstr(x[0]+x[1] == x[2]+x[3]+x[4])
m.addConstr(x[5]+x[6]==x[7])

# 模型求解
m.optimize()

# 输出结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))

