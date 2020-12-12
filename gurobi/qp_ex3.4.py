import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# 非凸模型， 司守奎 数学建模算法与应用 第2版 习题3.4
# gurobi中会自动对模型进行检测，如果是非凸需要设置m.params.NonConvex=2  (默认-1)


# 创建模型
m = gp.Model("ex3.4")
# 创建变量
x = m.addVars(3, lb=-GRB.INFINITY, obj=1, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
# 目标函数
m.setObjective(2*x[0]+3*x[0]*x[0]+3*x[1]+x[1]*x[1]+x[2], GRB.MAXIMIZE)
# 约束条件
m.addConstr(x[0]+2*x[0]*x[0]+x[1]+2*x[1]*x[1]+x[2]<=10)
m.addConstr(x[0]+x[0]*x[0]+x[1]+x[1]*x[1]-x[2]<=50)
m.addConstr(2*x[0]+x[0]*x[0]+2*x[1]+x[2]<=40)
m.addConstr(x[0]*x[0]+x[2]==2)
m.addConstr(x[0]+2*x[1]>=1)
m.addConstr(x[0]>=0)
m.params.NonConvex=2

# 模型求解
m.optimize()

# 打印结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))
