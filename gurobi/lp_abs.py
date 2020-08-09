import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


'''
目标函数带绝对值：|x1| + 2|x2| + 3|x3| + 4|x4| + 5|x5|
gurobi不支持直接在目标函数和约束条件中写带不等式的式子，因此需要建立同等数量的变量将目标转换为一般线性规划，
并在约束条件中限制a = |x|

如果目标函数带系数可以创建矩阵变量x = addMVars(4) 对系数列表 a = [1, 2, 3, 4]作矩阵计算 a@x
但比较坑的是gurobi中系数列表必须为numpy.array， 并且约束条件a[0] == x[0]是没办法写进addConstr()的，
不支持矩阵决策变量。
'''

# 创建模型
m = gp.Model("ex1.2")

# 创建变量
x = m.addVars(4, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
a = m.addVars(4, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

# 设置目标函数
m.setObjective(a[0] + 2*a[1] + 3*a[2] + 4*a[3], GRB.MINIMIZE)

# 设置约束
m.addConstr(x[0] - x[1] - x[2] + x[3]== 0, 'c0')
m.addConstr(x[0] - x[1] + x[2] - 3*x[3] == 1, 'c1')
m.addConstr(x[0] - x[1] - 2*x[2] + 3*x[3] == -0.5, 'c2')

# 绝对值约束
m.addConstrs(((a[i] == gp.abs_(x[i])) for i in range(4)))

# 模型求解
m.optimize()

# 输出结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))

