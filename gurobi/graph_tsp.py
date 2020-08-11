import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# TSP问题 司守奎 数学建模算法与应用 第2版 例子4.7.2

# 邻接矩阵
w = np.array([0, 8, 5, 9, 12, 14, 12, 16, 17, 22,
              8, 0, 9, 15, 17, 8, 11, 18, 14, 22,
              5, 9, 0, 7, 9, 11, 7, 12, 12, 17,
              9, 15, 7, 0, 3, 17, 10, 7, 15, 18,
              12, 17, 9, 3, 0, 8, 10, 6, 15, 15,
              14, 8, 11, 17, 8, 0, 9, 14, 8, 16,
              12, 11, 7, 10, 10, 9, 0, 8, 6, 11,
              16, 18, 12, 7, 6, 14, 8, 0, 11, 11,
              17, 14, 12, 15, 15, 8, 6, 11, 0, 10,
              22, 22, 17, 18, 15, 16, 11, 11, 10, 0]).reshape((10, 10))

# 创建模型
m = gp.Model("ex4.7.2")

# 创建变量
x = np.array([m.addVar(vtype=GRB.BINARY, name='c%s'%item) for item in range(100)])
x = x.reshape((10, 10))
# 顺序变量
u = m.addVars(10, vtype=gp.GRB.INTEGER)

# 设置目标函数
m.setObjective(gp.quicksum([w[i][j]*x[i][j] for i in range(10) for j in range(10) if i != j]), GRB.MINIMIZE)

# 设置约束
# 遍历每个点
m.addConstr(gp.quicksum([x[i][j] for i in range(10) for j in range(10)]) == 10)
# 不允许两点间折返
m.addConstrs(x[i][j]+x[j][i] <=1 for i in range(10) for j in range(10))
print('')
# 出度约束，每点出度不大于1
m.addConstrs(gp.quicksum(x[i][j] for j in range(10)) <=1 for i in range(10))
# 入度约束，每点入度不大于1
m.addConstrs(gp.quicksum(x[i][j] for i in range(10)) <=1 for j in range(10))
# 防止生成子回路
m.addConstrs(u[i] - u[j] + 10*x[i][j] <= 10-1 for i in range(1, 10) for j in range(1, 10) if i != j)
print('')

# 模型求解
m.optimize()

# 输出结果
print('Obj: %g' % m.objVal)
for v in m.getVars():
    print('%s %g'%(v.varName, v.x))

print('最短路径:  %g' % m.objVal)
for i in range(10):
    print('节点%d顺序：'%i, m.getVarByName('C10%d'%i).x)

