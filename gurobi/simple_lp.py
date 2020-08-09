import gurobipy as gp
from gurobipy import GRB

try:

    # Create a new model
    m = gp.Model("ex1.1")

    # Create variables
    x = m.addVars(3, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)

    # Set objective
    m.setObjective(3 * x[0] - x[1] - x[2], GRB.MAXIMIZE)

    # Add constraint
    m.addConstr(x[0] - 2 * x[1] + x[2] <= 11, 'c0')
    m.addConstr(-4 * x[0] + x[1] + 2 * x[2] >= 3, 'c1')
    m.addConstr(-2 * x[0] + x[2] == 1, 'c2')

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')