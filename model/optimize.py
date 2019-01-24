import sys

def optimizeRelationships(relPred,relNodes,gtNodeNeighbors):
    #if 'cvxpy' not in sys.modules:
    import cvxpy
    penalty=490
    useRel = cvxpy.Variable(relPred.size(0),boolean=True)

    obj =0
    huh=0
    for i in range(relPred.size(0)):
        obj += relPred[i].item()*useRel[i]
        huh +=useRel[i]


    constraint = [0]*len(gtNodeNeighbors)
    for i in range(len(gtNodeNeighbors)):
        relI=0
        for a,b in relNodes:
            j=None
            if a==i:
                j=b
            elif b==i:
                j=a
            if j is not None:
                constraint[i] += useRel[relI]
            relI+=1
        constraint[i] -= gtNodeNeighbors[i]
        #obj -= cvxpy.power(penalty,(cvxpy.abs(constraint[i]))) #this causes it to not miss on the same node more than once
        constraint[i] = cvxpy.abs(constraint[i])
        obj -= penalty*constraint[i]


    cs=[]
    for i in range(len(gtNodeNeighbors)):
        cs.append(constraint[i]<=1)
    problem = cvxpy.Problem(cvxpy.Maximize(obj),cs)
    problem.solve(solver=cvxpy.GLPK_MI)
    return useRel.value
#from gurobipy import *
#
#
#def optimizeRelationshipsGUROBI(relPred,relNodes,gtNodeNeighbors):
#    m = Model("mip1")
#
#    x = m.addVar(vtype=GRB.BINARY, name="x")
#
#    useRel=[]
#    for i in range(relPred.size(0)):
#        useRel.append( m.addVar(vtype=GRB.BINARY, name="e{}".format(i)) )
#
#    obj = LinExpr()
#    for i in range(relPred.size(0)):
#        obj += relPred[i].item()*useRel[i]
#
#    for i in range(numNodes):
#        constraint = LinExpr()
#        relI=0
#        for a,b in relNodes:
#            j=None
#            if a==i:
#                j=b
#            elif b==i:
#                j=a
#            if j is not None:
#                constraint += useRel[relI]
#        constraint -= gtNodeNeighbors[i]
#        obj -= penalty**(abs(constraint)) #this causes it to not miss on the same node more than once
#
#    m.setObjective(obj, GRB.MAXIMIZE)
#
#    m.optimize()
#
#    #for v in m.getVars():
#    #    print(v.varName, v.x)
#    ret = [0]*relPred.size(0)
#    for i in range(relPred.size(0)):
#        ret[i]=useRel[i].x
