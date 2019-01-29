import sys

def optimizeRelationships(relPred,relNodes,gtNodeNeighbors,penalty=490):
    #if 'cvxpy' not in sys.modules:
    import cvxpy
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
def optimizeRelationshipsSoft(relPred,relNodes,predNodeNeighbors,penalty=1.2):
    #if 'cvxpy' not in sys.modules:
    import cvxpy
    useRel = cvxpy.Variable(relPred.size(0),boolean=True)

    obj =0
    huh=0
    for i in range(relPred.size(0)):
        obj += relPred[i].item()*useRel[i]
        huh +=useRel[i]


    difference = [0]*len(predNodeNeighbors)
    for i in range(len(predNodeNeighbors)):
        relI=0
        for a,b in relNodes:
            j=None
            if a==i:
                j=b
            elif b==i:
                j=a
            if j is not None:
                difference[i] += useRel[relI]
            relI+=1
        difference[i] -= predNodeNeighbors[i]
        #obj -= cvxpy.power(penalty,(cvxpy.abs(difference[i]))) #this causes it to not miss on the same node more than once
        difference[i] = cvxpy.abs(difference[i])
        obj -= penalty*difference[i]
        obj -= penalty*cvxpy.maximum(1,difference[i]) + penalty #double penalty if difference>1
        obj -= penalty*cvxpy.maximum(2,difference[i]) + 2*penalty #triple penalty if difference>2


    cs=[]
    #for i in range(len(predNodeNeighbors)):
    #    cs.append(difference[i]<=4)
    problem = cvxpy.Problem(cvxpy.Maximize(obj),cs)
    #problem.solve(solver=cvxpy.GLPK_MI)
    problem.solve(solver=cvxpy.ECOS_BB)
    return useRel.value

def optimizeRelationshipsBlind(relPred,relNodes,penalty=0.5):
    #if 'cvxpy' not in sys.modules:
    import cvxpy
    useRel = cvxpy.Variable(relPred.size(0),boolean=True)

    obj =0
    huh=0
    for i in range(relPred.size(0)):
        obj += relPred[i].item()*useRel[i]
        huh +=useRel[i]

    maxId=0
    for a,b in relNodes:
        maxId=max(maxId,a,b)
    numNodes=maxId+1

    constraint = [0]*numNodes
    for i in range(numNodes):
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
        #constraint[i] -= gtNodeNeighbors[i]
        #obj -= cvxpy.power(penalty,(cvxpy.abs(constraint[i]))) #this causes it to not miss on the same node more than once
        #constraint[i] = cvxpy.abs(constraint[i])
        
        obj -= penalty*(cvxpy.maximum(constraint[i],1)-1)


    cs=[]
    for i in range(numNodes):
        cs.append(constraint[i]<=2)
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
