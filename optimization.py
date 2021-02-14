# optimization.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import itertools

#import pacmanPlot
#import graphicsUtils
import util
import math
from copy import copy

# You may add any helper functions you would like here:
# def somethingUseful():
#     return True

def findIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b)
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.
    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).
        If none of the constraint boundaries intersect with each other, return [].

    An intersection point is an N-dimensional point that satisfies the
    strict equality of N of the input constraints.
    This method must return the intersection points for all possible
    combinations of N constraints.

    """
    N = len(constraints) #number of constraints
    A = [] #all A values from constraints
    b = [] #all b values from constraints
    points = [] #list of points of intersections
    for i in range(N):
        A.append(constraints[i][0])
        b.append(constraints[i][1])
    d = np.linalg.matrix_rank(A) #gets rank so linalg.solve works
    if N < d:
        return [] #no solution if N < d
    elif N == d: #unique solution if N == d
        return (np.linalg.solve(A, b))
    else: #otherwise solves for all combinations of constraints
        temp = itertools.combinations(constraints, d) #gets all combinations
        combos = [*temp] 
        for matrix in combos: 
            tempA = [] #temporary A values for solving
            tempB = [] #temporary b values for solving
            for eq in matrix:
                tempA.append(eq[0])
                tempB.append(eq[1])
            if (np.linalg.det(tempA) != 0): #solves if deterent is not 0
                points.append(tuple(np.linalg.solve(tempA, tempB))) #adds point
    return points

def findFeasibleIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    feasible intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).

        If none of the lines intersect with each other, return [].
        If none of the intersections are feasible, return [].

    You will want to take advantage of your findIntersections function.

    """
    feasible = [] #holds all feasible intersections
    points = findIntersections(constraints) #gets all intersections
    for point in points:
        good = True
        for constraint in constraints: #if point follows all constraints
            if np.dot(constraint[0], point) > constraint[1] + pow(10, -12):
                good = False
        if good:
            feasible.append(point) #adds feasible point
    return feasible

def solveLP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    find a feasible point that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your findFeasibleIntersections function.

    """
    minPoint = tuple() #best feasible point
    minAns = float('inf') #best feasible objective
    feasible = findFeasibleIntersections(constraints) #gets all feasible points
    if len(feasible) > 0: #if there are feasible points
        for point in feasible:
            ans = np.dot(point, cost) #finds total cost from point
            if ans < minAns: #keeps track of best feasible objective
                minAns = ans
                minPoint = point
        return (minPoint, minAns)
    else:
        return None

def wordProblemLP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveLP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
            ((sunscreen_amount, tantrum_amount), maximal_utility)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    #simply sets up cost tuple and list of constraints and returns answer from
    #soolveLP
    cost = (-7.0, -4.0)
    constraints = [((-1.0, 0.0), -20), ((0.0, -1.0), -15.5), ((2.5, 2.5), 100), ((.5, .25), 50)]
    minAns = solveLP(constraints, cost)
    return (minAns[0], -minAns[1]) #makes objective positive from being maximum

def solveIP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    use the branch and bound algorithm to find a feasible point with
    interger values that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your solveLP function.
    """
    ans = solveLP(constraints, cost) #gets solution from solveLP
    if ans == None: #returns none if no feasible objective
        return ans
    good = True
    for i in range(len(ans[0])):
        val = ans[0][i] #goes through all points in ans
        if (val - math.floor(val) <= pow(10, -12) or math.ceil(val) - val <= pow(10, -12)):
            pass
        else: #if point is not an integer
            good = False
            temp1 = [0] * len(ans[0])
            temp2 = [0] * len(ans[0])
            temp1[i] = 1
            temp2[i] = -1
            lower = (tuple(temp1), math.floor(val)) #sets lower constraint
            upper = (tuple(temp2), -math.ceil(val)) #sets upper constraint
            break
    if good:
        return ans #returns ans if all points are integers
    else:
        lowerAns = solveIP(constraints+[lower], cost) #branches with lower constraint
        upperAns = solveIP(constraints+[upper], cost) #branches with upper constraint
        if lowerAns == None and upperAns == None:
            return None #returns None if no feasible integer points are found
        elif lowerAns != None and upperAns != None:
            if lowerAns[1] < upperAns[1]:
                return lowerAns #returns lower if it is a better objective
            else:
                return upperAns #returns upper if it is a better objective
        elif lowerAns != None:
            return lowerAns #returns lower if no upper solution is found
        elif upperAns != None:
            return upperAns #returns upper if no lower solution is found
        else:
            return None #otherwise returns None
    
            
def wordProblemIP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveIP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
        ((f_DtoG, f_DtoS, f_EtoG, f_EtoS, f_UtoG, f_UtoS), minimal_cost)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    #simply sets up cost tuple and constraint list, I also tried using
    #foodDistribution to get cost and constraints and it worked the same
    cost = (12, 20, 4, 5, 2, 1)
    constraints = []
    constraints += [((1.2, 0, 0, 0, 0, 0), 30)]
    constraints += [((0, 1.2, 0, 0, 0, 0), 30)]
    constraints += [((0, 0, 1.3, 0, 0, 0), 30)]
    constraints += [((0, 0, 0, 1.3, 0, 0), 30)]
    constraints += [((0, 0, 0, 0, 1.1, 0), 30)]
    constraints += [((0, 0, 0, 0, 0, 1.1), 30)]
    constraints += [((-1, 0, -1, 0, -1, 0), -15)]
    constraints += [((0, -1, 0, -1, 0, -1), -30)]
    constraints += [((-1, 0, 0, 0, 0, 0), 0)]
    constraints += [((0, -1, 0, 0, 0, 0), 0)]
    constraints += [((0, 0, -1, 0, 0, 0), 0)]
    constraints += [((0, 0, 0, -1, 0, 0), 0)]
    constraints += [((0, 0, 0, 0, -1, 0), 0)]
    constraints += [((0, 0, 0, 0, 0, -1), 0)]
    '''
    truck_limit = 30
    W = (1.2, 1.3, 1.1)
    C = (15, 30)
    T = [(12, 20), (4, 5), (2, 1)]
    return foodDistribution(truck_limit, W, C, T)
    '''
    return solveIP(constraints, cost)

def foodDistribution(truck_limit, W, C, T):
    """
    Given M food providers and N communities, return the integer
    number of units that each provider should send to each community
    to satisfy the constraints and minimize transportation cost.

    Input:
        truck_limit: Scalar value representing the weight limit for each truck
        W: A tuple of M values representing the weight of food per unit for each 
            provider, (w1, w2, ..., wM)
        C: A tuple of N values representing the minimal amount of food units each
            community needs, (c1, c2, ..., cN)
        T: A list of M tuples, where each tuple has N values, representing the 
            transportation cost to move each unit of food from provider m to
            community n:
            [ (t1,1, t1,2, ..., t1,n, ..., t1N),
              (t2,1, t2,2, ..., t2,n, ..., t2N),
              ...
              (tm,1, tm,2, ..., tm,n, ..., tmN),
              ...
              (tM,1, tM,2, ..., tM,n, ..., tMN) ]

    Output: A length-2 tuple of the optimal food amounts and the corresponding objective
            value at that point: (optimial_food, minimal_cost)
            The optimal food amounts should be a single (M*N)-dimensional tuple
            ordered as follows:
            (f1,1, f1,2, ..., f1,n, ..., f1N,
             f2,1, f2,2, ..., f2,n, ..., f2N,
             ...
             fm,1, fm,2, ..., fm,n, ..., fmN,
             ...
             fM,1, fM,2, ..., fM,n, ..., fMN)

            Return None if there is no feasible solution.
            You may assume that if a solution exists, it will be bounded,
            i.e. not infinity.

    You can take advantage of your solveIP function.

    """
    cost = []
    constraints = []
    M = len(W) #num food providers
    N = len(C) #num communities
    total = M*N #total number of points needed
    base = [0] * total #basic list initialized with all values being 0
    for i in range(M):
        for x in range(N):
            cost.append(T[i][x]) #sets up cost tuple getting the x-th cost from provider i
            cons = copy(base) #copies basic list
            cons[i*N+x] = W[i] #sets constraints for truck weight
            constraints += [(tuple(cons), truck_limit)]
    for i in range(N):
        cons = copy(base) #copies basic list
        for y in range(M):
            cons[y*N+i] = -1 
        constraints += [(tuple(cons), -C[i])] #sets constraint for food greater than or equal to minimum
    for i in range(total):
        cons = copy(base)
        cons[i] = -1
        constraints += [(tuple(cons), 0)] #sets constraint for point greater than or equal to 0
    return solveIP(constraints, cost)


if __name__ == "__main__":
    constraints = [((3, 2), 10),((1, -9), 8),((-3, 2), 40),((-3, -1), 20)]
    inter = findIntersections(constraints)
    print(inter)
    print()
    valid = findFeasibleIntersections(constraints)
    print(valid)
    print()
    print(solveLP(constraints, (3,5)))
    print()
    print(solveIP(constraints, (3,5)))
    print()
    print(wordProblemIP())
