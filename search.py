# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def _dfs(problem, state, expanded, moves):
    """
    Helper function for dfs that allows for recurrsion
    Returns a path which _bfs uses to return a list of moves for pacman
    """
    fringe = util.Stack() #I used a Stack for its LIFO structure
    temp = problem.getSuccessors(state)
    for i in temp: # adds successor states to fringe
        fringe.push(i)
    if problem.isGoalState(state): # tests if state is goal
        return [state]
    else:
        while not fringe.isEmpty(): # continues till fringe is empty or 
                                        # solution is found
            nextState = fringe.pop() # takes last state added to fringe
            if nextState[0] not in expanded:
                if problem.isGoalState(nextState[0]): # test if state is goal
                    moves += [(nextState[0], nextState[1])]
                    return [state, nextState[0]]
                expanded.add(nextState[0]) # adds node to expanded
                moves += [(nextState[0], nextState[1])]
                path = _dfs(problem, nextState[0], expanded, moves) # recurrsion
                if path != [] and state != problem.getStartState():
                    return [state] + path 
                elif path != [] and state == problem.getStartState():
                    return path
    return []
    

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    
    expanded = set() # used to not expand previously expanded nodes
    moves = [] # keeps track of actions for return
    if (problem.isGoalState(problem.getStartState()) == False):
        expanded.add(problem.getStartState())
        path = _dfs(problem, problem.getStartState(), expanded, moves)
        final = [] # used to return actions to take
        for move in path: # goes through every state in path
            for direc in moves:
                if direc[0] == move: # gets action associated to that state
                    final += [direc[1]] # adds action to return list
        return final
                    
def _bfs(problem, state, expanded, moves):
    '''
    Adds successor nodes to the end of fringe instead of the beginning to reach
    all of the closest nodes first. Parents are used at the end to get the 
    path which is returned to bfs
    '''
    fringe = util.Queue() # I used Queue for FIFO structure
    parents = {} # keeps track of parents of states for returning path
    temp = problem.getSuccessors(state)
    parents[state] = state # makes first state its own parent
    for i in temp: # adds successors to fringe
        if i not in parents:
            parents[i] = state # sets parent for new states
        fringe.push(i)
    if problem.isGoalState(state): # tests if state is goal
        return [state]
    else:
        while not fringe.isEmpty(): # continues till fringe is empty or return
            nextState = fringe.pop() # gets first added state from fringe
            if nextState[0] not in expanded:
                if problem.isGoalState(nextState[0]): # tests if state is goal
                    path = []
                    parent = parents[nextState]
                    if parent != state:
                        while parent != state: # goes through all parents till
                                                # start state is reached
                            if (nextState[0], nextState[1]) not in moves:
                                moves += [(nextState[0], nextState[1])]
                            path += [nextState[0]] # adds to path
                            nextState = parents[nextState] # gets parent state
                            parent = nextState 
                        if state not in path:
                            return path + [nextState[0]]
                        else:
                            return path
                    else:
                        if (nextState[0], nextState[1]) not in moves:
                            moves += [(nextState[0], nextState[1])]
                        if state not in path:
                            return path + [state, nextState[0]]
                        else:
                            return path + [nextState[0]]
                expanded.add(nextState[0])
                if (nextState[0], nextState[1]) not in moves: # prevents repeat
                    moves += [(nextState[0], nextState[1])]
                temp = problem.getSuccessors(nextState[0])
                for i in temp: # adds successor nodes to back of queue
                    if i not in parents:
                        parents[i] = nextState # sets parent of new states
                    fringe.push(i)
                
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    expanded = set() # exact same beginning function as dfs
    moves = []
    if (problem.isGoalState(problem.getStartState()) == False):
        expanded.add(problem.getStartState())
        path = _bfs(problem, problem.getStartState(), expanded, moves)
        final = []
        for move in path:
            for direc in moves:
                if direc[0] == move:
                    final = [direc[1]] + final
        return final

def _ucs(problem, state, expanded, moves):
    '''
    This function uses the same exact algorithm as in _bfs but the difference
    is I use a priority queue with the function priority() setting the priority
    of it within fringe
    '''
    def priority(given):
        '''
        function used to set priority in fringe based on path cost
        '''
        if given in parents: # adds cost of moving through parents first
            parent = parents[given]
            actions = []
            while state != parent: # continues till start state is reached
                actions = [given[1]] + actions
                given = parents[given]
                parent = parents[given]
            actions = [given[1]] + actions
            return problem.getCostOfActions(actions)
        else:
            return problem.getCostOfActions([given[1]])
    fringe = util.PriorityQueueWithFunction(priority)
    parents = {}
    temp = problem.getSuccessors(state)
    parents[state] = state
    for i in temp:
        if i not in parents:
            parents[i] = state
        fringe.push(i)
    if problem.isGoalState(state):
        return [state]
    else:
        while not fringe.isEmpty():
            nextState = fringe.pop()
            if nextState[0] not in expanded:
                if problem.isGoalState(nextState[0]):
                    path = []
                    parent = parents[nextState]
                    if parent != state:
                        while parent != state:
                            if (nextState[0], nextState[1]) not in moves:
                                moves += [(nextState[0], nextState[1])]
                            path += [nextState[0]]
                            nextState = parents[nextState]
                            parent = nextState
                        if state not in path:
                            return path + [nextState[0]]
                        else:
                            return path
                    else:
                        if (nextState[0], nextState[1]) not in moves:
                            moves += [(nextState[0], nextState[1])]
                        if state not in path:
                            return path + [state, nextState[0]]
                        else:
                            return path + [nextState[0]]
                expanded.add(nextState[0])
                if (nextState[0], nextState[1]) not in moves:
                    moves += [(nextState[0], nextState[1])]
                temp = problem.getSuccessors(nextState[0])
                for i in temp:
                    if i not in parents:
                        parents[i] = nextState
                    fringe.push(i)
                
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    expanded = set() # exact same beginning function as dfs
    moves = []
    if (problem.isGoalState(problem.getStartState()) == False):
        expanded.add(problem.getStartState())
        path = _ucs(problem, problem.getStartState(), expanded, moves)
        final = []
        for move in path:
            for direc in moves:
                if direc[0] == move:
                    final = [direc[1]] + final
        return final

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def _aStar(problem, heuristic, state, expanded, moves):
    '''
    This function uses the same exact algorithm as in _bfs and the same as _ucs
    as I use a priority queue with the function priority() setting the priority
    of it within fringe
    '''
    def priority(given):
        '''
        function used to set priority within fringe based on path cost plus 
        the value returned by the heuristtic given with the problem
        '''
        if given in parents: # only difference is adding cost of heuristic
            parent = parents[given]
            actions = []
            states = []
            while state != parent:
                actions = [given[1]] + actions
                states = [given[0]] + states
                given = parents[given]
                parent = parents[given]
            actions = [given[1]] + actions
            states = [given[0]] + states
            return heuristic(states.pop(), problem) + problem.getCostOfActions(actions)
        else:
            return heuristic(given[0], problem) + problem.getCostOfActions([given[1]])
    fringe = util.PriorityQueueWithFunction(priority)
    parents = {}
    temp = problem.getSuccessors(state)
    parents[state] = state
    for i in temp:
        if i not in parents:
            parents[i] = state
        fringe.push(i)
    if problem.isGoalState(state):
        return [state]
    else:
        while not fringe.isEmpty():
            nextState = fringe.pop()
            if nextState[0] not in expanded:
                if problem.isGoalState(nextState[0]):
                    path = []
                    parent = parents[nextState]
                    if parent != state:
                        while parent != state:
                            if (nextState[0], nextState[1]) not in moves:
                                moves += [(nextState[0], nextState[1])]
                            path += [nextState[0]]
                            nextState = parents[nextState]
                            parent = nextState
                        if state not in path:
                            return path + [nextState[0]]
                        else:
                            return path
                    else:
                        if (nextState[0], nextState[1]) not in moves:
                            moves += [(nextState[0], nextState[1])]
                        if state not in path:
                            return path + [state, nextState[0]]
                        else:
                            return path + [nextState[0]]
                expanded.add(nextState[0])
                if (nextState[0], nextState[1]) not in moves:
                    moves += [(nextState[0], nextState[1])]
                temp = problem.getSuccessors(nextState[0])
                for i in temp:
                    if i not in parents:
                        parents[i] = nextState
                    fringe.push(i)
                
    return []

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    expanded = set() # exact same beginning function as dfs
    moves = []
    if (problem.isGoalState(problem.getStartState()) == False):
        expanded.add(problem.getStartState())
        path = _aStar(problem, heuristic, problem.getStartState(), expanded, moves)
        final = []
        for move in path:
            for direc in moves:
                if direc[0] == move:
                    final = [direc[1]] + final
        return final


        

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
