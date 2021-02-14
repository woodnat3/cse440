# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        
        for i in range(self.iterations): #runs each iteration
            valsCopy = self.values.copy() #used to update values at the end
            for state in states: #goes through each state
                final = None #keeps track of best final value for state
                for action in self.mdp.getPossibleActions(state):
                    current = self.computeQValueFromValues(state, action)
                    if final == None or final < current: #sets final to best value
                        final = current
                if final == None:
                    final = 0
                valsCopy[state] = final 
            self.values = valsCopy #sets values after all states have been set with a value
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        trans = self.mdp.getTransitionStatesAndProbs(state, action)
        final = 0 #keeps track of final Q-Value
        for nextState, prob in trans:
            reward = self.mdp.getReward(state, action, nextState) #calculates the reward
            val = self.discount * self.values[nextState] #calculates the value of the state
            #use formula given in lecture
            final += prob * (reward + val) #adds Q-Value to final for this s'
        return final

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0: #if terminal state
            return None
        
        val = None #tracks best value from action
        act = None #tracks best action based on value
        for action in actions: #goes through all possible actions
            tempVal = self.computeQValueFromValues(state, action)
            if val == None or val < tempVal: #if new val is better, change action
                val = tempVal
                act = action
        return act

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates() #get all states
        for i in range(self.iterations): #does all iterations
            state = states[i%len(states)] #does one state per iteration
            if self.mdp.isTerminal(state): #passes if it is a terminal state
                continue
            actions = self.mdp.getPossibleActions(state)
            final = None #keeps track of best value from actions
            for action in actions:
                current = self.computeQValueFromValues(state, action)
                if final == None or final < current: #if current action has better value
                    final = current #use this value instead
            if final == None:
                final = 0
            self.values[state] = final #sets state to best value from action

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        priority = util.PriorityQueue()
        pred = {} #keeps track of predecessors
        states = self.mdp.getStates()
        for state in states: #goes through all states
          if not self.mdp.isTerminal(state):
            for action in self.mdp.getPossibleActions(state):
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                if nextState in pred: #if we have already seen nextState
                  pred[nextState].add(state) #add current to its predecessors
                else:
                  pred[nextState] = {state} #otherwise set current as its predecessor

        for state in states: #goes through all states after tracking predecessors
          if not self.mdp.isTerminal(state):
            bestVal = None #stores best value
            for action in self.mdp.getPossibleActions(state):
              qValue = self.computeQValueFromValues(state, action)
              if bestVal == None or qValue > bestVal:
                  bestVal = qValue
            diff = abs(bestVal - self.values[state]) #calculates difference
            priority.update(state, - diff) #updates priority queue with difference

        for i in range(self.iterations): #does all iterations
          if priority.isEmpty(): #only if priority queue isn't empty
            break
          temp = priority.pop() #takes state with highest priority
          if not self.mdp.isTerminal(temp):
            bestVal = None #stores best value
            actions = self.mdp.getPossibleActions(temp)
            for action in actions:
              qValue = self.computeQValueFromValues(temp, action)
              if bestVal == None or qValue > bestVal:
                  bestVal = qValue
            self.values[temp] = bestVal #value gets set to the best value

          for state in pred[temp]:
            if not self.mdp.isTerminal(state):
              bestVal = None #stores best value
              for action in self.mdp.getPossibleActions(state):
                qValue = self.computeQValueFromValues(state, action)
                if bestVal == None or qValue > bestVal:
                    bestVal = qValue
              diff = abs(bestVal - self.values[state]) #calculates difference
              if diff > self.theta: #change priority if difference is greater than theta
                priority.update(state, -diff) #uses difference to set priority

