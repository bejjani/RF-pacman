# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        vCurrent = util.Counter()
        vPrevious = util.Counter()
        states = mdp.getStates()
        "wrap this with an iteration loop"
        for iter in range(iterations):
            vPrevious = vCurrent.copy()
            for s in [state for state in states if not mdp.isTerminal(state)]:
                #print "iter %d for state %s" % (iter, s)
                vActions = util.Counter()
                for a in mdp.getPossibleActions(s):
                    #print a
                    vNextState = util.Counter()
                    for sNext, t in mdp.getTransitionStatesAndProbs(s, a):
                        #print "\t Next state: %s with  transition %f and value %f" % (sNext, t, vPrevious[sNext])
                        vNextState[sNext] = t * ( mdp.getReward(s, a, sNext) + self.discount * vPrevious[sNext] )
                    vActions[a] = vNextState.totalCount()
                    #print "\t %f" % vActions[a]
                vCurrent[s] = max(vActions.values())
        self.values = vCurrent.copy()


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
        vNextState = util.Counter()
        for sNext, t in self.mdp.getTransitionStatesAndProbs(state, action):
            vNextState[sNext] = t * ( self.mdp.getReward(state, action, sNext) + self.discount * self.values[sNext] )
        return vNextState.totalCount()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.mdp.getPossibleActions(state):
            return None
        else:
            actions = util.Counter()
            for a in self.mdp.getPossibleActions(state):
                actions[a] = self.computeQValueFromValues(state, a)
            return actions.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
