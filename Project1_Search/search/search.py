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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringeStrategy = util.PriorityQueueWithFunction(lambda x: -len(x[1]))
    return graphSearch(problem, fringeStrategy, nullHeuristic)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringeStrategy = util.PriorityQueueWithFunction(lambda x: len(x[1]))
    return graphSearch(problem, fringeStrategy, nullHeuristic)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringeStrategy = util.PriorityQueueWithFunction(lambda x: x[2])
    return graphSearch(problem, fringeStrategy, nullHeuristic)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringeStrategy = util.PriorityQueueWithFunction(lambda x: x[2])
    return graphSearch(problem, fringeStrategy, heuristic)

def getHeuristic(state, problem, heuristic, dic):
    """
    wrapper for the heuristics to avoid recomputing when already called for a specific state. The heuristics are stored in dic
    """
    if state not in dic:
        h = heuristic(state, problem)
        dic[state] = h
        return h
    else:
        return dic[state]

def graphSearch(problem, fringe, heuristic):
    """
    Generic Graph seach algorithm
    fringe is a priority queue of class PriorityQueueWithFunction that defines the node selection strategy used when calling pop()
    """
    startState = problem.getStartState()
    closed = set()
    """heuristics dictionary that only holds states that have been reached after an expansion"""
    h = {startState : heuristic(startState, problem)}
    """push in the fringe the (state, path, pathCost)"""
    fringe.push((startState, [], getHeuristic(startState, problem, heuristic, h)))
    while True:
        if fringe.isEmpty():
            raise Exception, 'GraphSearch could not find a solution'
        node_state, node_path, node_pathCost = fringe.pop()
        if problem.isGoalState(node_state):
            return node_path
        if node_state not in closed:
           closed.add(node_state)
           for childNode_state, childNode_path, childNode_actionCost in problem.getSuccessors(node_state):
               fringe.push((childNode_state \
                            , node_path + [childNode_path] \
                            , node_pathCost - getHeuristic(node_state, problem, heuristic, h) + childNode_actionCost + getHeuristic(childNode_state, problem, heuristic, h) \
                            ))
               #print "Child node: ", childNode_state, " ", node_path + [childNode_path]

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
