# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent

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

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

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
    
def breadthFirstSearch(problem, heuristic = nullHeuristic):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringeStrategy = util.PriorityQueueWithFunction(lambda x: len(x[1]))
    return graphSearch(problem, fringeStrategy, nullHeuristic)

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
            #raise Exception, 'GraphSearch could not find a solution'
            return []
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

# Abbreviations
bfs = breadthFirstSearch

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems
        # Note: this bit of Python trickery combines the search algorithm and the heuristic
        self.searchFunction = lambda x: fn(x, heuristic=heuristic)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        #print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y]

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return bfs(problem)

class AnyGhostSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to a ghost agent.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        # Store the ghosts' states for later reference
        self.ghosts = gameState.getGhostStates()
        self.ghostsPositions = [ghost.getPosition() for ghost in self.ghosts]

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return (x,y) in self.ghostsPositions

class ClosestGhostSearchAgent(SearchAgent):
    "Search for closest ghost using bfs"

    def findPathToClosestGhost(self, gameState):
        """
        Returns a path (a list of actions) to the closest ghost, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyGhostSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return bfs(problem)

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        newFoodCount = len(newFood.asList())

        closestGhost = 999999
        for ghost in newGhostStates:
            ghostDist = util.manhattanDistance(ghost.getPosition(),newPos)
            if ghostDist < closestGhost:
                closestGhost = ghostDist

        evalSearchAgent = ClosestDotSearchAgent(prob='PositionSearchProblem')
        distClosestDot = len(evalSearchAgent.findPathToClosestDot(successorGameState))
        #print "distClosestDot %d" % distClosestDot
 
        if action == "Stop":
            return 0
        if sum(newScaredTimes) > 0:
            return 1.25* ( 0.001/(distClosestDot+1) + 1.0/(newFoodCount+1) ) + 10
        if closestGhost < 5:
            return closestGhost/20 + 0.001/(distClosestDot+1) + 0.6/(newFoodCount+1)
        return 1.25* ( 0.001/(distClosestDot+1) + 2.0/(newFoodCount+1) )

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.value(gameState.generateSuccessor(self.index, action), 0, self.nextAgent(gameState, self.index)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def value(self, gameState, ply, agentIndex):
        """
          Dispatch function
        """
        "is terminal State?"
        if ply == self.depth and agentIndex == self.index:
            return self.evaluationFunction(gameState)
        "is winning/losing State?"
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        "is max agent?"
        if agentIndex == self.index: return self.maxValue(gameState, ply, agentIndex)
        "is min agent?"
        if agentIndex != self.index: return self.minValue(gameState, ply, agentIndex)

    def maxValue(self, gameState, ply, agentIndex):
        """
          Maximizer function
        """
        nextAgentIndex = self.nextAgent(gameState, agentIndex)
        if nextAgentIndex == self.index: ply+=1
        return max([self.value(gameState.generateSuccessor(agentIndex, action), ply, nextAgentIndex) for action in gameState.getLegalActions(agentIndex)])

    def minValue(self, gameState, ply, agentIndex):
        """
          Minimizer function
        """
        nextAgentIndex = self.nextAgent(gameState, agentIndex)
        if nextAgentIndex == self.index: ply+=1
        return min([self.value(gameState.generateSuccessor(agentIndex, action), ply, nextAgentIndex) for action in gameState.getLegalActions(agentIndex)])

    def nextAgent(self, gameState, agentIndex):
       """
         return next agent's index
       """
       return (agentIndex+1)%gameState.getNumAgents()
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        alpha = -999999
        beta = 999999
        scores = []
        ply = 0
        for action in legalMoves:
            v = self.value(gameState.generateSuccessor(self.index, action), ply, self.nextAgent(gameState, self.index), alpha, beta)
            scores.append(v)
            alpha = max(alpha, v)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def value(self, gameState, ply, agentIndex, alpha, beta):
        """
          Dispatch function
        """
        "is terminal State?"
        if ply == self.depth and agentIndex == self.index:
            return self.evaluationFunction(gameState)
        "is winning/losing State?"
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        "is max agent?"
        if agentIndex == self.index: return self.maxValue(gameState, ply, agentIndex, alpha, beta)
        "is min agent?"
        if agentIndex != self.index: return self.minValue(gameState, ply, agentIndex, alpha, beta)

    def maxValue(self, gameState, ply, agentIndex, alpha, beta):
        """
          Maximizer function
        """
        v = -999999
        nextAgentIndex = self.nextAgent(gameState, agentIndex)
        if nextAgentIndex == self.index: ply+=1
        for action in gameState.getLegalActions(agentIndex):
            vSuccessor = self.value(gameState.generateSuccessor(agentIndex, action), ply, nextAgentIndex, alpha, beta)
            v = max(v, vSuccessor)
            if v > beta: return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, ply, agentIndex, alpha, beta):
        """
          Minimizer function
        """
        v = 999999
        nextAgentIndex = self.nextAgent(gameState, agentIndex)
        if nextAgentIndex == self.index: ply+=1
        for action in gameState.getLegalActions(agentIndex):
            vSuccessor = self.value(gameState.generateSuccessor(agentIndex, action), ply, nextAgentIndex, alpha, beta)
            v = min(v, vSuccessor)
            if v < alpha: return v
            beta = min(beta, v)
        return v

    def nextAgent(self, gameState, agentIndex):
       """
         return next agent's index
       """
       return (agentIndex+1)%gameState.getNumAgents()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.value(gameState.generateSuccessor(self.index, action), 0, self.nextAgent(gameState, self.index)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def value(self, gameState, ply, agentIndex):
        """
          Dispatch function
        """
        "is terminal State?"
        if ply == self.depth and agentIndex == self.index:
            return self.evaluationFunction(gameState)
        "is winning/losing State?"
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        "is max agent?"
        if agentIndex == self.index: return self.maxValue(gameState, ply, agentIndex)
        "is min agent?"
        if agentIndex != self.index: return self.minValue(gameState, ply, agentIndex)

    def maxValue(self, gameState, ply, agentIndex):
        """
          Maximizer function
        """
        nextAgentIndex = self.nextAgent(gameState, agentIndex)
        if nextAgentIndex == self.index: ply+=1
        return max([self.value(gameState.generateSuccessor(agentIndex, action), ply, nextAgentIndex) for action in gameState.getLegalActions(agentIndex)])

    def minValue(self, gameState, ply, agentIndex):
        """
          Minimizer Expected function
        """
        nextAgentIndex = self.nextAgent(gameState, agentIndex)
        if nextAgentIndex == self.index: ply+=1
        v = [self.value(gameState.generateSuccessor(agentIndex, action), ply, nextAgentIndex) for action in gameState.getLegalActions(agentIndex)]
        return 1.0*sum(v)/len(v)

    def nextAgent(self, gameState, agentIndex):
       """
         return next agent's index
       """
       return (agentIndex+1)%gameState.getNumAgents()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #get distance to closest food
    dotSearchAgent = ClosestDotSearchAgent(prob='PositionSearchProblem')
    distClosestDot = len(dotSearchAgent.findPathToClosestDot(currentGameState))
    #get distance to closest ghost
    ghostSearchAgent = ClosestGhostSearchAgent(prob='PositionSearchProblem')
    distClosestGhost = len(ghostSearchAgent.findPathToClosestGhost(currentGameState))
    #remaining food count
    dotCount = len(currentGameState.getFood().asList())

    if distClosestGhost < 5:
        return distClosestGhost/20 + 0.001/(distClosestDot+1) + 0.6/(dotCount+1)
    return 1.25* ( 0.001/(distClosestDot+1) + 2.0/(dotCount+1) )

# Abbreviation
better = betterEvaluationFunction

