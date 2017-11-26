you should be able to play a game of Pacman by typing the following at the command line:

python pacman.py
Pacman lives in a shiny blue world of twisting corridors and tasty round treats. Navigating this world efficiently will be Pacman's first step in mastering his domain.







Project 1: Search

The simplest agent in searchAgents.py is called the GoWestAgent, which always goes West (a trivial reflex agent). This agent can occasionally win:

python pacman.py --layout testMaze --pacman GoWestAgent
But, things get ugly for this agent when turning is required:

python pacman.py --layout tinyMaze --pacman GoWestAgent
If Pacman gets stuck, you can exit the game by typing CTRL-c into your terminal.

Soon, your agent will solve not only tinyMaze, but any maze you want.

Note that pacman.py supports a number of options that can each be expressed in a long way (e.g., --layout) or a short way (e.g., -l). You can see the list of all options and their default values via:

python pacman.py -h
Also, all of the commands that appear in this project also appear in commands.txt, for easy copying and pasting. In UNIX/Mac OS X, you can even run all these commands in order with bash commands.txt.

First, test that the SearchAgent is working correctly by running:

python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
The command above tells the SearchAgent to use tinyMazeSearch as its search algorithm, which is implemented in search.py. Pacman should navigate the maze successfully.

python pacman.py -l tinyMaze -p SearchAgent
python pacman.py -l mediumMaze -p SearchAgent
python pacman.py -l bigMaze -z .5 -p SearchAgent

(BFS) algorithm in the breadthFirstSearch function in search.py. Again, write a graph search algorithm that avoids expanding any already visited states. Test your code the same way you did for depth-first search.

python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
Does BFS find a least cost solution? If not, check your implementation.

Hint: If Pacman moves too slowly for you, try the option --frameTime 0.

code should work equally well for the eight-puzzle search problem without any changes.

python eightpuzzle.py

A* search:
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem

Corners Problem: Heuristic
python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5

Note: AStarCornersAgent is a shortcut for

-p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic.


Eating All The Dots:
python pacman.py -l testSearch -p AStarFoodSearchAgent
python pacman.py -l trickySearch -p AStarFoodSearchAgent

Suboptimal Search:
Sometimes, even with A* and a good heuristic, finding the optimal path through all the dots is hard. In these cases, we'd still like to find a reasonably good path, quickly.
The following agent solves this maze (suboptimally!) in under a second with a path cost of 350:
python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5 

Object Glossary

Here's a glossary of the key objects in the code base related to search problems, for your reference:

SearchProblem (search.py)
A SearchProblem is an abstract object that represents the state space, successor function, costs, and goal state of a problem. You will interact with any SearchProblem only through the methods defined at the top of search.py

PositionSearchProblem (searchAgents.py)
A specific type of SearchProblem that you will be working with --- it corresponds to searching for a single pellet in a maze.

CornersProblem (searchAgents.py)
A specific type of SearchProblem that you will define --- it corresponds to searching for a path through all four corners of a maze.

FoodSearchProblem (searchAgents.py)
A specific type of SearchProblem that you will be working with --- it corresponds to searching for a way to eat all the pellets in a maze.

Search Function
A search function is a function which takes an instance of SearchProblem as a parameter, runs some algorithm, and returns a sequence of actions that lead to a goal. Example of search functions are depthFirstSearch and breadthFirstSearch, which you have to write. You are provided tinyMazeSearch which is a very bad search function that only works correctly on tinyMaze.

SearchAgent
SearchAgent is a class which implements an Agent (an object that interacts with the world) and does its planning through a search function. The SearchAgent first uses the search function provided to make a plan of actions to take to reach the goal state, and then executes the actions one at a time.















Project 2: Multi-Agent Search:

Multi-Agent Pacman

First, play a game of classic Pacman:

python pacman.py
Now, run the provided ReflexAgent in multiAgents.py:

python pacman.py -p ReflexAgent
Note that it plays quite poorly even on simple layouts:

python pacman.py -p ReflexAgent -l testClassic
Inspect its code (in multiAgents.py) and make sure you understand what it's doing.

python pacman.py -p ReflexAgent -l testClassic
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2

Options: Default ghosts are random; you can also play for fun with slightly smarter directional ghosts using -g DirectionalGhost. If the randomness is preventing you from telling whether your agent is improving, you can use -f to run with a fixed random seed (same random choices every game). You can also play multiple games in a row with -n. Turn off graphics with -q to run lots of games quickly.

Minimax:
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
When Pacman believes that his death is unavoidable, he will try to end the game as soon as possible because of the constant penalty for living. Sometimes, this is the wrong thing to do with random ghosts, but minimax agents always assume the worst:
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3

Alpha-Beta Pruning:
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

Expectimax:
To see how the ExpectimaxAgent behaves in Pacman, run:

python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
You should now observe a more cavalier approach in close quarters with ghosts. In particular, if Pacman perceives that he could be trapped but might escape to grab a few more pieces of food, he'll at least try. Investigate the results of these two scenarios:

python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10
















Project 3: Reinforcement Learning

Value iteration and Q-learning
MDPs

To get started, run Gridworld in manual control mode, which uses the arrow keys:

python gridworld.py -m
You will see the two-exit layout from class. The blue dot is the agent. Note that when you press up, the agent only actually moves north 80% of the time. Such is the life of a Gridworld agent!

You can control many aspects of the simulation. A full list of options is available by running:

python gridworld.py -h
The default agent moves randomly

python gridworld.py -g MazeGrid


Value Iteration:
python gridworld.py -a value -i 100 -k 10
python gridworld.py -a value -i 5

python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2

Q-Learning:
Note that your value iteration agent does not actually learn from experience. Rather, it ponders its MDP model to arrive at a complete policy before ever interacting with a real environment. When it does interact with the environment, it simply follows the precomputed policy (e.g. it becomes a reflex agent). This distinction may be subtle in a simulated environment like a Gridword, but it's very important in the real world, where the real MDP is not available.

With the Q-learning update in place, you can watch your Q-learner learn under manual control, using the keyboard:
python gridworld.py -a q -k 5 -m

Epsilon-greedy action selection:
python gridworld.py -a q -k 100 

With no additional code, you should be able to run a Q-learning crawler robot:
python crawler.py

python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1

Approximate Q-learning and State Abstraction:
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 

Note: If you want to watch 10 training games to see what's going on, use the command:
python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10

Approximate Q-learning agent that learns weights for features of states:
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid 

python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid 
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic 

