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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Stack used for DFS
    dfs_stack = util.Stack()
    # Iteration to add first set of children from start state
    for initial_child_nodes in problem.getSuccessors(problem.getStartState()):
        path = [initial_child_nodes[1]]
        dfs_stack.push((initial_child_nodes[0], path, initial_child_nodes[2]))
    # List containing visited nodes
    seen_list = [problem.getStartState()]

    while not dfs_stack.isEmpty():

        if not dfs_stack.isEmpty():
            popped_node = dfs_stack.pop()
            if problem.isGoalState(popped_node[0]):
                return popped_node[1]
            else:
                if popped_node[0] not in seen_list:
                    seen_list.append(popped_node[0])
                    for child_node in problem.getSuccessors(popped_node[0]):
                        if child_node[0] not in seen_list:
                            dfs_stack.push((child_node[0], (popped_node[1] + [child_node[1]]), child_node[2]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #queue used for BFS
    bfs_queue = util.Queue()
    # Iteration to add first set of children from start state
    for initial_child_nodes in problem.getSuccessors(problem.getStartState()):
        path = [initial_child_nodes[1]]
        bfs_queue.push((initial_child_nodes[0], path, initial_child_nodes[2]))
    # List containing visited nodes
    seen_list = [problem.getStartState()]

    while not bfs_queue.isEmpty():
        if not bfs_queue.isEmpty():
            popped_node = bfs_queue.pop()
            if problem.isGoalState(popped_node[0]):
                return popped_node[1]
            else:
                if popped_node[0] not in seen_list:
                    seen_list.append(popped_node[0])
                    for child_node in problem.getSuccessors(popped_node[0]):
                        if child_node[0] not in seen_list:
                            bfs_queue.push((child_node[0], (popped_node[1] + [child_node[1]]), child_node[2]))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    ucs_queue = util.PriorityQueue()
    # Iteration to add first set of children from start state
    for initial_child_nodes in problem.getSuccessors(problem.getStartState()):
        path = [initial_child_nodes[1]]
        ucs_queue.push((initial_child_nodes[0], path, initial_child_nodes[2]),initial_child_nodes[2])
    # List containing visited nodes
    seen_list = [problem.getStartState()]

    while not ucs_queue.isEmpty():
        popped_node = ucs_queue.pop()
        if problem.isGoalState(popped_node[0]):
            return popped_node[1]
        else:
            if popped_node[0] not in seen_list:
                seen_list.append(popped_node[0])
                for child_node in problem.getSuccessors(popped_node[0]):
                    if child_node[0] not in seen_list:
                        ucs_queue.push((child_node[0], (popped_node[1] + [child_node[1]]), child_node[2] + popped_node[2]),child_node[2] + popped_node[2])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    astar_queue = util.PriorityQueue()
    # Iteration to add first set of children from start state
    for initial_child_nodes in problem.getSuccessors(problem.getStartState()):
        path = [initial_child_nodes[1]]
        astar_queue.push((initial_child_nodes[0], path, initial_child_nodes[2]), initial_child_nodes[2] + heuristic(initial_child_nodes[0], problem))
    # List containing visited nodes
    seen_list = [problem.getStartState()]

    while not astar_queue.isEmpty():
        popped_node = astar_queue.pop()
        if problem.isGoalState(popped_node[0]):
            print(heuristic(popped_node[0],problem))
            return popped_node[1]
        else:
            if popped_node[0] not in seen_list:
                seen_list.append(popped_node[0])
                for child_node in problem.getSuccessors(popped_node[0]):
                    if child_node[0] not in seen_list:
                        astar_queue.push((child_node[0], (popped_node[1] + [child_node[1]]), child_node[2] + popped_node[2]), child_node[2] + popped_node[2] + heuristic(child_node[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
