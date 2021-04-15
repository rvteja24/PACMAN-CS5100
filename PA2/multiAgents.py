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
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        food_manhattan_dist = 1
        rem_food_val = 1 / len(newFood.asList()) if len(newFood.asList()) > 0 else 1
        ghost_val = 50
        if 0 in newScaredTimes:
            temp2 = []
            for state in newGhostStates:
                temp2.append(manhattanDistance(newPos, state.getPosition()))
            ghost_val = min(temp2) if min(temp2) < 7 else ghost_val
        if len(newFood.asList()) > 0:
            temp = []
            for food in newFood.asList():
                temp.append((manhattanDistance(newPos, food), food))
            food_manhattan_dist, fp = min(temp)
            temp = []
            newList = newFood.asList()
            newList.remove(fp)
            for food in newList:
                if fp != food:
                    temp.append((manhattanDistance(fp, food)))
            if len(temp) > 0:
                food_manhattan_dist += min(temp)
        val = (1 / food_manhattan_dist) + 900 * math.exp(rem_food_val) + (ghost_val)
        return val


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        pac_actions = gameState.getLegalActions(0)
        minimaxList = []
        if pac_actions:
            for each in pac_actions:
                new_game_state = gameState.generateSuccessor(0, each)
                minimaxList.append((each, self.getMinVal(1, new_game_state, self.depth)))
            return max(minimaxList, key=lambda x: x[1])[0]

    def getMinVal(self, ghost_index, gameState, ply):
        ghost_count = gameState.getNumAgents() - 1
        temp = []
        moves = gameState.getLegalActions(ghost_index)
        if moves:
            for move in moves:
                new_game_state = gameState.generateSuccessor(ghost_index, move)
                if new_game_state.isLose():
                    temp.append(self.evaluationFunction(new_game_state))
                elif new_game_state.isWin():
                    temp.append(self.evaluationFunction(new_game_state))
                else:
                    if ghost_index == ghost_count and ply == 1:
                        temp.append(self.evaluationFunction(new_game_state))
                    elif ghost_index == ghost_count and ply != 1:
                        temp.append(self.getPacMaxVals(new_game_state, ply))
                    else:
                        temp.append(self.getMinVal(ghost_index + 1, new_game_state, ply))
            return min(temp)
        else:
            return self.evaluationFunction(gameState)

    def getPacMaxVals(self, gameState, ply):
        pac_actions = gameState.getLegalActions(0)
        temp = []
        if pac_actions:
            for each in pac_actions:
                new_game_state = gameState.generateSuccessor(0, each)
                if new_game_state.isLose() or new_game_state.isWin():
                    temp.append(self.evaluationFunction(new_game_state))
                else:
                    temp.append(self.getMinVal(1, new_game_state, ply-1))
            return max(temp)
        else:
            return self.evaluationFunction(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """



    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getPacMaxValsWithPruning(gameState, 0, -math.inf, math.inf)[0]

    def getMinValWithPruning(self, ghost_index, game_state, ply, alpha, beta):
        ghost_count = game_state.getNumAgents() - 1
        minVal = math.inf
        temp = []
        moves = game_state.getLegalActions(ghost_index)
        if len(moves) == 0 or (ply == self.depth) or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        for move in moves:
            new_game_state = game_state.generateSuccessor(ghost_index, move)
            if ghost_index == ghost_count:
                e, val = self.getPacMaxValsWithPruning(new_game_state, ply+1,
                                                          alpha, beta)

            else:
                val = self.getMinValWithPruning(ghost_index + 1, new_game_state, ply,
                                                                                  alpha, beta)
            temp.append(val)
            minVal = min(temp)
            beta = min(minVal, beta)
            if beta < alpha:
                break
        return minVal

    def getPacMaxValsWithPruning(self, game_state, ply, alpha, beta):
        maxVal = -math.inf
        pac_actions = game_state.getLegalActions(0)
        temp = []
        if len(pac_actions) == 0 or game_state.isWin() or game_state.isLose() or ply == self.depth:
            return "", self.evaluationFunction(game_state)

        for each in pac_actions:
            new_game_state = game_state.generateSuccessor(0, each)
            val = self.getMinValWithPruning(1, new_game_state, ply, alpha, beta)
            temp.append((each, val))
            maxVal = max(temp, key=lambda x: x[1])[1]
            alpha = max(maxVal, alpha)
            if alpha > beta:
                break
        return max(temp, key=lambda x: x[1])

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
        return self.getPacMaxValsForExpectimax(gameState, 0)[0]

    def getExpectiVal(self, ghost_index, game_state, ply):
        ghost_count = game_state.getNumAgents() - 1
        temp = []
        moves = game_state.getLegalActions(ghost_index)
        if len(moves) == 0 or (ply == self.depth) or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        for move in moves:
            new_game_state = game_state.generateSuccessor(ghost_index, move)
            if ghost_index == ghost_count:
                e, val = self.getPacMaxValsForExpectimax(new_game_state, ply + 1)
            else:
                val = self.getExpectiVal(ghost_index + 1, new_game_state, ply)
            temp.append(val)

        return sum(temp)/len(moves)

    def getPacMaxValsForExpectimax(self, game_state, ply):
        pac_actions = game_state.getLegalActions(0)
        temp = []
        if len(pac_actions) == 0 or game_state.isWin() or game_state.isLose() or ply == self.depth:
            return "", self.evaluationFunction(game_state)

        for each in pac_actions:
            new_game_state = game_state.generateSuccessor(0, each)
            val = self.getExpectiVal(1, new_game_state, ply)
            temp.append((each, val))

        return max(temp, key=lambda x: x[1])


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Designed a function which would take into consideration the distance to nearest food and the closest from that food, food pellets remaining, distance of ghosts and the time remaining for the ghosts
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghosts]
    food_manhattan_dist = 1
    rem_food_val = 1 / len(foodList) if len(foodList) > 0 else 1
    ghost_val = 50
    if 0 in newScaredTimes or 1 in newScaredTimes or 2 in newScaredTimes:
        temp2 = []
        for state in ghosts:
            temp2.append(manhattanDistance(pos, state.getPosition()))
        ghost_val = min(temp2) if min(temp2) < 4 else ghost_val
    if len(foodList) > 0:
        temp = []
        for food in foodList:
            temp.append((manhattanDistance(pos, food), food))
        food_manhattan_dist, fp = min(temp)
        temp = []
        newList = foodList
        newList.remove(fp)
        for food in newList:
            if fp != food:
                temp.append((manhattanDistance(fp, food)))
        if len(temp) > 0:
            food_manhattan_dist += min(temp)
    val = math.exp(1/food_manhattan_dist) + 140 * math.exp(rem_food_val) + ghost_val
    return val


# Abbreviation
better = betterEvaluationFunction
