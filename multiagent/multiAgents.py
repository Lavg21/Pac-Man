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
from game import Directions
import random, util, math

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = 0  # we initialize the score with 0

        food = newFood.asList()  # we keep the food positions as a list
        # we retain the total number of the food displayed on the grid
        totalFood = len(food)
        foodDistance = math.inf  # we initialize the foodDistance with the highest value possible

        totalGhosts = len(newGhostStates)  # we retain the number of ghosts

        # if we don't have any food available
        if totalFood == 0:
            foodDistance = 0  # there are no distances

        for item in range(totalFood):
            # calculate the distance from every food available
            mhFood = manhattanDistance(newPos, food[item])
            nearestFood = 1000 * totalFood + mhFood  # calculate the nearest food available

            # the closer to food, the better, so we actualize the value of the foodDistance if we find something closer
            if nearestFood < foodDistance:
                foodDistance = nearestFood

            score -= foodDistance  # we add the foodDistance to the score

        for pos in range(totalGhosts):
            ghostPosition = successorGameState.getGhostPosition(
                pos + 1)  # get the ghost position
            # calculate the distance to the nearest ghost
            mhGhost = manhattanDistance(ghostPosition, newPos)

            if mhGhost <= 1:
                score -= math.inf  # it is too close to us, so we died

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        pacman_agent = 0

        def maximizer(state, depth):
          # verify if Pacman won or not
          # either way, the algorithm stops and we return the score
          if state.isWin() or state.isLose():
            return state.getScore()
          # get all legal actions for Pacman
          actions = state.getLegalActions(pacman_agent)
          # initialize maximum score
          max_score = -math.inf
          aux_score = max_score
          best_action = Directions.STOP
          # we're looking for the action that maximizes the score
          for action in actions:
            aux_score = minimizer(state.generateSuccessor(
                pacman_agent, action), depth, 1)
            if aux_score > max_score:
              # update score
              max_score = aux_score
              best_action = action
          # we've reached the last level
          if depth == 0:
            return best_action
          else:
            return max_score

        def minimizer(state, depth, agent):
          # check if the game is over and return the score
          if state.isWin() or state.isLose():
            return state.getScore()
          # compute next agent index (ghost)
          next_agent = agent + 1
          # if it's last ghost's turn, it will be Pacman's next turn
          if agent == state.getNumAgents() - 1:
            next_agent = pacman_agent
          # get all legal actions for the agent
          actions = state.getLegalActions(agent)
          # initialize minimum score
          min_score = math.inf
          aux_score = min_score
          # we're looking for the action that minimizes the score
          for action in actions:
            # check if it's last ghost
            if next_agent == pacman_agent:
              if depth == self.depth - 1:
                aux_score = self.evaluationFunction(
                    state.generateSuccessor(agent, action))
              else:
                aux_score = maximizer(
                    state.generateSuccessor(agent, action), depth+1)
            else:
              aux_score = minimizer(state.generateSuccessor(
                  agent, action), depth, next_agent)
            if aux_score < min_score:
              # update score
              min_score = aux_score
          return min_score
        return maximizer(gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        pacman = 0

        def maximizer(state, depth, alfa, beta):
            mx = -math.inf  # the initial maximum value
            nextAction = Directions.STOP  # stop command
            actions = state.getLegalActions(pacman)  # the actions of the agent

            # we terminate the state in either case and return the score
            if state.isWin() or state.isLose():
                return state.getScore()

            # we will try to find the max value for every successor
            # depending on the actions of the agent
            for action in actions:
                # retain the successor
                successor = state.generateSuccessor(
                    pacman, action)

                # we calculate the next value
                nextValue = minimizer(successor, depth, 1, alfa, beta)

                # we find the best value
                if nextValue > mx:
                    mx = nextValue  # keep it
                    nextAction = action  # and actualize the nextAction

                # we compare the best value with beta
                if mx > beta:
                    return mx

                alfa = max(mx, alfa)  # keep the new alfa value

            # if the algorithm reached the max depth
            if depth == 0:
                return nextAction  # then we stop
            else:
                return mx  # else we return the best value

        def minimizer(state, depth, agent, alfa, beta):

            mn = math.inf  # the initial minimum value
            nextAgent = agent + 1  # nextAgent = ghost
            actions = state.getLegalActions(agent)  # the actions of the agent

            # we terminate the state in either case and return the score
            if state.isLose() or state.isWin():
                return state.getScore()

            if agent == state.getNumAgents() - 1:
                nextAgent = pacman

            # we will try to find the min value for every successor
            # depending on the actions of the agent
            for action in actions:
                # we retain the successor
                successor = state.generateSuccessor(agent, action)

                if nextAgent == pacman:

                    if depth == self.depth - 1:
                        getScore = self.evaluationFunction(successor)
                    else:
                        getScore = maximizer(successor, depth + 1, alfa, beta)

                else:
                    getScore = minimizer(successor, depth, nextAgent, alfa, beta)

                # we try to get the min value
                if getScore < mn:
                    mn = getScore

                # we compare the score with alfa
                if mn < alfa:
                    return mn

                beta = min(mn, beta)  # keep the new alfa value

            return mn  # we return the minimum

        return maximizer(gameState, 0, -math.inf, math.inf)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
