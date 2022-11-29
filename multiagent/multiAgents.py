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
import random
import util
import math
import sys

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        score = 0  # we initialize the score with 0

        food = newFood.asList()  # we keep the food positions as a list
        totalFood = len(food)  # we retain the total number of the food displayed on the grid
        foodDistance = math.inf  # we initialize the foodDistance with the highest value possible
        
        totalGhosts = len(newGhostStates)  # we retain the number of ghosts

        # if we don't have any food available
        if totalFood == 0:
            foodDistance = 0  # there are no distances

        for item in range(totalFood):
            mhFood = manhattanDistance(newPos, food[item])  # calculate the distance from every food available
            nearestFood = 1000 * totalFood + mhFood  # calculate the nearest food available

            # the closer to food, the better, so we actualize the value of the foodDistance if we find something closer
            if nearestFood < foodDistance:
                foodDistance = nearestFood

            score -= foodDistance  # we add the foodDistance to the score

        for pos in range(totalGhosts):
            ghostPosition = successorGameState.getGhostPosition(pos + 1)  # get the ghost position
            mhGhost = manhattanDistance(ghostPosition, newPos)  # calculate the distance to the nearest ghost

            if mhGhost <= 1:
                score -= math.inf  # it is too close to us, so we died

        return score


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
        """
        "*** YOUR CODE HERE ***"
        pacman_agent = 0
        def maximizer(state, depth):
          if state.isWin() or state.isLose():
            return state.getScore()
          actions = state.getLegalActions(pacman_agent)
          max_score = float("-inf")
          score = max_score
          best_action = Directions.STOP
          for action in actions:
            score = minimizer(state.generateSuccessor(pacman_agent, action), depth, 1)
            if score > max_score:
              max_score = score
              best_action = action
          if depth == 0:
            return best_action
          else:
            return max_score
        
        def minimizer(state, depth, agent):
          if state.isWin() or state.isLose():
            return state.getScore()
          next_agent = agent + 1
          if agent == state.getNumAgents() - 1:
            next_agent = pacman_agent
          actions = state.getLegalActions(agent)
          min_score = float("inf")
          score = min_score
          for action in actions:
            if next_agent == pacman_agent: # we are on the last ghost and it will be Pacman's turn next
              if depth == self.depth - 1:
                score = self.evaluationFunction(state.generateSuccessor(agent, action))
              else:
                score = maximizer(state.generateSuccessor(agent, action), depth+1)
            else:
              score = minimizer(state.generateSuccessor(
                  agent, action), depth, next_agent)
            if score < min_score:
              min_score = score
          return min_score
        return maximizer(gameState, 0)
        #util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AlfaBeta(gameState, agent, depth):

            # the maximum and minimum values
            mn = math.inf
            mx = -math.inf

            bestValue = []  # the list which will contain the final result

            # If the algorithm reached the max depth
            if depth == self.depth:
                return 0, self.evaluationFunction(gameState)

            # Increase the depth if the ghosts finished a round
            if agent == gameState.getNumAgents() - 1:
                depth += 1

            # If we don't have an agent, we terminate the state
            if not gameState.getLegalActions(agent):
                return 0, self.evaluationFunction(gameState)

            # Calculate the nextAgent
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index  # nextAgent = pacman
            else:
                nextAgent = agent + 1  # nextAgent = ghost

            # We will try to find the MinMax value for every successor
            # depending on the actions of the new agent
            actions = gameState.getLegalActions(agent)
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)  # retain the successor

                # If the list of results is empty
                if not bestValue:
                    nextValue = AlfaBeta(successor, nextAgent, depth)  # we find the best value

                    bestValue.append(nextValue[0])  # and we put it in the list
                    bestValue.append(action)  # along with the action

                    # We fix the limits for the first node
                    if agent == self.index:
                        mn = max(mn, bestValue[0])
                    else:
                        mx = min(mx, bestValue[0])
                else:

                    previousValue = bestValue[0]  # store the previous value
                    nextValue = AlfaBeta(successor, nextAgent, depth)  # calculate the next value

                    # The MaxAgent is pacman
                    if agent == self.index:
                        # we check which value is better and keep it
                        if nextValue[0] > previousValue:
                            bestValue[0] = nextValue[0]
                            bestValue[1] = action
                            mn = max(bestValue[0], mn)  # we actualize the min value
                        else:
                            bestValue[0] = nextValue[0]
                            bestValue[1] = action
                            mx = min(bestValue[0], mx)  # we actualize the max value
                    # The MinAgent is a ghost
                    else:
                        # we check which value is better and keep it
                        if nextValue[0] < previousValue:
                            bestValue[0] = nextValue[0]
                            bestValue[1] = action
                            mn = min(bestValue[0], mx)  # we actualize the max value
                        else:
                            bestValue[0] = nextValue[0]
                            bestValue[1] = action
                            mx = max(bestValue[0], mn)  # we actualize the min value

                    # We check which MinMax value is better than the previous one
                    if (bestValue[0] < mn and agent != self.index) or (bestValue[0] > mx and agent == self.index):
                        return bestValue  # and return it

            return bestValue

        # We call the AlfaBeta with pacman as player at first and initial depth 0
        return AlfaBeta(gameState, self.index, 0)


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
