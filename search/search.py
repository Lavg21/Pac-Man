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
    return [s, s, w, s, w, w, s, w]


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
    stack = util.Stack()
    visited = []
    initialNode = (problem.getStartState(), [])

    stack.push(initialNode)

    while not stack.isEmpty():
        currState, moves = stack.pop()
        if currState not in visited:
            visited.append(currState)

            if problem.isGoalState(currState):
                return moves
            else:
                successors = problem.getSuccessors(currState)
                for successorState, successorMove, successorCost in successors:
                    newMove = moves + [successorMove]
                    newNode = (successorState, newMove)
                    stack.push(newNode)
    return moves


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()  # bfs is using a queue
    start = (problem.getStartState(), [])  # the start node: location, path
    visited = []  # list to check if a node was already visited

    queue.push(start)  # insert the start node in queue

    while not queue.isEmpty():
        current_node = queue.pop()
        # 0 for location, 1 for path

        if problem.isGoalState(current_node[0]):
            return current_node[1]  # if current_node is the goal state, then we return its path

        if current_node[0] not in visited:
            visited.append(current_node[0])  # we add it to the list of visited nodes

            successors = list(problem.getSuccessors(current_node[0]))  # keep the successors and their paths

            for successor in successors:
                if successor[0] not in visited:
                    path = current_node[1] + [successor[1]]  # we reconstruct the path
                    queue.push((successor[0], path))  # we add them to the queue

    # return empty if we don't have any nodes left to explore
    return []
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priorityQueue = util.PriorityQueue()
    visited = {}
    initialNode = (problem.getStartState(), [], 0)

    priorityQueue.push(initialNode, 0)

    while not priorityQueue.isEmpty():
        currState, moves, currCost = priorityQueue.pop()
        if (currState not in visited) or (currCost < visited[currState]):
            visited[currState] = currCost
            if problem.isGoalState(currState):
                return moves
            else:
                successors = problem.getSuccessors(currState)
                for successorState, successorMove, successorCost in successors:
                    newMove = moves + [successorMove]
                    newCost = currCost + successorCost
                    newNode = (successorState, newMove, newCost)

                    priorityQueue.update(newNode, newCost)
    return moves


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    priority_queue = util.PriorityQueue()  # aStarSearch is using a priority queue
    start = (problem.getStartState(), [], 0)  # the start node: location, path, cost
    visited = []  # list to check if a node was already visited

    priority_queue.push(start, 0)  # insert the start node in queue

    while not priority_queue.isEmpty():
        current_node = priority_queue.pop()
        # 0 for location, 1 for path, 2 for cumulative cost

        if problem.isGoalState(current_node[0]):
            return current_node[1]  # if current_node is the goal state, then we return its path

        if current_node[0] not in visited:
            visited.append(current_node[0])  # we add it to the list of visited nodes

            successors = list(problem.getSuccessors(current_node[0]))  # keep the successors and their paths

            for successor in successors:
                if successor[0] not in visited:
                    path = current_node[1] + [successor[1]]  # we reconstruct the path
                    # the path cost to the current node + the path cost to the current successor
                    initial_cost = current_node[2] + successor[2]
                    # the total cost is the sum  of the previous cost and the heuristic of the successor
                    cost = initial_cost + heuristic(successor[0], problem)
                    priority_queue.push((successor[0], path, initial_cost), cost)  # we add them to the priority queue

    # return empty if we don't have any nodes left to explore
    return []

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
