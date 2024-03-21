# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
import numpy as np
from util import nearestPoint

import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


MAX_DEPTH = 20

class MCTSNode(object):

    def __init__(self, gameState, agent, action, parent, enemy_pos, borderline):
        '''MCTS node properties initialization'''
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        self.child = []
        self.visits = 1
        self.q_value = 0.0
        self.epsilon = 0.8
        self.rewards = 0
        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != 'Stop']
        self.unexploredActions = self.legalActions[:]
        self.action = action
        '''Game properties initialization'''
        self.gameState = gameState.deepCopy()
        self.enemy_pos = enemy_pos
        self.borderline = borderline
        self.agent = agent
    
    '''MCTS implementation'''
    def expansion(self):
        if self.depth >= MAX_DEPTH:
            return self

        if self.unexploredActions != []:
            action = self.unexploredActions.pop()
            current_game_state = self.gameState.deepCopy()
            next_game_state = current_game_state.generateSuccessor(self.agent.index, action)
            child_node = MCTSNode(next_game_state, self.agent, action, self, self.enemy_pos, self.borderline)
            self.child.append(child_node)
            return child_node

        if util.flipCoin(self.epsilon):
            next_best_node = self.sel_child_with_highest_UBC1()
        else:
            next_best_node = random.choice(self.child)
        return next_best_node.expansion()

    def simulation(self):
        timeLimit = 0.99
        start = time.time()
        while (time.time() - start < timeLimit):

            node_selected = self.expansion()

            reward = node_selected.cal_reward()

            node_selected.backpropagation(reward)

        return self.sel_child_with_highest_UBC1().action
    
    
    def backpropagation(self, reward):
        self.visits += 1
        self.q_value += reward
        if self.parent is not None:
            self.parent.backpropagation(reward)

    def findANodeInTree(self, gameState):
        if self.gameState == gameState:
            return self
        for child in self.child:
            found = child.findANodeInTree(gameState)
            if found:
                return found
        return None
    
    def sel_child_with_highest_UBC1(self):
        best_score = -np.inf
        best_child = None
        for candidate in self.child:
            # UBC1 value
            score = candidate.q_value / candidate.visits + 2 * math.sqrt((2*math.log(self.visits)) / candidate.visits)
            if score > best_score:
                best_score = score
                best_child = candidate
        return best_child
    
    '''Game related functions'''

    def cal_reward(self):
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        if current_pos == self.gameState.getInitialAgentPosition(self.agent.index):
            return -1000
        value = self.get_features() * MCTSNode.get_weight(self)
        return value

    def get_features(self):
        gameState = self.gameState
        features = util.Counter()
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        features['getFood']=len(self.agent.getFood(gameState).asList())
        features['capsule']=len(self.agent.getCapsules(gameState))
        features['enemy']=len(self.enemy_pos)
        features['minDistToFood'] = self.agent.get_min_dist_to_food(gameState)
        features['distanceToHomeAndFood'] = self.agent.getMazeDistance(current_pos, gameState.getInitialAgentPosition(self.agent.index))*(50-features['getFood'])
        return features

    def get_weight(self):
        return {'minDistToFood': -10, 'getFood': 100, 'capsule': 100, 'enemy': -100, 'distanceToHomeAndFood': -1}

# --------------------------------------------------------------------------

##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.food_count = int(len(self.getFood(gameState).asList()))
        self.arena_width = gameState.data.layout.width
        self.arena_height = gameState.data.layout.height
        self.friendly_borders = self.detect_my_border(gameState)
        self.hostile_borders = self.detect_enemy_border(gameState)
        # self.MCTSNode = MCTSNode(gameState, self, None, None, self.detect_enemy_ghost(gameState), self.friendly_borders)

    def detect_my_border(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2 - 1
        else:
            border_x = self.arena_width // 2
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def detect_enemy_border(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2
        else:
            border_x = self.arena_width // 2 - 1
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def detect_enemy_ghost(self, gameState):
        """
        Return Observable Oppo-Ghost Index
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if (not enemyState.isPacman) and enemyState.scaredTimer == 0:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    enemyList.append(enemy)
        return enemyList

    def detect_enemy_approaching(self, gameState):
        """
        Return Observable Oppo-Ghost Position Within 5 Steps
        """
        dangerGhosts = []
        ghosts = self.detect_enemy_ghost(gameState)
        myPos = gameState.getAgentPosition(self.index)
        for g in ghosts:
            distance = self.getMazeDistance(myPos, gameState.getAgentPosition(g))
            if distance <= 5:
                dangerGhosts.append(g)
        return dangerGhosts

    def detect_enemy_pacman(self, gameState):
        """
        Return Observable Oppo-Pacman Position
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if enemyState.isPacman and gameState.getAgentPosition(enemy) != None:
                enemyList.append(enemy)
        return enemyList

    def chooseAction(self, gameState):
        """
        Picks best actions.
        """
        start = time.time()
        actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        carrying = agent_state.numCarrying
        isPacman = agent_state.isPacman

        if isPacman:
            appr_ghost_pos = [gameState.getAgentPosition(g) for g in self.detect_enemy_approaching(gameState)]
            foodList = self.getFood(gameState).asList()

            if not appr_ghost_pos:
                values = [self.evaluate_off(gameState, a) for a in actions]
                maxValue = max(values)
                bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                action_chosen = random.choice(bestActions)

            elif len(foodList) < 2 or carrying > 7:
                rootNode = MCTSNode(gameState, self, None, None, appr_ghost_pos, self.friendly_borders)
                action_chosen = MCTSNode.simulation(rootNode)
                # # find the node in the tree
                # node= self.MCTSNode.findANodeInTree(gameState)
                # if node:
                #     action_chosen = MCTSNode.simulation(node)
                # else:
                #     rootNode = MCTSNode(gameState, self, None, None, appr_ghost_pos, self.friendly_borders)
                #     action_chosen = MCTSNode.simulation(rootNode)
                #     # get the possible parent of rootNode
                #     # get the next game state
                #     next_game_state = gameState.generateSuccessor(self.index, action_chosen)
            else:
                rootNode = MCTSNode(gameState, self, None, None, appr_ghost_pos, self.friendly_borders)
                action_chosen = MCTSNode.simulation(rootNode)

        else:
            ghosts = self.detect_enemy_ghost(gameState)
            values = [self.evaluate_def(gameState, a, ghosts) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            action_chosen = random.choice(bestActions)

        return action_chosen

    def evaluate_off(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_off_features(gameState, action)
        weights = self.get_off_weights(gameState, action)
        return features * weights

    def get_off_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        next_tate = self.get_next_state(gameState, action)
        if next_tate.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1
        else:
            if len(self.getFood(next_tate).asList()) > 0:
                features['minDistToFood'] = self.get_min_dist_to_food(next_tate)

        if next_tate.getAgentState(self.index).numCarrying > 7:
                    # compute the distance to the start point
            current_pos = next_tate.getAgentState(self.index).getPosition()
            features['distanceToHome'] = self.getMazeDistance(current_pos, gameState.getInitialAgentPosition(self.index))
        # encourage the agent to be Pacman
        features['isPacman'] = next_tate.getAgentState(self.index).isPacman
        return features

    def get_off_weights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'minDistToFood': -1, 'getFood': 100, 'distanceToHome': -10, 'isPacman': 1000}

    def evaluate_def(self, gameState, action, ghosts):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_def_features(gameState, action)
        weights = self.get_def_weights(gameState, action)
        return features * weights

    def get_def_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_next_state(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            current_pos = successor.getAgentState(self.index).getPosition()
            min_distance = min([self.getMazeDistance(current_pos, food) for food in foodList])
            features['distanceToFood'] = min_distance
        return features

    def get_def_weights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1}

    def get_next_state(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def get_min_dist_to_food(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        return min([self.getMazeDistance(myPos, f) for f in self.getFood(gameState).asList()])


class DefensiveAgent(OffensiveAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_off_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        next_tate = self.get_next_state(gameState, action)
        if next_tate.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1
        else:
            if len(self.getFood(next_tate).asList()) > 0:
                features['minDistToFood'] = self.get_min_dist_to_food(next_tate)
        # Computes distance to invaders we can see
        next_state = self.get_next_state(gameState, action)
        my_state = next_state.getAgentState(self.index)
        my_pos = my_state.getPosition()
        enemies = [next_state.getAgentState(i) for i in self.getOpponents(next_state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(my_pos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        return features

    def get_off_weights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'minDistToFood': -1, 'getFood': 100, 'numInvaders': -1000, 'invaderDistance': -1000, 'stop': -100, 'reverse': -2}

    def get_def_features(self, gameState, action):
        features = util.Counter()
        next_state = self.get_next_state(gameState, action)

        my_state = next_state.getAgentState(self.index)
        my_pos = my_state.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if my_state.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [next_state.getAgentState(i) for i in self.getOpponents(next_state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(my_pos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_def_weights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

class SmartOffense(CaptureAgent):
    """
    The offensive agent uses q-learing to learn an optimal offensive policy 
    over hundreds of games/training sessions. The policy changes this agent's 
    focus to offensive features such as collecting pellets/capsules, avoiding ghosts, 
    maximizing scores via eating pellets etc.
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.epsilon = 0.0  # exploration prob
        self.alpha = 0.2  # learning rate
        self.discountRate = 0.8
        self.weights = {'closest-food': -2.2558226236802597,
                        'bias': 1.0856704846852672,
                        '#-of-ghosts-1-step-away': -0.18419418670562,
                        'successorScore': -0.027287497346388308,
                        'eats-food': 9.970429654829946}
        """
		Open weights file if it exists, otherwise start with empty weights.
		NEEDS TO BE CHANGED BEFORE SUBMISSION
		try:
			with open('weights.txt', "r") as file:
				self.weights = eval(file.read())
		except IOError:
				return
		"""
    # ------------------------------- Q-learning Functions -------------------------------

    """
	Iterate through all features (closest food, bias, ghost dist),
	multiply each of the features' value to the feature's weight,
	and return the sum of all these values to get the q-value.
	"""

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return features * self.weights

    """
	Iterate through all q-values that we get from all
	possible actions, and return the highest q-value
	"""

    def getValue(self, gameState):
        qVals = []
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                qVals.append(self.getQValue(gameState, action))
            return max(qVals)

    """
	Iterate through all q-values that we get from all
	possible actions, and return the action associated
	with the highest q-value.
	"""

    def getPolicy(self, gameState):
        values = []
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        if len(legalActions) == 0:
            return None
        else:
            for action in legalActions:
                #self.updateWeights(gameState, action)
                values.append((self.getQValue(gameState, action), action))
        return max(values)[1]

    """
	Calculate probability of 0.1.
	If probability is < 0.1, then choose a random action from
	a list of legal actions.
	Otherwise use the policy defined above to get an action.
	"""

    def chooseAction(self, gameState):
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        action = None

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon)
            if prob:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(gameState)
        return action

    # ------------------------------ Features And Weights --------------------------------

    # Define features to use. NEEDS WORK

    def getFeatures(self, gameState, action):
        # Extract the grid of food and wall locations
        food = gameState.getBlueFood()
        walls = gameState.getWalls()
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
            for opponent in opAgents:
                opPos = gameState.getAgentPosition(opponent)
                opIsPacman = gameState.getAgentState(opponent).isPacman
                if opPos and not opIsPacman:
                    ghosts.append(opPos)

        # Initialize features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(successor)

        # Bias
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Number of Ghosts scared
        # features['#-of-scared-ghosts'] = sum(gameState.getAgentState(opponent).scaredTimer != 0 for opponent in opAgents)

        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / \
                (walls.width * walls.height)

        # Normalize and return
        features.divideAll(10.0)
        return features

    """
	Iterate through all features and for each feature, update
	its weight values using the following formula:
	w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
	"""

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        # Calculate the reward. NEEDS WORK
        reward = nextState.getScore() - gameState.getScore()

        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState)
                          ) - self.getQValue(gameState, action)
            self.weights[feature] = self.weights[feature] + \
                self.alpha*correction * features[feature]

    # -------------------------------- Helper Functions ----------------------------------

    # Finds the next successor which is a grid position (location tuple).

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    # Update weights file at the end of each game
    # def final(self, gameState):
        # print self.weights
        #file = open('weights.txt', 'w')
        # file.write(str(self.weights))

class HardDefense(CaptureAgent):
    """
    A simple reflex agent that takes score-maximizing actions. It's given 
    features and weights that allow it to prioritize defensive actions over any other.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        """

        CaptureAgent.registerInitialState(self, gameState)
        self.myAgents = CaptureAgent.getTeam(self, gameState)
        self.opAgents = CaptureAgent.getOpponents(self, gameState)
        self.myFoods = CaptureAgent.getFood(self, gameState).asList()
        self.opFoods = CaptureAgent.getFoodYouAreDefending(
            self, gameState).asList()

    # Finds the next successor which is a grid position (location tuple).
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # Returns a counter of features for the state
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    # Returns a dictionary of features for the state
    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -100, 'stop': -100, 'reverse': -2}

    # Computes a linear combination of features and feature weights
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    # Choose the best action for the current agent to take
    def chooseAction(self, gameState):
        agentPos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)

        # Distances between agent and foods
        distToFood = []
        for food in self.myFoods:
            distToFood.append(self.distancer.getDistance(agentPos, food))

        # Distances between agent and opponents
        distToOps = []
        for opponent in self.opAgents:
            opPos = gameState.getAgentPosition(opponent)
            if opPos != None:
                distToOps.append(self.distancer.getDistance(agentPos, opPos))

        # Get the best action based on values
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)
