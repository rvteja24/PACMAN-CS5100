# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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
from math import floor

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            states = self.mdp.getStates()
            prev_values = self.values.copy()
            for each in states:
                actions = self.mdp.getPossibleActions(each)
                action_value_list = []
                for act in actions:
                    sum_over_each_act = 0
                    transistion_probs_and_acts = self.mdp.getTransitionStatesAndProbs(each, act)
                    for each_value in transistion_probs_and_acts:
                        reward = self.mdp.getReward(each, act, each_value[0])
                        prob = each_value[1]
                        each_act = prob*(reward + (self.discount * (prev_values[each_value[0]])))
                        sum_over_each_act += each_act
                    action_value_list.append(sum_over_each_act)
                if len(action_value_list) == 0:
                    action_value_list.append(0)
                self.values[each] = max(action_value_list)


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
        transistion_probs_and_acts = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for each in transistion_probs_and_acts:
            q_value += each[1] * ((self.mdp.getReward(state, action, each[0])) + (self.discount * (self.values[each[0]])))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        action_value = []
        for each in actions:
            action_value.append((each, self.computeQValueFromValues(state, each)))
        if len(action_value) == 0:
            action_value.append(("None", 0))
        return max(action_value, key= lambda x: x[1])[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        #print(len(states))
        for i in range(self.iterations):
            k = floor(i/len(states))
            each = states[i-(k*len(states))]
            prev_values = self.values.copy()
            actions = self.mdp.getPossibleActions(each)
            action_value_list = []
            for act in actions:
                sum_over_each_act = 0
                transistion_probs_and_acts = self.mdp.getTransitionStatesAndProbs(each, act)
                for each_value in transistion_probs_and_acts:
                    reward = self.mdp.getReward(each, act, each_value[0])
                    prob = each_value[1]
                    each_act = prob*(reward + (self.discount * (prev_values[each_value[0]])))
                    sum_over_each_act += each_act
                action_value_list.append(sum_over_each_act)
            if len(action_value_list) == 0:
                action_value_list.append(0)
            self.values[each] = max(action_value_list)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {}
        for each in states:
            actions = self.mdp.getPossibleActions(each)
            for act in actions:
                transition_probabilities = self.mdp.getTransitionStatesAndProbs(each, act)
                for k in transition_probabilities:
                    if k[1] > 0:
                        if k[0] in predecessors.keys():
                            predecessors[k[0]].add(each)
                        else:
                            predecessors[k[0]] = set()
                            predecessors[k[0]].add(each)

        p_queue = util.PriorityQueue()

        for each in states:
            if not self.mdp.isTerminal(each):
                val = self.values[each]
                actions = self.mdp.getPossibleActions(each)
                q_vals = []
                for act in actions:
                    q_val = self.computeQValueFromValues(each, act)
                    q_vals.append(q_val)
                diff = abs(val - max(q_vals))
                p_queue.push(each, -diff)

        for i in range(self.iterations):
            if p_queue.isEmpty():
                break
            sta = p_queue.pop()
            actions = self.mdp.getPossibleActions(sta)
            q_vals = []
            for act in actions:
                q_val = self.computeQValueFromValues(sta, act)
                q_vals.append(q_val)
            self.values[sta] = max(q_vals)
            preds = predecessors.get(sta)
            for ea in preds:
                val = self.values[ea]
                actions = self.mdp.getPossibleActions(ea)
                q_vals = []
                for act in actions:
                    q_val = self.computeQValueFromValues(ea, act)
                    q_vals.append(q_val)
                diff = abs(val - max(q_vals))
                if diff > self.theta:
                    p_queue.update(ea, -diff)

