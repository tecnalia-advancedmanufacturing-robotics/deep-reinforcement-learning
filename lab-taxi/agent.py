import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, env, nA=6, nS=500):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        env.nA = nA
        self.env = env
        self.Q = np.zeros((nS, nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.env.action_space.sample()

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += 0
