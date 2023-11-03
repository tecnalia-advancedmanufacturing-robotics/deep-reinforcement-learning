import numpy as np


def epsilon_greedy(env, Qstate, epsilon):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Qstate)
    return action


def sarsa0_estimating_function(env, Q, state, action, next_state, gamma, epsilon):
    next_action = epsilon_greedy(env, Q[next_state], epsilon)
    return Q[next_state][next_action]


def sarsamax_estimating_function(env, Q, state, action, next_state, gamma, epsilon):
    next_action = epsilon_greedy(env, Q[next_state], 0)
    return Q[next_state][next_action]


def expected_sarsa_estimating_function(env, Q, state, action, next_state, gamma, epsilon):
    optimal_action = epsilon_greedy(env, Q[next_state], 0)
    probs = np.ones(env.nA) * epsilon / (env.nA)
    probs[optimal_action] += 1 - epsilon
    return np.dot(Q[next_state], probs)


class InigoAgent:

    def __init__(self, env, nA=6, nS=500):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        env.nA = nA
        self.env = env
        self.Q = np.zeros((nS, nA))
        self.i_episode = 0
        self.epsilon = 0.001
        self.alpha = 0.01
        self.gamma = 1.0
        # self.estimating_function = sarsa0_estimating_function
        # self.estimating_function = sarsamax_estimating_function
        self.estimating_function = expected_sarsa_estimating_function

    def select_action(self, state):
        return epsilon_greedy(self.env, self.Q[state], self.epsilon)

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
        if done:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
        else:
            Gt = self.estimating_function(self.env, self.Q, state, action, next_state, self.gamma, self.epsilon)
            self.Q[state][action] += self.alpha * (reward + self.gamma * Gt - self.Q[state][action])
