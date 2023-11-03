from agent import Agent
from monitor import interact
import gym
import numpy as np
from renderer import save_random_agent_gif

env = gym.make('Taxi-v3', render_mode='rgb_array')
agent = Agent(env)
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=200000)

save_random_agent_gif(env, agent)
