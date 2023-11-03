import os
import imageio
import numpy as np


def save_random_agent_gif(env, agent, num_episodes=5):
    frames = []
    for i in range(num_episodes):
        state = env.reset()[0]
        for t in range(500):
            action = agent.select_action(state)

            frame = env.render()
            frames.append(frame)

            state, _, d1, d2, _ = env.step(action)
            if d1 or d2:
                break

    env.close()
    frames = np.array(frames)
    imageio.mimwrite(os.path.join('./', f'{agent.__class__.__name__}.gif'), frames, fps=10)
