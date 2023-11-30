import matplotlib.pyplot as plt
import matplotlib.animation


class RandomAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, *args, **kwargs):
        return self.action_space.sample()

    def reset_episode(self, *args, **kwargs):
        return self.action_space.sample()


def render(env, agent, seed=505):
    plt.style.use('ggplot')

    state = env.reset(seed=seed)[0]
    action = agent.reset_episode(state)
    score = 0
    frames = []
    states = []
    while True:
        out = env.render()
        frames.append(out)
        states.append(state)
        state, reward, terminated, truncated, _ = env.step(action)
        action = agent.act(state, reward, terminated or truncated, 'test')
        score += reward
        if terminated or truncated:
            break

    fig = plt.figure()
    im = plt.imshow(out)

    def animation(i):
        im.set_data(frames[i])
        fig.set_title(f'State: {state}')
        return im

    animation = matplotlib.animation.FuncAnimation(fig, animation, frames=len(frames), interval=50)
    plt.close()

    print('Final score:', score)
    env.close()
    return animation.to_html5_video()
