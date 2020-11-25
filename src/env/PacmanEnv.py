import gym


class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def step(self, action):
        pass
