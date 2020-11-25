import numpy as np

from gym.spaces import Box, Discrete
from src.env.pacman_env import PacmanEnv
from src.map import Map


def test_box():
    space = Box(low=0, high=100, shape=(10, 10), dtype=np.int32)
    assert type(space.sample()) is np.int32


def test_action_space():
    env = PacmanEnv()
    assert type(env.action_space.sample()) is str
