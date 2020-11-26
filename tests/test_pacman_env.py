import numpy as np

from gym.spaces import Box, Discrete
from src.env.pacman_env import PacmanEnv
from src.map import Map
from src.utils.game_mode import GameMode


def test_action_space():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=10,
        enable_render=False)
    assert type(env.action_space.sample()) is int


def test_env_reset():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=10,
        enable_render=False
    )
    env.reset()
    assert env.get_mode() == GameMode.normal
    assert env.game.player.anim_frame == 3
    assert env.game.player.vel_x == 0
    assert env.game.player.vel_y == 0


def test_env_render():
    env = PacmanEnv(
        layout='classic',
        frame_to_skip=10,
        enable_render=True
    )
    env.reset()
    env.render()
    assert 1 == 1
