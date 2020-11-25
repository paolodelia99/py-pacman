from enum import Enum
from typing import Union

import gym
from gym import spaces
from gym.utils import seeding

from src.game import Game
from src.controller import Controller
from src.map import Map
from src.utils.game_mode import GameMode


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NONE = 4


class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, layout: str, frame_to_skip: int, enable_render=True, state_active=False):
        """

        :type frame_to_skip: int
        """
        self.layout = layout
        self.state_active = state_active
        self.enable_render = enable_render
        self.frame_to_skip = frame_to_skip
        self.action_space = spaces.Discrete(4)
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(
            maze=self.maze,
            screen=Controller.get_screen(state_active, self.width, self.height) if enable_render else None,
            sounds_active=False,
            state_active=state_active
        )

        self.seed()

    def __del__(self):
        if self.enable_render:
            self.game.quit_game()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.game.restart()
        self.game.set_mode(GameMode.normal)

    def render(self, mode='human'):
        self.game.init_screen()
        self.game.draw()

    def act(self, action: int):
        """
        Perform an action on the game. We lockstep frames with actions. If act is not called the game will not run.

        :argument action
            The index of the action we wish to perform. The index usually corresponds to the index item returned by getActionSet().

        :returns: Returns the reward that the agent has accumlated while performing the action.

        """
        return sum(self._one_step_action(action) for i in range(self.frame_to_skip))

    def step(self, action: Union[Action, int]):
        pass

    def _one_step_action(self, action):
        pass


ACTION_LOOKUP = {
    0: Action.LEFT,
    1: Action.RIGHT,
    2: Action.UP,
    3: Action.DOWN,
    4: Action.NONE,
}
