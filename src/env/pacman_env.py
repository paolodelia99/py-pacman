from typing import Union, Tuple

import gym
import numpy as np
import pygame as pg
from gym import spaces
from gym.utils import seeding

from src.controller import Controller
from src.game import Game
from src.map import Map
from src.utils.action import Action
from src.utils.game_mode import GameMode


class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}
    reward_range = (-10, 5)

    def __init__(self, layout: str, frame_to_skip: int, enable_render=True, state_active=False):
        """

        :param layout:
        :param frame_to_skip:
        :param enable_render:
        :param state_active:
        """
        self.layout = layout
        self.state_active = state_active
        self.enable_render = enable_render
        if enable_render:
            pg.init()
        self.frame_to_skip = frame_to_skip
        self.action_space = spaces.Discrete(5)
        self.maze = Map(layout)
        self.width, self.height = self.maze.get_map_sizes()
        self.game = Game(
            maze=self.maze,
            screen=Controller.get_screen(state_active, self.width, self.height) if enable_render else None,
            sounds_active=False,
            state_active=state_active
        )
        self.timer = 0
        self.reinit_game = False

        self.seed()

    def __del__(self):
        if self.enable_render:
            pg.quit()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.game.restart()
        self.game.player.regenerate()
        self.game.set_mode(GameMode.normal)

    def render(self, mode='human'):
        if mode == 'human':
            self.game.init_screen()
            self.game.draw()
            pg.display.flip()
        elif mode == 'rgb_array':
            return pg.surfarray.array3d(self.game.screen)

    def close(self):
        self.__del__()

    def act(self, action: Action):
        """
        Perform an action on the game. We lockstep frames with actions. If act is not called the game will not run.

        :param action The index of the action we wish to perform

        :returns: Returns the reward that the agent has accumulated while performing the action.

        """
        return sum(self._one_step_action(action) for _ in range(self.frame_to_skip))

    def step(self, action: Union[Action, int]):
        action = Action(action) if type(action) is int else action
        rewards = self.act(action)
        done = self.get_mode() == GameMode.game_over or self.get_mode() == GameMode.black_screen
        obs = self.get_state_matrix()
        info = {
            'win': self.get_mode() == GameMode.black_screen,
            'player position': self.get_player_position(),
            'player pixel position': self.get_player_pixel_position(),
            'player lives': self.game.player.lives,
            'game mode': self.get_mode().value,
            'game score': self.game.score
        }
        return obs, rewards, done, info

    def _one_step_action(self, action: Union[Action, int]):
        self.check_game_mode()

        if self.get_mode() is GameMode.game_over:
            return 0
        elif self.get_mode() is GameMode.black_screen:
            return 0
        elif self.reinit_game:
            self.reinit_game = False
            return 0

        prev_reward = self.game.total_rewards

        self.game.player.change_player_vel(action, self.game)
        self.game.move_players()

        succ_reward = self.game.total_rewards

        return succ_reward - prev_reward

    def get_mode(self) -> GameMode:
        return self.game.game_mode

    def get_state_matrix(self) -> np.ndarray:
        return self.maze.state_matrix

    def get_player_position(self) -> Tuple[int, int]:
        """

        :return: a tuple containing respectively the x and y coordinate
                int the grid
        """
        return self.game.player.nearest_col, self.game.player.nearest_row

    def get_player_pixel_position(self) -> Tuple[int, int]:
        """

        :return: a tuple containing respectively the x and y pixel position in the
                game
        """
        return self.game.player.x, self.game.player.y

    def check_game_mode(self):
        mode = self.get_mode()

        if mode is GameMode.hit_ghost:
            self.game.player.lives -= 1
            if self.game.player.lives == -1:
                self.game.set_mode(GameMode.game_over)
            else:
                self.game.init_players_in_map()
                self.game.make_ghosts_normal()
                self.game.set_mode(GameMode.normal)
                self.reinit_game = True
        elif mode == GameMode.wait_after_eating_ghost:

            self.game.move_ghosts()

            if self.maze.get_number_of_pellets() == 0:
                self.game.set_mode(GameMode.black_screen)
            elif self.game.are_all_ghosts_vulnerable():
                self.game.set_mode(GameMode.change_ghosts)
            elif self.game.are_all_ghosts_normal():
                self.game.set_mode(GameMode.normal)

        self.game.check_ghosts_state()
