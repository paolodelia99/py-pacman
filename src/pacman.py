import os
from typing import Union, Tuple

import pygame as pg

from src.constants import TILE_SIZE, ROOT_DIR
from src.utils.action import Action
from src.utils.game_mode import GameMode


class Pacman(object):

    def __init__(self):
        self._lives = 3
        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 3
        self._current_action = Action.LEFT

        self.nearest_row = 0
        self.nearest_col = 0
        self.home_x = 0
        self.home_y = 0

        self.anim_frame = 1
        self.anim_l = {}
        self.anim_r = {}
        self.anim_u = {}
        self.anim_d = {}
        self.anim_s = {}
        self.current_anim = self.anim_s

        self.pellet_snd_num = 0

    def load_frames(self):
        for i in range(1, 9):
            self.anim_l[i] = pg.image.load(
                os.path.join(ROOT_DIR, "res", "sprite", "pacman-l " + str(i) + ".gif")).convert()
            self.anim_r[i] = pg.image.load(
                os.path.join(ROOT_DIR, "res", "sprite", "pacman-r " + str(i) + ".gif")).convert()
            self.anim_u[i] = pg.image.load(
                os.path.join(ROOT_DIR, "res", "sprite", "pacman-u " + str(i) + ".gif")).convert()
            self.anim_d[i] = pg.image.load(
                os.path.join(ROOT_DIR, "res", "sprite", "pacman-d " + str(i) + ".gif")).convert()
            self.anim_s[i] = pg.image.load(os.path.join(ROOT_DIR, "res", "sprite", "pacman.gif")).convert()

    def init_home(self, home_x: int, home_y: int):
        if self.home_x == 0 and self.home_y == 0:
            self.home_x, self.home_y = home_x, home_y

        self.x = self.home_x * TILE_SIZE
        self.y = self.home_y * TILE_SIZE
        self.nearest_row = self.home_y
        self.nearest_col = self.home_x

    def regenerate(self):
        self.lives = 3
        self.init_home(0, 0)
        self.set_vel_to_zero()

    def get_position(self) -> Tuple[int, int]:
        return self.nearest_col, self.nearest_row

    def get_pixel_pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def move(self, game):
        self.nearest_row = int(((self.y + TILE_SIZE / 2) / TILE_SIZE))
        self.nearest_col = int(((self.x + TILE_SIZE / 2) / TILE_SIZE))
        poss_x, poss_y = self.x + self.vel_x, self.y + self.vel_y

        if not game.check_if_player_hit_wall(poss_x, poss_y):
            self.x += self.vel_x
            self.y += self.vel_y

            game.check_if_hit_something()
            game.check_collision_with_ghosts()
            # todo: check collision with fruit
        else:
            self.vel_y, self.vel_x = 0, 0

    def draw(self, screen, game_mode):

        if game_mode == GameMode.game_over:
            return False

        # set the current frame array to match the direction pacman is facing
        if self.vel_x > 0:
            self.current_anim = self.anim_r
        elif self.vel_x < 0:
            self.current_anim = self.anim_l
        elif self.vel_y > 0:
            self.current_anim = self.anim_d
        elif self.vel_y < 0:
            self.current_anim = self.anim_u

        screen.blit(self.current_anim[self.anim_frame],
                    (self.x, self.y))

        if game_mode in [GameMode.normal, GameMode.change_ghosts, GameMode.wait_after_eating_ghost]:
            if not self.vel_x == 0 or not self.vel_y == 0:
                self.anim_frame += 1

            if self.anim_frame == 9:
                self.anim_frame = 1

    def set_start_anim(self):
        self.current_anim = self.anim_s
        self.anim_frame = 3

    def set_vel_to_zero(self):
        self.vel_x = 0
        self.vel_y = 0

    def change_player_vel(self, action: Union[Action, int], game):
        action = Action(int(action)) if type(action) is int else action
        if action == action.LEFT:
            if not (self.vel_x == -self.speed and self.vel_y == 0) \
                    and not game.check_if_player_hit_wall(
                    self.x - self.speed,
                    self.y):
                self.vel_x = -self.speed
                self.vel_y = 0
        elif action == action.RIGHT:
            if not (self.vel_x == self.speed and self.vel_y == 0) \
                    and not game.check_if_player_hit_wall(
                    self.x + self.speed,
                    self.y):
                self.vel_x = self.speed
                self.vel_y = 0
        elif action == action.UP:
            if not (self.vel_y == -self.speed and self.vel_x == 0) \
                    and not game.check_if_player_hit_wall(
                    self.x,
                    self.y - self.speed):
                self.vel_y = -self.speed
                self.vel_x = 0
        elif action == action.DOWN:
            if not (self.vel_y == +self.speed and self.vel_x == 0) \
                    and not game.check_if_player_hit_wall(
                    self.x,
                    self.y + self.speed):
                self.vel_y = self.speed
                self.vel_x = 0

    @property
    def lives(self):
        return self._lives

    @lives.setter
    def lives(self, lives):
        self._lives = lives

    def get_vel(self) -> Tuple[int, int]:
        return self.vel_x, self.vel_y

    def print_position(self):
        print(f"Pacman col: {self.nearest_col}, row: {self.nearest_row}")

    @property
    def current_action(self) -> Action:
        return self._current_action

    @current_action.setter
    def current_action(self, action: Action):
        self._current_action = action
