import os
import sys
from typing import Dict

import pygame as pg
from pygame.mixer import SoundType

from src.constants import TILE_SIZE
from src.map import Map
from src.utils.functions import check_if_hit
from src.utils.game_mode import GameMode
from src.utils.ghost_state import GhostState


class Pacman(object):
    snd_power_pellet: SoundType
    snd_eat_fruit: SoundType
    snd_eat_gh: SoundType
    snd_pellet: Dict[int, SoundType]

    def __init__(self, sounds_active: bool):
        self.lives = 3
        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 3
        self.sounds_active = sounds_active

        self.nearest_row = 0
        self.nearest_col = 0

        self.homeX = 0
        self.homeY = 0

        self.anim_frame = 1
        self.anim_l = {}
        self.anim_r = {}
        self.anim_u = {}
        self.anim_d = {}
        self.anim_s = {}
        self.load_frames()
        self.current_anim = self.anim_s

        self.pellet_snd_num = 0
        self.load_sounds()

    def load_frames(self):
        for i in range(1, 9):
            self.anim_l[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-l " + str(i) + ".gif")).convert()
            self.anim_r[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-r " + str(i) + ".gif")).convert()
            self.anim_u[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-u " + str(i) + ".gif")).convert()
            self.anim_d[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-d " + str(i) + ".gif")).convert()
            self.anim_s[i] = pg.image.load(os.path.join("res", "sprite", "pacman.gif")).convert()

    def load_sounds(self):
        if self.sounds_active:
            self.snd_pellet = {
                0: pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "pellet1.wav")),
                1: pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "pellet2.wav"))
            }
            self.snd_eat_gh = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "eatgh2.wav"))
            self.snd_eat_fruit = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "eatfruit.wav"))
            self.snd_power_pellet = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "powerpellet.wav"))

    def init_home(self, home_x: int, home_y: int):
        self.x = home_x * TILE_SIZE
        self.y = home_y * TILE_SIZE

    def move(self, maze: Map, game):
        self.nearest_row = int(((self.y + TILE_SIZE / 2) / TILE_SIZE))
        self.nearest_col = int(((self.x + TILE_SIZE / 2) / TILE_SIZE))
        poss_x, poss_y = self.x + self.vel_x, self.y + self.vel_y

        if not self.check_hit_wall(poss_x, poss_y, maze):
            self.x += self.vel_x
            self.y += self.vel_y

            self.check_if_hit_something(maze, game)
            self.check_collision_with_ghosts(game)
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

    def check_hit_wall(self, x: int, y: int, maze: Map) -> bool:
        num_collision = 0

        for row in range(self.nearest_row - 1, self.nearest_row + 2):
            for col in range(self.nearest_col - 1, self.nearest_col + 2):
                if ((x - (col * TILE_SIZE) < TILE_SIZE) and
                        (x - (col * TILE_SIZE) > -TILE_SIZE) and
                        (y - (row * TILE_SIZE) > -TILE_SIZE) and
                        (y - (row * TILE_SIZE) < TILE_SIZE)):
                    try:
                        if maze.is_wall(row, col):
                            num_collision += 1
                    except Exception as e:
                        print(e)

        return num_collision > 0

    def check_keyboard_inputs(self, maze: Map):
        if pg.key.get_pressed()[pg.K_LEFT]:
            if not (self.vel_x == -self.speed and self.vel_y == 0) \
                    and not self.check_hit_wall(
                    self.x - self.speed,
                    self.y,
                    maze):
                self.vel_x = -self.speed
                self.vel_y = 0
        elif pg.key.get_pressed()[pg.K_RIGHT]:
            if not (self.vel_x == self.speed and self.vel_y == 0) \
                    and not self.check_hit_wall(
                    self.x + self.speed,
                    self.y,
                    maze):
                self.vel_x = self.speed
                self.vel_y = 0
        elif pg.key.get_pressed()[pg.K_UP]:
            if not (self.vel_y == -self.speed and self.vel_x == 0) \
                    and not self.check_hit_wall(
                    self.x,
                    self.y - self.speed,
                    maze):
                self.vel_y = -self.speed
                self.vel_x = 0
        elif pg.key.get_pressed()[pg.K_DOWN]:
            if not (self.vel_y == +self.speed and self.vel_x == 0) \
                    and not self.check_hit_wall(
                    self.x,
                    self.y + self.speed,
                    maze):
                self.vel_y = self.speed
                self.vel_x = 0

    def check_if_hit_something(self, maze: Map, game):
        for row in range(self.nearest_row - 1, self.nearest_row + 2):
            for col in range(self.nearest_col - 1, self.nearest_col + 2):
                if ((self.x - (col * TILE_SIZE) < TILE_SIZE) and
                        (self.x - (col * TILE_SIZE) > -TILE_SIZE) and
                        (self.y - (row * TILE_SIZE) > -TILE_SIZE) and
                        (self.y - (row * TILE_SIZE) < TILE_SIZE)):

                    if maze.map_matrix[row][col] == 14:
                        # got a pellet
                        maze.remove_biscuit(row, col)
                        if self.sounds_active:
                            self.snd_pellet[self.pellet_snd_num].play()
                        self.pellet_snd_num = 1 - self.pellet_snd_num
                        game.add_score(10)

                        if maze.get_number_of_pellets() == 0:
                            game.set_mode(6)
                    elif maze.map_matrix[row][col] == 15:
                        # got a power pellet
                        game.set_mode(9)
                        maze.remove_biscuit(row, col)
                        if self.sounds_active:
                            self.snd_power_pellet.play()
                        game.add_score(100)
                        game.make_ghosts_vulnerable()
                    elif maze.map_matrix[row][col] == 11:
                        # ran into a horizontal door
                        for i in range(maze.shape[1]):
                            if not i == col:
                                if maze.map_matrix[row][i] == 11:
                                    self.x = i * TILE_SIZE
                                    if self.vel_x > 0:
                                        self.x += TILE_SIZE
                                    else:
                                        self.x -= TILE_SIZE
                    elif maze.map_matrix[row][col] == 12:
                        # ran into a vertical door
                        for i in range(maze.shape[0]):
                            if not i == row:
                                if maze.map_matrix[i][col] == 12:
                                    self.y = i * TILE_SIZE
                                    if self.vel_y > 0:
                                        self.y += TILE_SIZE
                                    else:
                                        self.y -= TILE_SIZE

    def check_collision_with_ghosts(self, game):
        for ghost in game.ghosts:
            if check_if_hit(self.x, self.y, ghost.x, ghost.y, TILE_SIZE // 2):
                if ghost.state == GhostState.normal:
                    game.set_mode(GameMode.hit_ghost)
                elif ghost.state == GhostState.vulnerable:
                    game.add_score(ghost.value)
                    game.draw_ghost_value(ghost.value)
                    game.duplicate_vulnerable_ghosts_value()
                    if self.sounds_active:
                        self.snd_eat_gh.play()

                    ghost.set_spectacles()
                    game.set_mode(GameMode.wait_after_eating_ghost)

    def set_start_anim(self):
        self.current_anim = self.anim_s
        self.anim_frame = 3

    def set_vel_to_zero(self):
        self.vel_x = 0
        self.vel_y = 0
