import pygame as pg
import os

from src.constants import TILE_SIZE
from src.map import Map
from src.utils.game_mode import GameMode


class Pacman(object):

    def __init__(self):
        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 2

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
        self.snd_eatgh = None
        self.snd_eatfruit = None
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
        pass

    def init_position(self):
        self.x = 9 * TILE_SIZE
        self.y = 16 * TILE_SIZE

    def move(self, map: Map):
        self.nearest_row = int(((self.y + TILE_SIZE / 2) / TILE_SIZE))
        self.nearest_col = int(((self.x + TILE_SIZE / 2) / TILE_SIZE))
        poss_x, poss_y = self.x + self.vel_x, self.y + self.vel_y

        if not self.check_hit_wall(poss_x, poss_y, map):
            self.x += self.vel_x
            self.y += self.vel_y
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

        if game_mode == GameMode.normal:
            if not self.vel_x == 0 or not self.vel_y == 0:
                self.anim_frame += 1

            if self.anim_frame == 9:
                self.anim_frame = 1

    def check_hit_wall(self, x: int, y: int, map_: Map) -> bool:
        num_collision = 0

        for row in range(self.nearest_row - 1, self.nearest_row + 2):
            for col in range(self.nearest_col - 1, self.nearest_col + 2):
                if ((x - (col * TILE_SIZE) < TILE_SIZE) and
                        (x - (col * TILE_SIZE) > -TILE_SIZE) and
                        (y - (row * TILE_SIZE) > -TILE_SIZE) and
                        (y - (row * TILE_SIZE) < TILE_SIZE)):
                    try:
                        if map_.is_wall(row, col):
                            num_collision += 1
                    except Exception as e:
                        print(e)

        print(num_collision > 0)
        return num_collision > 0
