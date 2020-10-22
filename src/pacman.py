import pygame as pg
import os

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

        self.anim_pacmanL = {}
        self.anim_pacmanR = {}
        self.anim_pacmanU = {}
        self.anim_pacmanD = {}
        self.anim_pacmanS = {}
        self.anim_pacmanCurrent = {}
        self.load_frames()

        self.pellet_snd_num = 0
        self.snd_eatgh = None
        self.snd_eatfruit = None
        self.load_sounds()

    def load_frames(self):
        for i in range(1, 9):
            self.anim_pacmanL[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-l " + str(i) + ".gif")).convert()
            self.anim_pacmanR[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-r " + str(i) + ".gif")).convert()
            self.anim_pacmanU[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-u " + str(i) + ".gif")).convert()
            self.anim_pacmanD[i] = pg.image.load(
                os.path.join("res", "sprite", "pacman-d " + str(i) + ".gif")).convert()
            self.anim_pacmanS[i] = pg.image.load(os.path.join("res", "sprite", "pacman.gif")).convert()

    def load_sounds(self):
        pass

    def draw(self, screen, game):

        if game.mode == GameMode.game_over:
            return False

        # set the current frame array to match the direction pacman is facing
        if self.vel_x > 0:
            self.anim_pacmanCurrent = self.anim_pacmanR
        elif self.vel_x < 0:
            self.anim_pacmanCurrent = self.anim_pacmanL
        elif self.vel_y > 0:
            self.anim_pacmanCurrent = self.anim_pacmanD
        elif self.vel_y < 0:
            self.anim_pacmanCurrent = self.anim_pacmanU

        screen.blit(self.anim_pacmanCurrent[self.anim_fram],
                    (self.x - game.screen_pixel_pos[0], self.y - game.screen_pixel_pos[1]))

        if game.mode == GameMode.normal:
            if not self.vel_x == 0 or not self.vel_y == 0:
                self.anim_fram += 1

            if self.anim_fram == 9:
                self.anim_fram = 1
