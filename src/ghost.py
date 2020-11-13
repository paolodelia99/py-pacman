import os
import sys

import pygame as pg
from pygame.surface import SurfaceType

from src.constants import TILE_SIZE, VULNERABLE_GHOST_COLOR, WHITE_GHOST_COLOR
from src.pacman import Pacman
from src.utils.functions import get_image_surface
from src.utils.game_mode import GameMode
from src.utils.ghost_state import GhostState


class Ghost(object):
    img_glasses: SurfaceType

    def __init__(self, ghost_id, ghost_color):
        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 1
        self.nearest_row = 0
        self.nearest_col = 0
        self.id = ghost_id
        self.state = GhostState.normal
        self.value = 0

        self.home_x = 0
        self.home_y = 0

        self.current_path = ""
        self.ghost_color = ghost_color

        self.anim = Ghost.load_ghost_animation(self.ghost_color)

        self.anim_fram = 1
        self.anim_delay = 0
        self.load_assets()

    @staticmethod
    def load_ghost_animation(color):
        anim = {}

        for i in range(1, 7):
            anim[i] = pg.image.load(
                os.path.join("res", "sprite", "ghost " + str(i) + ".gif")).convert()

            # change the ghost color in this frame
            for y in range(0, TILE_SIZE):
                for x in range(0, TILE_SIZE):

                    if anim[i].get_at((x, y)) == (255, 0, 0, 255):
                        # default, red ghost body color
                        anim[i].set_at((x, y), color)

        return anim

    def load_assets(self):
        self.img_glasses = get_image_surface(os.path.join(sys.path[0], "res", "tiles", "glasses.gif"))

    def draw(self, screen, game, player: Pacman):

        pupil_set = None

        if game.game_mode == GameMode.game_over:
            return False

        # ghost eyes --
        for y in range(6, 12):
            for x in [5, 6, 8, 9]:
                self.anim[self.anim_fram].set_at((x, y), (248, 248, 248, 255))
                self.anim[self.anim_fram].set_at((x + 9, y), (248, 248, 248, 255))

                if player.x > self.x and player.y > self.y:
                    # player is to lower-right
                    pupil_set = (8, 9)
                elif player.x < self.x and player.y > self.y:
                    # player is to lower-left
                    pupil_set = (5, 9)
                elif player.x > self.x and player.y < self.y:
                    # player is to upper-right
                    pupil_set = (8, 6)
                elif player.x < self.x and player.y < self.y:
                    # player is to upper-left
                    pupil_set = (5, 6)
                else:
                    pupil_set = (5, 9)

        for y in range(pupil_set[1], pupil_set[1] + 3):
            for x in range(pupil_set[0], pupil_set[0] + 2, 1):
                self.anim[self.anim_fram].set_at((x, y), (0, 0, 255, 255))
                self.anim[self.anim_fram].set_at((x + 9, y), (0, 0, 255, 255))

        if self.state == GhostState.normal:
            # draw regular ghost (this one)
            screen.blit(self.anim[self.anim_fram],
                        (self.x, self.y))
        elif self.state == GhostState.vulnerable:
            if game.are_ghosts_vulnerable() and game.mode_timer < 260:
                # blue
                screen.blit(Ghost.load_ghost_animation(VULNERABLE_GHOST_COLOR)[self.anim_fram],
                            (self.x, self.y))
            else:
                # blue/white flashing
                temp_timer_i = int((360 - game.mode_timer) / 10)
                if temp_timer_i == 1 or temp_timer_i == 3 or temp_timer_i == 5 or temp_timer_i == 7 or temp_timer_i == 9:
                    screen.blit(Ghost.load_ghost_animation(WHITE_GHOST_COLOR)[self.anim_fram],
                                (self.x, self.y))
                else:
                    screen.blit(Ghost.load_ghost_animation(VULNERABLE_GHOST_COLOR)[self.anim_fram],
                                (self.x, self.y))

        elif self.state == GhostState.spectacles:
            # draw glasses
            screen.blit(self.img_glasses,
                        (self.x, self.y))

        if game.game_mode == GameMode.wait_after_finishing_level or game.game_mode == GameMode.flash_maze:
            # don't animate ghost if the level is complete
            return False

        self.anim_delay += 1

        if self.anim_delay == 2:
            self.anim_fram += 1

            if self.anim_fram == 7:
                # wrap to beginning
                self.anim_fram = 1

            self.anim_delay = 0

    def set_normal(self):
        if self.state == GhostState.vulnerable:
            self.state = GhostState.normal
            self.value = 0

    def duplicate_value(self):
        self.value *= 2

    def init_home(self, home_x: int, home_y: int):
        self.home_x = home_x * TILE_SIZE
        self.home_y = home_y * TILE_SIZE
        self.x = self.home_x
        self.y = self.home_y

    def set_vulnerable(self):
        self.state = GhostState.vulnerable
        self.value = 200

    def set_spectacles(self):
        self.state = GhostState.spectacles
        self.value = 0
        self.speed *= 4
        self.x = self.nearest_col * TILE_SIZE
        self.y = self.nearest_row * TILE_SIZE
        # fixme: find path to home
