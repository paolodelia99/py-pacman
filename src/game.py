import os
import sys

import pygame as pg

from src.pacman import Pacman
from .constants import GHOST_COLORS
from .ghost import Ghost
from .map import Map
from .utils.functions import get_image_surface
from .utils.game_mode import GameMode
from .utils.path_finder import PathFinder


class Game(object):

    def __init__(self, screen, layout_name):
        self.screen = screen
        self.layout_name = layout_name
        self.layout_path = os.path.join('res', 'layouts', layout_name + '.lay')
        self.map_ = Map(self.layout_path, screen)

        self.is_run = True
        self.is_game_run = False
        self.pause = False
        self.game_mode = GameMode(1)

        self.screen_bg = get_image_surface(os.path.join('res', 'backgrounds', '1.gif'))
        self.fruitType = None

        self.player = Pacman()
        self.ghosts = [Ghost(i, GHOST_COLORS[i]) for i in range(0, 4)]
        self.path_finder = PathFinder()

    def load_assets(self):
        pass

    def init_game_attributes(self):
        pass

    def init_players_in_map(self):
        self.player.init_position()

    def init_screen(self):
        self.screen.blit(self.screen_bg, (0, 0))

    def start_game(self):
        self.load_assets()
        self.game_loop()

    def game_loop(self):
        clock = pg.time.Clock()
        self.init_players_in_map()

        while self.is_run:
            self.init_screen()
            self.event_loop()
            self.draw()

            if self.game_mode == GameMode.normal:
                self.player.move(self.map_)

            pg.display.flip()
            clock.tick(60)

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.is_run = False
                elif event.key == pg.K_LEFT:
                    if not (self.player.vel_x == -self.player.speed and self.player.vel_y == 0) \
                            and not self.player.check_hit_wall(
                                    self.player.x - self.player.speed,
                                    self.player.y,
                                    self.map_):
                        self.player.vel_x = -self.player.speed
                        self.player.vel_y = 0
                elif event.key == pg.K_RIGHT:
                    if not (self.player.vel_x == self.player.speed and self.player.vel_y == 0) \
                            and not self.player.check_hit_wall(
                                    self.player.x + self.player.speed,
                                    self.player.y,
                                    self.map_):
                        self.player.vel_x = self.player.speed
                        self.player.vel_y = 0
                elif event.key == pg.K_UP:
                    if not (self.player.vel_y == -self.player.speed and self.player.vel_x == 0) \
                            and not self.player.check_hit_wall(
                                    self.player.x,
                                    self.player.y - self.player.speed,
                                    self.map_):
                        self.player.vel_y = -self.player.speed
                        self.player.vel_x = 0
                elif event.key == pg.K_DOWN:
                    if not (self.player.vel_y == +self.player.speed and self.player.vel_x == 0) \
                            and not self.player.check_hit_wall(
                                    self.player.x,
                                    self.player.y + self.player.speed,
                                    self.map_):
                        self.player.vel_y = self.player.speed
                        self.player.vel_x = 0

    def draw(self):
        self.map_.draw()
        self.player.draw(self.screen, self.game_mode)
