import os
import pygame as pg
import sys

from src.pacman import Pacman
from .constants import GHOST_COLORS
from .ghost import Ghost
from .map import Map
from .utils.path_finder import PathFinder
from .utils.functions import get_image_surface


class Game(object):

    def __init__(self, screen, layout_name):
        self.screen = screen
        self.layout_name = layout_name
        self.layout_path = os.path.join('res', 'layouts', layout_name + '.lay')
        self.map_ = Map(self.layout_path, screen)

        self.is_run = True
        self.is_game_run = False
        self.pause = False

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
        pass

    def init_screen(self):
        self.screen.blit(self.screen_bg, (0, 0))

    def start_game(self):
        self.load_assets()
        self.game_loop()

    def game_loop(self):

        while self.is_run:
            self.init_screen()
            self.event_loop()
            self.map_.draw()

            pg.display.flip()

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.is_run = False
