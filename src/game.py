import os
import pygame as pg

from src.pacman import Pacman
from .ghost import Ghost
from .constants import GHOST_COLORS
from .map import Map
from .utils.path_finder import PathFinder


class Game(object):

    def __init__(self, screen, layout_name):
        self.screen = screen
        self.layout_name = layout_name
        self.lvl_width = 0
        self.lvl_height = 0
        self.map = {}
        self.pellets = 0

        self.edge_light_color = (0, 0, 0, 255)
        self.edge_shadow_color = None
        self.fill_color = None
        self.pellet_color = None
        self.fruitType = None

        self.player = Pacman()
        self.ghosts = [Ghost(i, GHOST_COLORS[i]) for i in range(0, 4)]
        self.path_finder = PathFinder()

        self.tile_id_name = {}
        self.tile_id = {}
        self.tile_id_image = {}

    def load_map(self):
        map_ = Map(self.layout_name)

    def load_assets(self):
        pass

    def init_game_attributes(self):
        pass

    def init_players_in_map(self):
        pass

    def start_game(self):
        self.load_assets()
        self.load_map()
        self.game_loop()

    def game_loop(self):
        pass

    def get_cross_ref(self):
        pass

    def is_wall(self, row, col):
        pass
