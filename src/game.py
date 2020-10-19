import os
import pygame as pg

from src.pacman import Pacman
from .ghost import Ghost
from .constants import GHOST_COLORS
from .utils.path_finder import PathFinder


class Game(object):

    def __init__(self, screen):
        self.screen = screen
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

    def load_level(self):
        f = open("../res/levels/0.txt", 'r')
        line_num = -1
        row_num = 0
        is_reading_level_data = False

        for line in f:

            line_num += 1

            while len(line) > 0 and (line[-1] == "\n" or line[-1] == "\r"): line = line[:-1]
            while len(line) > 0 and (line[0] == "\n" or line[0] == "\r"): line = line[1:]
            str_split_by_space = line.split(' ')
            j = str_split_by_space[0]

            if j == "'" or j == "":
                use_line = False
            elif j == "#":
                use_line = False

                first_word = str_split_by_space[1]

                if first_word == "lvl_width":
                    self.lvl_width = int(str_split_by_space[2])
                elif first_word == "lvl_height":
                    self.lvl_height = int(str_split_by_space[2])
                elif first_word == "edgecolor":
                    # edge color keyword for backwards compatibility (single edge color) mazes
                    red = int(str_split_by_space[2])
                    green = int(str_split_by_space[3])
                    blue = int(str_split_by_space[4])
                    self.edge_light_color = (red, green, blue, 255)
                    self.edge_shadow_color = (red, green, blue, 255)
                elif first_word == "edge_ligth_color":
                    red = int(str_split_by_space[2])
                    green = int(str_split_by_space[3])
                    blue = int(str_split_by_space[4])
                    self.edge_light_color = (red, green, blue, 255)
                elif first_word == "edge_shadow_color":
                    red = int(str_split_by_space[2])
                    green = int(str_split_by_space[3])
                    blue = int(str_split_by_space[4])
                    self.edge_shadow_color = (red, green, blue, 255)
                elif first_word == "fill_color":
                    red = int(str_split_by_space[2])
                    green = int(str_split_by_space[3])
                    blue = int(str_split_by_space[4])
                    self.fill_color = (red, green, blue, 255)
                elif first_word == "pellet_color":
                    red = int(str_split_by_space[2])
                    green = int(str_split_by_space[3])
                    blue = int(str_split_by_space[4])
                    self.pellet_color = (red, green, blue, 255)
                elif first_word == "fruittype":
                    self.fruitType = int(str_split_by_space[2])
                elif first_word == "startleveldata":
                    is_reading_level_data = True
                    row_num = 0
                elif first_word == "endleveldata":
                    is_reading_level_data = False

            else:
                use_line = True

            # this is a map data line
            if use_line:
                if is_reading_level_data:
                    for k in range(0, self.lvl_width, 1):
                        self.set_map_tile(row_num, k, int(str_split_by_space[k]))
                        this_id = int(str_split_by_space[k])
                        if this_id == 4:
                            # starting position for pac-man
                            self.player.home_x = k * 16
                            self.player.home_y = row_num * 16
                            self.set_map_tile(row_num, k, 0)
                        elif 10 <= this_id <= 13:
                            # one of the ghosts
                            self.ghosts[this_id - 10].home_x = k * 16
                            self.ghosts[this_id - 10].home_y = row_num * 16
                            self.set_map_tile(row_num, k, 0)
                        elif this_id == 2:
                            # pellet
                            self.pellets += 1

                    row_num += 1

        # reload all tiles and set appropriate colors
        self.get_cross_ref()

        # load map into the pathfinder object
        self.path_finder.resize_map(self.lvl_height, self.lvl_width)

        for row in range(0, self.path_finder.size[0], 1):
            for col in range(0, self.path_finder.size[1], 1):
                if self.is_wall(row, col):
                    self.path_finder.set_type(row, col, 1)
                else:
                    self.path_finder.set_type(row, col, 0)

        # do all the level-starting stuff
        # self.restart(ghosts, path, this_fruit, player, game)

    def load_assets(self):
        pass

    def init_game_attributes(self):
        pass

    def init_players_in_map(self):
        pass

    def start_game(self):
        self.load_assets()
        self.load_level()
        self.game_loop()

    def game_loop(self):
        pass

    def get_cross_ref(self):
        f = open(os.path.join("../res", "crossref.txt"), 'r')

        line_num = 0

        for i in f.readlines():
            while len(i) > 0 and (i[-1] == '\n' or i[-1] == '\r'): i = i[:-1]
            while len(i) > 0 and (i[0] == '\n' or i[0] == '\r'): i = i[1:]
            str_split_by_space = i.split(' ')

            j = str_split_by_space[0]

            if j == "'" or j == "" or j == "#":
                use_line = False
            else:
                use_line = True

            if use_line:
                self.tile_id_name[int(str_split_by_space[0])] = str_split_by_space[1]
                self.tile_id[str_split_by_space[1]] = int(str_split_by_space[0])

                this_id = int(str_split_by_space[0])
                if not this_id in [23]:
                    self.tile_id_image[this_id] = pg.image.load(
                        os.path.join("../res", "tiles", str_split_by_space[1] + ".gif")).convert()
                else:
                    self.tile_id_image[this_id] = pg.Surface((16, 16))

                # change colors in tile_id_image to match maze colors
                for y in range(0, 16, 1):
                    for x in range(0, 16, 1):

                        if self.tile_id_image[this_id].get_at((x, y)) == (255, 206, 255, 255):
                            # wall edge
                            self.tile_id_image[this_id].set_at((x, y), self.edge_light_color)

                        elif self.tile_id_image[this_id].get_at((x, y)) == (132, 0, 132, 255):
                            # wall fill
                            self.tile_id_image[this_id].set_at((x, y), self.fill_color)

                        elif self.tile_id_image[this_id].get_at((x, y)) == (255, 0, 255, 255):
                            # pellet color
                            self.tile_id_image[this_id].set_at((x, y), self.edge_shadow_color)

                        elif self.tile_id_image[this_id].get_at((x, y)) == (128, 0, 128, 255):
                            # pellet color
                            self.tile_id_image[this_id].set_at((x, y), self.pellet_color)

            line_num += 1

    def is_wall(self, row, col):
        if row > self.lvl_height - 1 or row < 0:
            return True

        if col > self.lvl_width - 1 or col < 0:
            return True

            # check the offending tile ID
        result = self.get_map_tile(row, col)

        # if the tile was a wall
        if 100 <= result <= 199:
            return True
        else:
            return False

    def set_map_tile(self, row, col, new_value):
        self.map[(row * self.lvl_width) + col] = new_value

    def get_map_tile(self, row, col):
        if 0 <= row < self.lvl_height and 0 <= col < self.lvl_width:
            return self.map[(row * self.lvl_width) + col]
        else:
            return 0
