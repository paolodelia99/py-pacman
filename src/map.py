import os
import sys
from itertools import product

import numpy as np

from .constants import STATE_LOOKUP_TABLE, \
    TILE_LOOKUP_TABLE, \
    TILE_SIZE, \
    IMG_EDGE_LIGHT_COLOR, \
    IMG_EDGE_SHADOW_COLOR, \
    IMG_FILL_COLOR, \
    IMG_PELLET_COLOR
from .utils.functions import get_image_surface


class Map:

    def __init__(self, layout_name, screen):
        self.edge_light_color = (0, 0, 255, 255)
        self.edge_shadow_color = (0, 0, 255, 255)
        self.fill_color = (0, 0, 0, 255)
        self.pellet_color = (255, 255, 255, 255)
        self.layout_name = layout_name
        self.screen = screen
        self.map_matrix = np.loadtxt(self.layout_name).astype(int)
        self.shape = self.map_matrix.shape
        self.state_matrix = self.build_state_matrix()
        self.tile_map = self.build_tile_map()

    def is_wall(self, row: int, col: int) -> bool:
        if row > self.shape[0] or row < 0:
            return True

        if col > self.shape[1] or col < 0:
            return True

        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
            return self.state_matrix[row][col] == 2
        else:
            return False

    def is_ghost(self, x: int, y: int) -> bool:
        return self.state_matrix[x:int, y:int] == -1

    def is_biscuit(self, x: int, y: int) -> bool:
        return self.state_matrix[x:int, y:int] == 1

    def is_pill(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 15

    def remove_biscuit(self, x: int, y: int):
        self.map_matrix[x][y] = 10
        self.state_matrix[x][y] = 0
        self.tile_map[x, y] = get_image_surface(os.path.join(
                        sys.path[0],
                        "res",
                        "tiles",
                        TILE_LOOKUP_TABLE[self.map_matrix[x][y]]
                    ))

    def get_player_home(self) -> tuple:
        home_y, home_x = np.where(self.map_matrix == 40)
        return int(home_x[0]), int(home_y[0])

    def update_ghosts_pos(self):
        pass

    def draw(self):
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                self.screen.blit(self.tile_map[row, col], (col * TILE_SIZE, row * TILE_SIZE))

    def build_tile_map(self):
        tile_map = {}

        for i in range(self.map_matrix.shape[0]):
            for j in range(self.map_matrix.shape[1]):
                if self.map_matrix[i][j] in [33, 40, 11, 12]:
                    # position of pacman or the ghost
                    tile_map[i, j] = get_image_surface(os.path.join(sys.path[0], "res", "tiles", TILE_LOOKUP_TABLE[10]))
                else:
                    tile_map[i, j] = get_image_surface(os.path.join(
                        sys.path[0],
                        "res",
                        "tiles",
                        TILE_LOOKUP_TABLE[self.map_matrix[i][j]]
                    ))
                self.recolor_tile(tile_map[i, j])

        return tile_map

    def build_state_matrix(self) -> np.ndarray:
        state_matrix = np.ndarray(shape=(self.map_matrix.shape[0], self.map_matrix.shape[1])).astype(np.int)

        for i in range(self.map_matrix.shape[0]):
            for j in range(self.map_matrix.shape[1]):
                state_matrix[i][j] = STATE_LOOKUP_TABLE[self.map_matrix[i][j]]

        return state_matrix

    def get_neighbors(self, r: int, c: int, el: int) -> int:
        def get(row, col):
            return 0 <= row < self.map_matrix.shape[0] and 0 <= col < self.map_matrix.shape[1] and self.map_matrix[
                row, col] == el

        neighbors_list = [get(i, j) for i, j in product(range(r - 1, r + 2), range(c - 1, c + 2))]

        return sum(map(bool, neighbors_list)) - 1

    def get_number_of_pellets(self) -> int:
        unique, counts = np.unique(self.map_matrix, return_counts=True)
        return dict(zip(unique, counts))[10]

    def recolor_tile(self, tile):
        for y in range(0, TILE_SIZE, 1):
            for x in range(0, TILE_SIZE, 1):

                if tile.get_at((x, y)) == IMG_EDGE_LIGHT_COLOR:
                    # wall edge
                    tile.set_at((x, y), self.edge_light_color)

                elif tile.get_at((x, y)) == IMG_FILL_COLOR:
                    # wall fill
                    tile.set_at((x, y), self.fill_color)

                elif tile.get_at((x, y)) == IMG_EDGE_SHADOW_COLOR:
                    # pellet color
                    tile.set_at((x, y), self.edge_shadow_color)

                elif tile.get_at((x, y)) == IMG_PELLET_COLOR:
                    # pellet color
                    tile.set_at((x, y), self.pellet_color)
