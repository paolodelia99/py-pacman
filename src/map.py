import numpy as np
import os
from itertools import product
import pygame as pg


def get_image_surface(file_path):
    return pg.image.load(file_path).convert()


class Map:

    def __init__(self, layout_name):
        self.layout_name = layout_name
        self.map_matrix = np.loadtxt(self.layout_name).astype(int)
        self.tile_map = self.build_tile_map()

    def is_wall(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 0

    def is_ghost(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 3

    def is_biscuit(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 1

    def is_pill(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 4

    def remove_biscuit_pill(self, x: int, y: int):
        self.map_matrix[x:int, y:int] = 2

    def update_ghosts_pos(self):
        pass

    def draw(self):
        pass

    def build_tile_map(self):
        tile_map = {}

        for i in range(self.map_matrix.shape[0]):
            for j in range(self.map_matrix.shape[1]):
                pass

        return tile_map

    def get_neighbors(self, r: int, c: int, el: int) -> int:
        def get(row, col):
            return 0 <= row < self.map_matrix.shape[0] and 0 <= col < self.map_matrix.shape[1] and self.map_matrix[
                row, col] == el

        neighbors_list = [get(i, j) for i, j in product(range(r - 1, r + 2), range(c - 1, c + 2))]

        return sum(map(bool, neighbors_list)) - 1
