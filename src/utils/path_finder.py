from typing import Tuple

import numpy as np


class NotValidPointException(Exception):

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        print(f"the point ({self.x}, {self.y}) is not valid")


class PathFinder(object):

    def __init__(self, state_map: np.ndarray):
        self.state_map = state_map
        self.map_shape = state_map.shape

    def get_min_path(self, x1: int, y1: int, x2: int, y2: int):
        self.check_point_validity(x1, y1)
        self.check_point_validity(x2, y2)
        # fixme: a* algo to implement

        open_set = set()
        closed_set = set()
        current: Tuple[int, int] = (x1, y1)

        open_set.add(current)

        while open_set:
            current = []

    def check_point_validity(self, x: int, y: int):
        if not (0 < x < self.map_shape[1] and 0 < y < self.map_shape[1]):
            raise NotValidPointException(x, y)
