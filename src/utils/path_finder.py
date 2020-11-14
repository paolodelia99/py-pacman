from typing import Set, List, Optional, Tuple

import numpy as np

from .matrix_point import MatrixPoint
from ..constants import INVERT_ORIENTATION_TABLE
from .functions import get_neighbors


class NotValidPointException(Exception):

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self):
        print(f"the point ({self.x}, {self.y}) is not valid")


class NoPathFoundException(Exception):

    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

    def __str__(self):
        print(f"Not path found between {self.start} and {self.goal}")


class PathFinder(object):

    def __init__(self, state_map: np.ndarray):
        self.state_map = state_map
        self.map_shape = state_map.shape

    def get_min_path(self, x1: int, y1: int, x2: int, y2: int) -> Optional[List[str]]:
        self.check_point_validity(x1, y1)
        self.check_point_validity(x2, y2)

        open_set: Set[MatrixPoint] = set()
        closed_set: Set[MatrixPoint] = set()
        start = MatrixPoint(self.state_map[y1][x1], x1, y1)
        current = start.__copy__()
        goal = MatrixPoint(self.state_map[y2][x2], x2, y2)

        open_set.add(current)

        while open_set:
            current = min(open_set, key=lambda item: item.get_cost(goal))

            if current.same_position(goal):
                return PathFinder.return_min_path(current)

            open_set.remove(current)
            closed_set.add(current)

            for k, point in get_neighbors(self.state_map, current.y, current.x).items():
                if point is not None:

                    if point in closed_set:
                        continue

                    if point in open_set:
                        new_g = current.g + 1
                        if point.g > new_g:
                            point.g = new_g
                            point.set_parent(current)
                    else:
                        point.g = current.g + 1
                        point.set_parent(current)
                        open_set.add(point)

        raise NoPathFoundException(start, goal)

    def check_point_validity(self, x: int, y: int):
        if not (0 <= x < self.map_shape[1] and 0 <= y < self.map_shape[0]):
            raise NotValidPointException(x, y)

    @staticmethod
    def return_min_path(m_point: MatrixPoint) -> List[str]:
        path = []
        while m_point.parent:
            path.append(INVERT_ORIENTATION_TABLE[m_point.get_parent_orientation()])
            m_point = m_point.parent
        path.reverse()
        return path

    def get_random_allow_position(self) -> Tuple[int, int]:
        res = np.where(self.state_map == 0)
        rnd = np.random.randint(0, high=len(res[0]))
        return res[1][rnd], res[0][rnd]
