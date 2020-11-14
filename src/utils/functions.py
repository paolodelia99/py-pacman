from typing import Dict, Union, Any

import numpy as np
import pygame as pg


def get_image_surface(file_path: str) -> pg.Surface:
    return pg.image.load(file_path).convert()


def check_if_hit(x1: int, y1: int, x2: int, y2: int, threshold: int) -> bool:
    if (x1 - x2 < threshold) and (x1 - x2 > -threshold) \
            and (y1 - y2 < threshold) and (y1 - y2 > -threshold):
        return True
    else:
        return False


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def get_neighbors(matrix: np.ndarray, row: int, col: int) -> Dict[str, Union[int, Any]]:
    neighbors = {
        'L': None,
        'R': None,
        'U': None,
        'D': None
    }

    if row == 0:
        if col == 0:
            neighbors['D'] = matrix[1][0]
            neighbors['R'] = matrix[0][1]
        elif col == matrix.shape[1] - 1:
            neighbors['D'] = matrix[row + 1][col]
            neighbors['L'] = matrix[row][col - 1]
        else:
            neighbors['D'] = matrix[row + 1][col]
            neighbors['L'] = matrix[row][col - 1]
            neighbors['R'] = matrix[row][col + 1]
    elif row == matrix.shape[0] - 1:
        if col == 0:
            neighbors['U'] = matrix[row - 1][col]
            neighbors['R'] = matrix[row][col + 1]
        elif col == matrix.shape[1] - 1:
            neighbors['U'] = matrix[row - 1][col]
            neighbors['L'] = matrix[row][col - 1]
        else:
            neighbors['U'] = matrix[row - 1][col]
            neighbors['L'] = matrix[row][col - 1]
            neighbors['R'] = matrix[row][col + 1]
    else:
        neighbors['D'] = matrix[row + 1][col]
        neighbors['U'] = matrix[row - 1][col]
        neighbors['L'] = matrix[row][col - 1]
        neighbors['R'] = matrix[row][col + 1]

    return neighbors
