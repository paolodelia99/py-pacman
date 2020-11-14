from typing import Dict, Union, Any

import numpy as np
import pygame as pg
from .matrix_point import MatrixPoint


def get_image_surface(file_path: str) -> pg.Surface:
    return pg.image.load(file_path).convert()


def check_if_hit(x1: int, y1: int, x2: int, y2: int, threshold: int) -> bool:
    if (x1 - x2 < threshold) and (x1 - x2 > -threshold) \
            and (y1 - y2 < threshold) and (y1 - y2 > -threshold):
        return True
    else:
        return False


def get_neighbors(matrix: np.ndarray, row: int, col: int) -> Dict[str, Union[MatrixPoint, Any]]:
    """

    :rtype: MatrixPoint
    :param matrix:
    :param row:
    :param col:
    :return: a dict containing all the neighbors point
    """
    neighbors = {
        'L': None,
        'R': None,
        'U': None,
        'D': None
    }

    if row == 0:
        if col == 0:
            neighbors['D'] = MatrixPoint(matrix[1][0], 0, 1)
            neighbors['R'] = MatrixPoint(matrix[0][1], 1, 0)
        elif col == matrix.shape[1] - 1:
            neighbors['D'] = MatrixPoint(matrix[row + 1][col], col, row + 1)
            neighbors['L'] = MatrixPoint(matrix[row][col - 1], col - 1, row)
        else:
            neighbors['D'] = MatrixPoint(matrix[row + 1][col], col, row + 1)
            neighbors['L'] = MatrixPoint(matrix[row][col - 1], col - 1, row)
            neighbors['R'] = MatrixPoint(matrix[row][col + 1], col + 1, row)
    elif row == matrix.shape[0] - 1:
        if col == 0:
            neighbors['U'] = MatrixPoint(matrix[row - 1][col], col, row - 1)
            neighbors['R'] = MatrixPoint(matrix[row][col + 1], col + 1, row)
        elif col == matrix.shape[1] - 1:
            neighbors['U'] = MatrixPoint(matrix[row - 1][col], col, row - 1)
            neighbors['L'] = MatrixPoint(matrix[row][col - 1], col - 1, row)
        else:
            neighbors['U'] = MatrixPoint(matrix[row - 1][col], col, row - 1)
            neighbors['L'] = MatrixPoint(matrix[row][col - 1], col - 1, row)
            neighbors['R'] = MatrixPoint(matrix[row][col + 1], col + 1, row)
    else:
        if col == 0:
            neighbors['D'] = MatrixPoint(matrix[row + 1][col], col, row + 1)
            neighbors['U'] = MatrixPoint(matrix[row - 1][col], col, row - 1)
            neighbors['R'] = MatrixPoint(matrix[row][col + 1], col + 1, row)
        elif col == matrix.shape[1] - 1:
            neighbors['D'] = MatrixPoint(matrix[row + 1][col], col, row + 1)
            neighbors['U'] = MatrixPoint(matrix[row - 1][col], col, row - 1)
            neighbors['L'] = MatrixPoint(matrix[row][col - 1], col - 1, row)
        else:
            neighbors['D'] = MatrixPoint(matrix[row + 1][col], col, row + 1)
            neighbors['U'] = MatrixPoint(matrix[row - 1][col], col, row - 1)
            neighbors['L'] = MatrixPoint(matrix[row][col - 1], col - 1, row)
            neighbors['R'] = MatrixPoint(matrix[row][col + 1], col + 1, row)

    return neighbors
