from typing import Tuple

from src.ghost import Ghost
from src.map import Map
from numpy import int64
import numpy as np


def test_map_init():
    maze = Map('classic')
    assert maze.map_matrix.shape == (22, 19)
    assert maze.map_matrix.min() == 10
    assert maze.map_matrix.max() == 50
    assert maze.map_matrix.dtype == int64
    assert maze.map_matrix[16][9] == 40


def test_state_map():
    maze = Map('classic')
    assert maze.state_matrix.shape == (22, 19)
    assert maze.state_matrix.min() == -10
    assert maze.state_matrix.max() == 2
    assert maze.state_matrix.dtype == int64


def test_tile_map():
    maze = Map('classic')
    maze.build_tile_map()
    assert type(maze.tile_map) == dict
    assert maze.tile_map.__len__() == 418


def test_get_player_init_position():
    maze = Map('classic')
    assert maze.get_player_home() == (9, 16)


def test_get_number_of_pellets():
    maze = Map('classic')
    assert maze.get_number_of_pellets() == 181


def test_update_ghosts_positions():
    maze = Map('classic')
    ghosts = [Ghost(i, (255, 0, 0, 255)) for i in range(4)]

    def get_random_allow_position() -> Tuple[int, int]:
        res = np.where(maze.state_matrix == 1)
        rnd = np.random.randint(0, high=len(res[0]))
        return res[1][rnd], res[0][rnd]

    rand_pos = []

    for ghost in ghosts:
        x, y = get_random_allow_position()
        rand_pos.append((x, y))
        ghost.nearest_col = x
        ghost.nearest_row = y

    maze.update_ghosts_position(ghosts)

    for x, y in rand_pos:
        assert maze.state_matrix[y][x] == -1
