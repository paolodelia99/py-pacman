from src.map import Map
from src.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from numpy import int64
import pygame as pg

screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


def test_map_init():
    maze = Map('../res/layouts/test.lay', screen)
    assert maze.map_matrix.shape == (22, 19)
    assert maze.map_matrix.min() == 10
    assert maze.map_matrix.max() == 40
    assert maze.map_matrix.dtype == int64
    assert maze.map_matrix[16][9] == 40


def test_state_map():
    maze = Map('../res/layouts/test.lay', screen)
    assert maze.state_matrix.shape == (22, 19)
    assert maze.state_matrix.min() == -1
    assert maze.state_matrix.max() == 2
    assert maze.state_matrix.dtype == int64


def test_tile_map():
    map_ = Map('../res/layouts/test.lay', screen)
    assert type(map_.tile_map) == dict
    assert map_.tile_map.__len__() == 418


def test_get_neighbors():
    maze = Map('../res/layouts/classic.lay', screen)
    assert maze.get_neighbors(0, 0, 0) == 2
    assert maze.get_neighbors(1, 9, 0) == 4
    assert maze.get_neighbors(1, 1, 1) == 1
