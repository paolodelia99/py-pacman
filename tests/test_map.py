from src.map import Map
from src.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from numpy import int64
import pygame as pg

screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


def test_map_init():
    maze = Map('classic')
    assert maze.map_matrix.shape == (22, 19)
    assert maze.map_matrix.min() == 10
    assert maze.map_matrix.max() == 40
    assert maze.map_matrix.dtype == int64
    assert maze.map_matrix[16][9] == 40


def test_state_map():
    maze = Map('classic')
    assert maze.state_matrix.shape == (22, 19)
    assert maze.state_matrix.min() == -1
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
