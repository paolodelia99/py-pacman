import pygame as pg
import numpy as np
from src.game import Game
from src.map import Map


def test_load_map():
    g = Game(
        screen=pg.display.set_mode((600, 400)),
        layout_name='classic-layout'
    )
    assert g.maze.map_matrix.shape == (22, 19)


def test_get_rba_array():
    maze = Map('classic')
    g = Game(
        maze=maze,
        screen=pg.display.set_mode(maze.get_map_sizes()),
        sounds_active=False,
        state_active=False
    )
    arr = g.get_rba_array()
    assert type(arr) is np.ndarray
    assert arr.shape == (456, 552, 3)