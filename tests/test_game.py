import pygame as pg
from src.game import Game


def test_load_map():
    g = Game(
        screen=pg.display.set_mode((600, 400)),
        layout_name='classic-layout'
    )
    assert g.maze.map_matrix.shape == (22, 19)
