import pygame as pg
from src.game import Game


def test_load_map():
    g = Game(pg.display.set_mode((600, 400)))
    g.load_layout()
