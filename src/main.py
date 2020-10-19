import pygame as pg
from .controller import Controller


def main():
    pg.init()
    controller = Controller()
    controller.load_menu()