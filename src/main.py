import pygame as pg
from .controller import Controller


def main(args):
    pg.init()
    controller = Controller(layout_name=args.layout[0])
    controller.load_menu()