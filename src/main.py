import pygame as pg
from .controller import Controller


def main(args):
    pg.init()
    controller = Controller(layout_name=args.layout[0], act_sound=args.sound)
    controller.load_menu()
