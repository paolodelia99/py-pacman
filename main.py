"""
This is an attempt to recreate the game of Pacman
"""

import sys
import pygame as pg
from src.main import main
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the Pacman Game')
    parser.add_argument('-lay', '--layout', type=str, nargs=1,
                        help="Name of layout to load in the game")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
    pg.quit()
    sys.exit()
