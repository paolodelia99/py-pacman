"""
This is an attempt to recreate the game of Pacman
"""

import argparse
import sys

import pygame as pg

from src.main import main


def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the Pacman Game')
    parser.add_argument('-lay', '--layout', type=str, nargs=1,
                        help="Name of layout to load in the game")
    parser.add_argument('-snd', '--sound', action='store_true',
                        help="Activate sounds in the game")
    parser.add_argument('-stt', '--state', action='store_true',
                        help="Display the state matrix of the game")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
    pg.quit()
    sys.exit()
