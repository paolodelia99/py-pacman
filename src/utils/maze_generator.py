import os
import sys

import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the maze generator')
    parser.add_argument('-ln', '--layout_name', type=str, nargs=1,
                        help="Name of the new maze layout to create")
    parser.add_argument('-w', '--width', type=int, nargs=1,
                        help='Width of the maze')
    parser.add_argument('-hg', '--height', type=int, nargs=1,
                        help='Height of the maze')

    return parser.parse_args()


def create_map(name, width: int, height: int):
    filename_maze = f'../../res/layouts/{name}.lay'
    maze = np.ndarray(shape=(width, height)).astype(int)

    for i in range(width):
        for j in range(height):
            if i == 0 and j == 0:
                maze[i][j] = 18
            elif i == 0 and j == height - 1:
                maze[i][j] = 19
            elif i == width - 1 and j == 0:
                maze[i][j] = 16
            elif i == width - 1 and j == height - 1:
                maze[i][j] = 17
            elif i == 0 or i == width - 1:
                maze[i][j] = 25
            elif j == 0 or j == height - 1:
                maze[i][j] = 26
            else:
                maze[i][j] = 10

    np.savetxt(os.path.join(sys.path[0], filename_maze), maze.astype(int), fmt='%i')


if __name__ == '__main__':
    args = parse_args()

    create_map(name=args.layout_name[0], width=args.width[0], height=args.height[0])
