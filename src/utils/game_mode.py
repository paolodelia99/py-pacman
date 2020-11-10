from enum import Enum


class GameMode(Enum):
    ready = 0
    normal = 1
    hit_ghost = 2
    game_over = 3
    wait_to_start = 4
    wait_after_eating_ghost = 5
    wait_after_finishing_level = 6
    flash_maze = 7
    extra_pacman = 8
    change_ghosts = 9
    black_screen = 10
