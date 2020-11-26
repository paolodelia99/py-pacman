import os
from pathlib import Path
from typing import Tuple, List, Dict

from src.utils.game_mode import GameMode

GHOST_COLORS: Dict[int, Tuple[int, int, int, int]] = {
    0: (255, 0, 0, 255),
    1: (255, 128, 255, 255),
    2: (128, 255, 255, 255),
    3: (255, 128, 0, 255)
}

ROOT_DIR = Path(__file__).parent.parent

VULNERABLE_GHOST_COLOR: Tuple[int, int, int, int] = (50, 50, 255, 255)
WHITE_GHOST_COLOR: Tuple[int, int, int, int] = (255, 255, 255, 255)

EDGE_LIGHT_COLOR: Tuple[int, int, int, int] = (255, 206, 255, 255)
FILL_COLOR: Tuple[int, int, int, int] = (132, 0, 132, 255)
EDGE_SHADOW_COLOR: Tuple[int, int, int, int] = (255, 0, 255, 255)
PELLET_COLOR: Tuple[int, int, int, int] = (128, 0, 128, 255)
WHITE_EDGE_LIGHT_COLOR: Tuple[int, int, int, int] = (255, 255, 254, 255)
WHITE_EDGE_SHADOW_COLOR: Tuple[int, int, int, int] = (255, 255, 254, 255)
WHITE_FILL_COLOR: Tuple[int, int, int, int] = (0, 0, 0, 255)

SCREEN_TILE_SIZE_HEIGHT: int = 23
SCREEN_TILE_SIZE_WIDTH: int = 19

TILE_SIZE: int = 24

SCORE_COLWIDTH: int = 13

SCREEN_WIDTH: int = SCREEN_TILE_SIZE_WIDTH * TILE_SIZE
SCREEN_HEIGHT: int = SCREEN_TILE_SIZE_HEIGHT * TILE_SIZE

MODES_TO_ZERO: List = [GameMode.ready, GameMode.hit_ghost, GameMode.wait_to_start,
                       GameMode.wait_after_eating_ghost, GameMode.wait_after_finishing_level,
                       GameMode.flash_maze]

MOVE_MODES: List = [GameMode.normal, GameMode.change_ghosts, GameMode.wait_after_eating_ghost]

TILE_LOOKUP_TABLE: Dict[int, str] = {
    10: 'blank.gif',
    11: 'door-h.gif',
    12: 'door-v.gif',
    13: 'ghost-door.gif',
    14: 'pellet.gif',
    15: 'pellet-power.gif',
    16: 'wall-corner-ll.gif',
    17: 'wall-corner-lr.gif',
    18: 'wall-corner-ul.gif',
    19: 'wall-corner-ur.gif',
    20: 'wall-end-b.gif',
    21: 'wall-end-l.gif',
    22: 'wall-end-r.gif',
    23: 'wall-end-t.gif',
    24: 'wall-nub.gif',
    25: 'wall-straight-horiz.gif',
    26: 'wall-straight-vert.gif',
    27: 'wall-t-bottom.gif',
    28: 'wall-t-left.gif',
    29: 'wall-t-right.gif',
    30: 'wall-t-top.gif',
    31: 'wall-x.gif',
    32: 'x-paintwall.gif',
    33: 'ghost-blinky.gif',
    34: 'ghost-inky.gif',
    35: 'ghost-pinky.gif',
    36: 'ghost-sue.gif',
    50: 'blank.gif'
}

STATE_LOOKUP_TABLE: Dict[int, int] = {
    10: 0,
    11: 0,
    12: 0,
    13: -10,
    14: 1,
    15: 2,
    16: -10,
    17: -10,
    18: -10,
    19: -10,
    20: -10,
    21: -10,
    22: -10,
    23: -10,
    24: -10,
    25: -10,
    26: -10,
    27: -10,
    28: -10,
    29: -10,
    30: -10,
    31: -10,
    32: -10,
    33: -5,
    34: -5,
    35: -5,
    36: -5,
    40: 0,
    50: 0
}

STATE_COLOR_LOOKUP_TABLE: Dict[int, Tuple[int, int, int]] = {
    0: (255, 204, 204),
    1: (255, 102, 102),
    2: (255, 0, 0),
    5: (153, 0, 0),
    -10: (0, 0, 255),
    -5: (51, 51, 255)
}

PATH_FINDER_LOOKUP_TABLE: Dict[int, int] = {
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 1000,
    17: 1000,
    18: 1000,
    19: 1000,
    20: 1000,
    21: 1000,
    22: 1000,
    23: 1000,
    24: 1000,
    25: 1000,
    26: 1000,
    27: 1000,
    28: 1000,
    29: 1000,
    30: 1000,
    31: 1000,
    32: 1000,
    33: 0,
    34: 0,
    35: 0,
    36: 0,
    40: 0,
    50: 1000
}

INVERT_ORIENTATION_TABLE: Dict[str, str] = {
    'D': 'U',
    'U': 'D',
    'L': 'R',
    'R': 'L'
}
