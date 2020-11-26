from src.constants import TILE_SIZE
from src.pacman import Pacman
from src.map import Map


def test_init_home():
    p = Pacman()
    maze = Map('classic')
    home_x, home_y = maze.get_player_home()
    p.init_home(home_x, home_y)
    assert p.home_x == 9
    assert p.home_y == 16
    assert p.x == p.home_x * TILE_SIZE
    assert p.y == p.home_y * TILE_SIZE
    assert p.nearest_col == p.home_x
    assert p.nearest_row == p.home_y
    p.init_home(0, 0)
    assert p.home_x == 9
    assert p.home_y == 16
    assert p.x == p.home_x * TILE_SIZE
    assert p.y == p.home_y * TILE_SIZE
    assert p.nearest_col == p.home_x
    assert p.nearest_row == p.home_y
