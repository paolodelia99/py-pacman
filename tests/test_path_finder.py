from src.constants import PATH_FINDER_LOOKUP_TABLE
from src.map import Map
from src.utils.path_finder import PathFinder

maze = Map('test')


def test_get_random_allow_position():
    path_matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)
    path_finder = PathFinder(path_matrix)
    x, y = path_finder.get_random_allow_position()
    assert path_matrix[y][x] == 0


def test_path_matrix_101():
    path_matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)
    assert path_matrix.shape == (22, 19)
    assert path_matrix.max() == 100
    assert path_matrix.min() == 0


def test_a_star_101():
    path_matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)
    path_finder = PathFinder(path_matrix)
    path = path_finder.get_min_path(12, 12, 9, 10)
    assert path == ['U', 'U', 'U', 'U', 'L', 'L', 'L', 'D', 'D']


def test_a_star_201():
    path_matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)
    path_finder = PathFinder(path_matrix)
    path = path_finder.get_min_path(1, 1, 6, 6)
    assert path == ['R', 'R', 'R', 'D', 'D', 'D', 'R', 'R', 'D', 'D']


def test_a_star_301():
    path_matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)
    path_finder = PathFinder(path_matrix)
    path = path_finder.get_min_path(9, 8, 1, 18)
    assert path == ['L', 'L', 'L', 'D', 'D', 'D', 'D', 'D', 'D', 'L', 'L', 'D', 'D', 'D', 'D', 'L', 'L', 'L']
