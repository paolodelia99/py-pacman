from src.constants import PATH_FINDER_LOOKUP_TABLE
from src.map import Map
from src.utils.path_finder import PathFinder

maze = Map('test')


def test_path_matrix_101():
    path_matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)
    assert path_matrix.shape == (22, 19)
    assert path_matrix.max() == 100
    assert path_matrix.min() == 0