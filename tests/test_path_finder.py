from src.constants import PATH_FINDER_LOOKUP_TABLE
from src.map import Map
from src.utils.path_finder import PathFinder

maze = Map('test')
path_matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)


def test_path_matrix():
    assert path_matrix.shape == (22, 19)
