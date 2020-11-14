from src.constants import PATH_FINDER_LOOKUP_TABLE
from src.map import Map
from src.utils.functions import get_neighbors, manhattan_distance

maze = Map('test')
matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)


def test_manhattan_distance():
    assert manhattan_distance(9, 10, 2, -5) == 22
    assert manhattan_distance(1, 1, 9, 9) == 16
    assert manhattan_distance(0, 0, 100, -100) == 200


def test_neighbors_inside_map():
    neighbors = get_neighbors(matrix, 16, 9)
    assert neighbors['R'] == 0
    assert neighbors['L'] == 0
    assert neighbors['U'] == 100
    assert neighbors['D'] == 100


def test_neighbors_left_upper_angle():
    neighbors = get_neighbors(matrix, 0, 0)
    assert neighbors['R'] == 100
    assert neighbors['L'] is None
    assert neighbors['U'] is None
    assert neighbors['D'] == 100


def test_neighbors_right_upper_angle():
    neighbors = get_neighbors(matrix, 0, matrix.shape[1] - 1)
    assert neighbors['R'] is None
    assert neighbors['L'] == 100
    assert neighbors['U'] is None
    assert neighbors['D'] == 100


def test_neighbors_left_lower_angle():
    neighbors = get_neighbors(matrix, matrix.shape[0] - 1, 0)
    assert neighbors['R'] == 100
    assert neighbors['L'] is None
    assert neighbors['U'] == 100
    assert neighbors['D'] is None


def test_neighbors_right_lower_angle():
    neighbors = get_neighbors(matrix, matrix.shape[0] - 1, matrix.shape[1] - 1)
    assert neighbors['R'] is None
    assert neighbors['L'] == 100
    assert neighbors['U'] == 100
    assert neighbors['D'] is None
