from src.constants import PATH_FINDER_LOOKUP_TABLE
from src.map import Map
from src.utils.functions import get_neighbors

maze = Map('test')
matrix = maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE)


def test_neighbors_inside_map():
    neighbors = get_neighbors(matrix, 16, 9)
    assert neighbors['R'].value == 0
    assert neighbors['L'].value == 0
    assert neighbors['U'].value == 100
    assert neighbors['D'].value == 100


def test_neighbors_left_upper_angle():
    neighbors = get_neighbors(matrix, 0, 0)
    assert neighbors['R'].value == 100
    assert neighbors['L'] is None
    assert neighbors['U'] is None
    assert neighbors['D'].value == 100


def test_neighbors_right_upper_angle():
    neighbors = get_neighbors(matrix, 0, matrix.shape[1] - 1)
    assert neighbors['R'] is None
    assert neighbors['L'].value == 100
    assert neighbors['U'] is None
    assert neighbors['D'].value == 100


def test_neighbors_left_lower_angle():
    neighbors = get_neighbors(matrix, matrix.shape[0] - 1, 0)
    assert neighbors['R'].value == 100
    assert neighbors['L'] is None
    assert neighbors['U'].value == 100
    assert neighbors['D'] is None


def test_neighbors_right_lower_angle():
    neighbors = get_neighbors(matrix, matrix.shape[0] - 1, matrix.shape[1] - 1)
    assert neighbors['R'] is None
    assert neighbors['L'].value == 100
    assert neighbors['U'].value == 100
    assert neighbors['D'] is None
