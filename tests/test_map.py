from src.map import Map

from numpy import int64


def test_map_init():
    map_ = Map('../res/layouts/classic.lay')
    assert map_.map_matrix.shape == (22, 19)
    assert map_.map_matrix.min() == 0
    assert map_.map_matrix.max() == 6
    assert map_.map_matrix.dtype == int64


def test_get_neighbors():
    map_ = Map('../res/layouts/classic.lay')
    assert map_.get_neighbors(0, 0, 0) == 2
    assert map_.get_neighbors(1, 9, 0) == 4
    assert map_.get_neighbors(1, 1, 1) == 1
