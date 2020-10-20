from src.map import Map


def test_map_init():
    map_ = Map('../res/layouts/classic-layout.lay')
    assert map_.map_matrix.shape == (22, 19)
