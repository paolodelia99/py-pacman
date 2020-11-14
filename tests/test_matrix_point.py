from src.utils.matrix_point import MatrixPoint


def test_manhattan_distance():
    assert MatrixPoint.manhattan_distance(9, 10, 2, -5) == 22
    assert MatrixPoint.manhattan_distance(1, 1, 9, 9) == 16
    assert MatrixPoint.manhattan_distance(0, 0, 100, -100) == 200


