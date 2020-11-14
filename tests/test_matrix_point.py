from src.utils.matrix_point import MatrixPoint, manhattan_distance, Point


def test_manhattan_distance():
    assert manhattan_distance(9, 10, 2, -5) == 22
    assert manhattan_distance(1, 1, 9, 9) == 16
    assert manhattan_distance(0, 0, 100, -100) == 200


def test_point_101():
    point = Point(1, 2)
    assert point.x == 1
    assert point.y == 2


def test_check_point_equality():
    p1 = Point(0, 0)
    p2 = Point(0, 0)
    assert p1 == p2


def test_point_manhattan_distance():
    p1 = Point(1, 2)
    p2 = Point(4, 3)
    assert p1.get_distance(p2) == 4
    assert p2.get_distance(p1) == 4


def test_is_cross_neighbor():
    p1 = Point(0, 0)
    p2 = Point(0, 1)
    p3 = Point(1, 1)
    assert p1.is_cross_neighbor(p2) is True
    assert p1.is_cross_neighbor(p3) is False


def test_get_neighbor_orientation():
    p1 = Point(0, 0)
    p2 = Point(0, 1)
    p3 = Point(1, 1)
    assert p1.get_cross_neighbor_orientation(p2) == 'D'
    assert p1.get_cross_neighbor_orientation(p3) is None


def test_matrix_point_101():
    point = MatrixPoint(0, 1, 1)
    assert point.value == 0
    assert point.x == 1
    assert point.y == 1
    assert point.parent is None


def test_distance():
    point1 = MatrixPoint(0, 1, 1)
    point2 = MatrixPoint(0, 3, 3)
    assert point1.get_distance(point2) == 4
    assert point2.get_distance(point1) == 4


def test_get_cost():
    point1 = MatrixPoint(2, 1, 1)
    point2 = MatrixPoint(0, 3, 2)
    assert point1.get_cost(point2) == 5
    assert point2.get_cost(point1) == 3


def test_set_parent():
    point1 = MatrixPoint(2, 1, 1)
    point2 = MatrixPoint(0, 2, 2)
    point3 = MatrixPoint(0, 1, 2)
