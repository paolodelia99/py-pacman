from .functions import manhattan_distance


class MatrixPoint(object):

    def __init__(self, value: int, x: int, y: int):
        self.value = value
        self.x = x
        self.y = y
        self.parent = None

    def get_distance(self, point2: MatrixPoint) -> int:
        return manhattan_distance(self.x, self.y, point2.x, point2.y)

    def get_cost(self, point2: MatrixPoint) -> int:
        return self.value + self.get_distance(point2)

    def set_parent(self, point2: MatrixPoint):
        self.parent = point2
