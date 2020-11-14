class MatrixPoint(object):

    def __init__(self, value: int, x: int, y: int):
        self.value = value
        self.x = x
        self.y = y
        self.parent = None

    def get_distance(self, point2) -> int:
        return MatrixPoint.manhattan_distance(self.x, self.y, point2.x, point2.y)

    def get_cost(self, point2) -> int:
        return self.value + self.get_distance(point2)

    def set_parent(self, point2):
        self.parent = point2

    @staticmethod
    def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)
