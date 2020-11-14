from typing import Dict, Any, Optional


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


class Point(object):

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_distance(self, point2) -> int:
        return manhattan_distance(self.x, self.y, point2.x, point2.y)

    def is_cross_neighbor(self, point2) -> bool:
        return [point2 == point for k, point in self.get_cross_neighbors().items()].count(True) > 0

    def get_cross_neighbors(self) -> Dict[str, Any]:
        return {
            'U': Point(self.x, self.y - 1),
            'D': Point(self.x, self.y + 1),
            'L': Point(self.x - 1, self.y),
            'R': Point(self.x + 1, self.y)
        }

    def get_cross_neighbor_orientation(self, point2) -> Optional[str]:
        if self.is_cross_neighbor(point2):
            for k, point in self.get_cross_neighbors().items():
                if point2 == point:
                    return k
        else:
            return None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"Point x: {self.x}, y: {self.y}"


class MatrixPoint(Point):

    def __init__(self, value: int, x: int, y: int):
        super().__init__(x, y)
        self.value = value
        self.parent = None
        self.g = 0

    def get_cost(self, point2) -> int:
        return self.value + self.g + self.get_distance(point2)

    def set_parent(self, point2):
        if self.get_distance(point2) == 1 and self.is_cross_neighbor(point2):
            self.parent = point2

    def is_cross_neighbor(self, point2) -> bool:
        return [point2.same_position(point) for k, point in self.get_cross_neighbors().items()].count(True) > 0

    def same_position(self, point2: Point):
        return self.x == point2.x and self.y == point2.y

    def get_parent_orientation(self):
        return self.get_cross_neighbor_orientation(self.parent)

    def get_cross_neighbor_orientation(self, point2) -> Optional[str]:
        if self.is_cross_neighbor(point2):
            for k, point in self.get_cross_neighbors().items():
                if point2.same_position(point):
                    return k
        else:
            return None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.value == other.value

    def __hash__(self):
        return hash((self.x, self.y, self.value))

    def __str__(self):
        return f"Matrix Point value: {self.value}, x: {self.x}, y: {self.y}"

    def __copy__(self):
        return MatrixPoint(self.value, self.x, self.y)
