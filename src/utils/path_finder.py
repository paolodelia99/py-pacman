class PathFinder(object):

    def __init__(self, state_map):
        self.state_map = state_map

    @staticmethod
    def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)
