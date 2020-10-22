import numpy as np


class Map:

    def __init__(self, layout_name):
        self.layout_name = layout_name
        self.map_matrix = np.loadtxt(self.layout_name).astype(int)

    def is_wall(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 0

    def is_ghost(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 3

    def is_biscuit(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 1

    def is_pill(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 4

    def remove_biscuit_pill(self, x: int, y: int):
        self.map_matrix[x:int, y:int] = 2

    def update_ghosts_pos(self):
        pass

    def draw(self):
        pass
