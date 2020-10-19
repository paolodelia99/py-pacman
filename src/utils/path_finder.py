from .node import Node


class PathFinder(object):

    def __init__(self):
        # map is a 1-DIMENSIONAL array.
        # use the unfold( (row, col) ) function to convert a 2D coordinate pair
        # into a 1D index to use with this array.
        self.map = {}
        self.size = (-1, -1)  # rows by columns

        self.path_chain_rev = ""
        self.path_chain = ""

        # starting and ending nodes
        self.start = (-1, -1)
        self.end = (-1, -1)

        # current node (used by algorithm)
        self.current = (-1, -1)

        # open and closed lists of nodes to consider (used by algorithm)
        self.open_list = []
        self.closed_list = []

        # used in algorithm (adjacent neighbors path finder is allowed to consider)
        self.neighbor_set = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def resize_map(self, num_rows, num_cols):
        self.map = {}
        self.size = (num_rows, num_cols)

        # initialize path_finder map to a 2D array of empty nodes
        for row in range(0, self.size[0], 1):
            for col in range(0, self.size[1], 1):
                self.set_node(row, col, Node())
                self.set_type(row, col, 0)

    def clean_up_tmp(self):
        # this resets variables needed for a search (but preserves the same map / maze)
        self.path_chain_rev = ""
        self.path_chain = ""
        self.current = (-1, -1)
        self.open_list = []
        self.closed_list = []

    def find_path(self, startPos, endPos):

        self.clean_up_tmp()

        # (row, col) tuples
        self.start = startPos
        self.end = endPos

        # add start node to open list
        self.add_to_open_list(self.start)
        self.set_g(self.start, 0)
        self.set_h(self.start, 0)
        self.set_f(self.start, 0)

        do_continue = True

        while do_continue:

            this_lowest_f_node = self.get_lowest_f_node()

            if not this_lowest_f_node == self.end and not this_lowest_f_node == False:
                self.current = this_lowest_f_node
                self.remove_from_open_list(self.current)
                self.add_to_closed_list(self.current)

                for offset in self.neighbor_set:
                    this_neighbor = (self.current[0] + offset[0], self.current[1] + offset[1])

                    if not this_neighbor[0] < 0 and not this_neighbor[1] < 0 and not this_neighbor[0] > self.size[
                        0] - 1 and not this_neighbor[1] > self.size[1] - 1 and not self.get_type(this_neighbor) == 1:
                        cost = self.get_g(self.current) + 10

                        if self.is_in_open_list(this_neighbor) and cost < self.get_g(this_neighbor):
                            self.remove_from_open_list(this_neighbor)

                        # if self.is_in_closed_list( this_neighbor ) and cost < self.get_g( this_neighbor ):
                        #   self.RemoveFromClosedList( this_neighbor )

                        if not self.is_in_open_list(this_neighbor) and not self.is_in_closed_list(this_neighbor):
                            self.add_to_open_list(this_neighbor)
                            self.set_g(this_neighbor, cost)
                            self.calc_h(this_neighbor)
                            self.calc_f(this_neighbor)
                            self.set_parent(this_neighbor, self.current)
            else:
                do_continue = False

        if not this_lowest_f_node:
            return False

        # reconstruct path
        self.current = self.end
        while not self.current == self.start:
            # build a string representation of the path using R, L, D, U
            if self.current[1] > self.get_parent(self.current)[1]:
                self.path_chain_rev += 'R'
            elif self.current[1] < self.get_parent(self.current)[1]:
                self.path_chain_rev += 'L'
            elif self.current[0] > self.get_parent(self.current)[0]:
                self.path_chain_rev += 'D'
            elif self.current[0] < self.get_parent(self.current)[0]:
                self.path_chain_rev += 'U'
            self.current = self.get_parent(self.current)
            self.set_type(self.current[0], self.current[1], 4)

        # because path_chain_rev was constructed in reverse order, it needs to be reversed!
        for i in range(len(self.path_chain_rev) - 1, -1, -1):
            self.path_chain += self.path_chain_rev[i]

        # set start and ending positions for future reference
        self.set_type(self.start[0], self.start[1], 2)
        self.set_type(self.end[0], self.start[1], 3)

        return self.path_chain

    def unfold(self, row, col):
        # this function converts a 2D array coordinate pair (row, col)
        # to a 1D-array index, for the object's 1D map array.
        return (row * self.size[1]) + col

    def set_node(self, row, col, new_node):
        # sets the value of a particular map cell (usually refers to a node object)
        self.map[self.unfold(row, col)] = new_node

    def get_type(self, val):
        row, col = val
        return self.map[self.unfold(row, col)].type

    def set_type(self, row, col, new_value):
        self.map[self.unfold(row, col)].type = new_value

    def get_f(self, val):
        row, col = val
        return self.map[self.unfold(row, col)].f

    def get_g(self, val):
        row, col = val
        return self.map[self.unfold(row, col)].g

    def GetH(self, val):
        row, col = val
        return self.map[self.unfold(row, col)].h

    def set_g(self, val, new_value):
        row, col = val
        self.map[self.unfold(row, col)].g = new_value

    def set_h(self, val, new_value):
        row, col = val
        self.map[self.unfold(row, col)].h = new_value

    def set_f(self, val, new_value):
        row, col = val
        self.map[self.unfold(row, col)].f = new_value

    def calc_h(self, val):
        row, col = val
        self.map[self.unfold(row, col)].h = abs(row - self.end[0]) + abs(col - self.end[0])

    def calc_f(self, val):
        row, col = val
        unfold_index = self.unfold(row, col)
        self.map[unfold_index].f = self.map[unfold_index].g + self.map[unfold_index].h

    def add_to_open_list(self, val):
        row, col = val
        self.open_list.append((row, col))

    def remove_from_open_list(self, val):
        row, col = val
        self.open_list.remove((row, col))

    def is_in_open_list(self, val):
        row, col = val
        if self.open_list.count((row, col)) > 0:
            return True
        else:
            return False

    def get_lowest_f_node(self):
        lowest_value = 1000  # start arbitrarily high
        lowest_pair = (-1, -1)

        for ordered_pair in self.open_list:
            if self.get_f(ordered_pair) < lowest_value:
                lowest_value = self.get_f(ordered_pair)
                lowest_pair = ordered_pair

        if not lowest_pair == (-1, -1):
            return lowest_pair
        else:
            return False

    def add_to_closed_list(self, val):
        row, col = val
        self.closed_list.append((row, col))

    def is_in_closed_list(self, val):
        row, col = val
        if self.closed_list.count((row, col)) > 0:
            return True
        else:
            return False

    def set_parent(self, val, val2):
        row, col = val
        parent_row, parent_col = val2
        self.map[self.unfold(row, col)].parent = (parent_row, parent_col)

    def get_parent(self, val):
        row, col = val
        return self.map[self.unfold(row, col)].parent

    def draw(self, screen, level):
        for row in range(0, self.size[0], 1):
            for col in range(0, self.size[1], 1):
                this_tile = self.get_type((row, col))
                screen.blit(level.tile_id_image[this_tile], (col * 32, row * 32))
