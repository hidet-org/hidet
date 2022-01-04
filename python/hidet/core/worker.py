
class Worker:
    pass


class Grid(Worker):
    def __init__(self, grid_dim=None, block_dim=None):
        self.grid_dim = grid_dim
        self.block_dim = block_dim


class ThreadBlock(Worker):
    def __init__(self, block_dim=None):
        self.block_dim = block_dim


class Warp(Worker):
    def __init__(self):
        pass


class Thread(Worker):
    def __init__(self):
        pass


class Host(Worker):
    def __init__(self):
        pass
