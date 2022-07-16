
class MmaConfig:
    def __init__(self, m, n, k, a_dtype, b_dtype, c_dtype):
        self.m = m
        self.n = n
        self.k = k
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.a_regs = None
        self.b_regs = None
        self.c_regs = None
