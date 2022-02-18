class Node:
    def __str__(self):
        from hidet.ir.functors.printer import astext
        return astext(self)

    def __repr__(self):
        return str(self)

    def __int__(self):
        raise NotImplementedError()