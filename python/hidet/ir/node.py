class Node:
    def __str__(self):
        from hidet.ir.functors.printer import astext
        return astext(self)
