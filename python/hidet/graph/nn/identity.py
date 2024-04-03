from hidet.graph.nn.module import Module


class Identity(Module):
    """
    Identity function.

    Used as a dummy for replacing modules (e.g. remove a layer in module list
    but need to keep indices in container to match torch model)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x
