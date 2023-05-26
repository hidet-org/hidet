from hidet.graph import ops
from hidet.graph.nn.module import Module


class Relu(Module):
    def forward(self, x):
        return ops.relu(x)


class Gelu(Module):
    def forward(self, x):
        return x * (ops.erf(x * (1.0 / 1.4142135381698608)) + 1.0) * 0.5


class Tanh(Module):
    def forward(self, x):
        return ops.tanh(x)
