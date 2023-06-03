from hidet.graph import ops
from hidet.graph.nn.module import Module


class Relu(Module):
    def forward(self, x):
        return ops.relu(x)


class Gelu(Module):
    def forward(self, x):
        return ops.gelu(x)


class Tanh(Module):
    def forward(self, x):
        return ops.tanh(x)
