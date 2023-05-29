from hidet.graph import ops
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor, randn


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = randn(shape=[num_embeddings, embedding_dim], dtype='float32', mean=0.0, stddev=1.0)

    def forward(self, indices: Tensor) -> Tensor:
        return ops.take(self.weight, indices, axis=0)
