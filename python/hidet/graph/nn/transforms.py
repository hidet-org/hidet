# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from hidet.graph import ops
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor, empty


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = empty(shape=[num_embeddings, embedding_dim], dtype='float32')

    def forward(self, indices: Tensor) -> Tensor:
        return ops.take(self.weight, indices, axis=0)


class SymQuantEmbedding(Module):
    def __init__(self, weight: Tensor, quant_dtype: str = 'int8'):
        super().__init__()
        self.qweight, self.scale = ops.symmetric_quantize(weight, quant_dtype)

    def forward(self, indices: Tensor) -> Tensor:
        return ops.take(ops.symmetric_dequantize(ops.barrier(self.qweight), self.scale), indices, axis=0)
