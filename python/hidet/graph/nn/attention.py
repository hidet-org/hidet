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
from typing import Optional
import math

from hidet.graph import ops
from hidet.graph.nn.container import ModuleList
from hidet.graph.nn.linear import Linear
from hidet.graph.nn.module import Module
from hidet.graph.tensor import Tensor
from hidet.utils.py import prod


class CrossAttention(Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.upcast = upcast
        self.out_bias = out_bias

        self.to_q = Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = Linear(self.cross_attention_dim, self.inner_dim, bias=False)
        self.to_v = Linear(self.cross_attention_dim, self.inner_dim, bias=False)

        self.to_out = ModuleList([Linear(self.inner_dim, self.query_dim, bias=out_bias)])

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Optional[Tensor] = None, temperature_scaling: float = 1.0
    ) -> Tensor:
        ndim = len(hidden_states.shape)
        if ndim == 4:
            bs, c, h, w = hidden_states.shape
            hidden_states = hidden_states.reshape([bs, c, h * w]).transpose(1, 2)

        bs, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        q = self.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        inner_dim = k.shape[-1]
        head_dim = inner_dim // self.heads

        other_dims = bs * inner_dim
        assert (prod(q.shape) % other_dims) == 0

        q, k, v = tuple(t.reshape((bs, -1, self.heads, head_dim)).transpose(1, 2).to('bfloat16') for t in (q, k, v))
        q = q * (1 / math.sqrt(head_dim))
        k = k.transpose(-1, -2)

        # Use softmax temperature parameter to prevent QK matmul causing float overflow
        # due to limited fp16 range. May cause accuracy issues, should only be applied
        # for attention layers that have overflow issue. Alternate solution is to
        # cast to fp32 and use mm/softmax/mm attention
        assert temperature_scaling >= 1.0
        if temperature_scaling != 1.0:
            q = q / temperature_scaling

        hidden_states = ops.attention(q, k, v).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states.transpose(1, 2).reshape((bs, -1, inner_dim))
        hidden_states = self.to_out[0](hidden_states)

        if ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape((bs, c, h, w))

        return hidden_states
