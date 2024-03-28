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
from hidet.graph.nn.linear import Linear
from hidet.graph.nn.module import Module


class Relu(Module):
    def forward(self, x):
        return ops.relu(x)


class Gelu(Module):
    def forward(self, x):
        return ops.gelu(x)


class Geglu(Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        hidden_states, gate = ops.split(x, 2, axis=2)
        return hidden_states * ops.gelu(gate)


class Tanh(Module):
    def forward(self, x):
        return ops.tanh(x)
