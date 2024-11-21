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
from dataclasses import dataclass

import torch
import hidet

from hidet.utils.benchmark import Bench


@dataclass
class BenchMatmulConfig:
    dtype: torch.dtype = torch.bfloat16
    space: int = 2


# Each function to be benchmarked takes in a config, which can be anything
#   and stays constant between runs, and an int. And returns a function that
#   takes no arguments and returns nothing; this is the function to be benchmarked.


def torch_matmul(config: BenchMatmulConfig, C: int):
    a = torch.randn((C, C), device='cuda', dtype=config.dtype)
    b = torch.randn((C, C), device='cuda', dtype=config.dtype)
    return lambda: torch.matmul(a, b)


def hidet_matmul(config: BenchMatmulConfig, C: int):
    a = hidet.from_torch(torch.randn((C, C), device='cuda', dtype=config.dtype))
    b = hidet.from_torch(torch.randn((C, C), device='cuda', dtype=config.dtype))
    sa = hidet.symbol_like(a)
    sb = hidet.symbol_like(b)
    ys = hidet.ops.matmul(sa, sb)
    g = hidet.trace_from(ys, [sa, sb])
    g = hidet.graph.optimize(g)
    func = g.build(space=config.space)
    return lambda: func(a, b)


bn = Bench(x_name='C', x_vals=[128 * i for i in range(2, 4)], config=BenchMatmulConfig())
bn.bench(torch_matmul)
bn.bench(hidet_matmul)
bn.measure_flops(lambda config, c: torch.finfo(config.dtype).bits // 8 * c**2)

data = bn.run()
data.show_plot()
data.print_data()
