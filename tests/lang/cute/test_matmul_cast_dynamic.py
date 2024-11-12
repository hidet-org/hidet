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
import numpy as np

import hidet
import torch
from hidet import ops
from hidet.graph.tensor import asarray


def check_matmul_dynamic(a_shape, b_shape, bias_shape, torch_op, hidet_op, dtype="float16", device="cuda"):
    lo = -3
    hi = 3
    dtype = getattr(torch, dtype)
    a_concrete_shape = tuple((i if isinstance(i, int) else i[1]) for i in a_shape)
    a_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in a_shape]

    b_concrete_shape = tuple((i if isinstance(i, int) else i[1]) for i in b_shape)
    b_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in b_shape]

    bias_concrete_shape = tuple((i if isinstance(i, int) else i[1]) for i in bias_shape)
    bias_symbolic_shape = [(i if isinstance(i, int) else i[0]) for i in bias_shape]

    a = torch.randint(low=lo, high=hi, size=a_concrete_shape, dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=b_concrete_shape, dtype=dtype, device=device)
    bias = torch.randint(low=lo, high=hi, size=bias_concrete_shape, dtype=dtype, device=device)

    a_hidet = hidet.from_torch(a)
    b_hidet = hidet.from_torch(b)
    bias_hidet = hidet.from_torch(bias)

    sym_a = hidet.symbol(a_symbolic_shape, dtype=a_hidet.dtype, device=a_hidet.device)
    sym_b = hidet.symbol(b_symbolic_shape, dtype=b_hidet.dtype, device=b_hidet.device)
    sym_bias = hidet.symbol(bias_symbolic_shape, dtype=bias_hidet.dtype, device=bias_hidet.device)

    sym_output = hidet_op(sym_a, sym_b, sym_bias)

    graph: hidet.FlowGraph = hidet.trace_from(sym_output, inputs=[sym_a, sym_b, sym_bias])

    hidet.option.cache_dir("dynamic")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    with hidet.graph.PassContext() as ctx:
        ctx.set_parallel_k()
        graph_opt: hidet.FlowGraph = hidet.graph.optimize(graph)

    def hidet_opt(inp_a, inp_b, inp_bias):
        return graph_opt(inp_a, inp_b, inp_bias)

    torch_result = torch_op(a, b, bias)
    hidet_result = hidet_opt(a_hidet, b_hidet, bias_hidet).cpu()

    np.testing.assert_allclose(actual=torch_result.cpu().numpy(), desired=hidet_result.cpu().numpy(), rtol=1e-2)


def test_matmul_dynamic_fallback():
    a_shape = [("b", 16), ("s", 96), 256]
    b_shape = [256, 30522]
    bias_shape = [("b", 16), ("s", 96), 30522]

    torch_op = lambda a, b, bias: ((a @ b) + bias).to(torch.float32)
    hidet_op = lambda a, b, bias: hidet.ops.cast(hidet.ops.matmul(a, b) + bias, "float32")

    check_matmul_dynamic(a_shape, b_shape, bias_shape, torch_op, hidet_op)


def test_matmul_dynamic():
    a_shape = [("b", 16), ("s", 96), 256]
    b_shape = [256, 30522]
    bias_shape = [("b", 16), ("s", 96), 30522]

    torch_op = lambda a, b, bias: (a @ b).to(torch.float32)
    hidet_op = lambda a, b, bias: hidet.ops.cast(hidet.ops.matmul(a, b), "float32")

    check_matmul_dynamic(a_shape, b_shape, bias_shape, torch_op, hidet_op)
