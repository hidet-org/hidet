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
import pytest

import hidet
import torch
from hidet import ops
from hidet.graph.tensor import asarray
from torch import nn
from hidet.testing.torch_utils import Backend, bench_model


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def check_matmul_dynamic(a_shape, b_shape, bias_shape, torch_op, hidet_op, dtype="bfloat16", device="cuda"):
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

    # hidet.option.cache_dir("dynamic")
    hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    graph_opt: hidet.FlowGraph = hidet.graph.optimize(graph)

    def hidet_opt(inp_a, inp_b, inp_bias):
        return graph_opt(inp_a, inp_b, inp_bias)

    torch_result = torch_op(a, b, bias)
    hidet_result = hidet_opt(a_hidet, b_hidet, bias_hidet).cpu()

    np.testing.assert_allclose(
        actual=torch_result.to(torch.float32).cpu().numpy(), desired=hidet_result.to('float32').cpu().numpy(), rtol=1e-2
    )


@pytest.mark.requires_cuda
def test_matmul_dynamic_fallback():
    a_shape = [("b", 16), ("s", 96), 256]
    b_shape = [256, 30522]
    bias_shape = [("b", 16), ("s", 96), 30522]

    torch_op = lambda a, b, bias: ((a @ b) + bias).to(torch.float32)
    hidet_op = lambda a, b, bias: hidet.ops.cast(hidet.ops.matmul(a, b) + bias, "float32")

    check_matmul_dynamic(a_shape, b_shape, bias_shape, torch_op, hidet_op)


@pytest.mark.requires_cuda
def test_matmul_dynamic():
    a_shape = [("b", 16), ("s", 96), 256]
    b_shape = [256, 30522]
    bias_shape = [("b", 16), ("s", 96), 30522]

    torch_op = lambda a, b, bias: (a @ b).to(torch.float32)
    hidet_op = lambda a, b, bias: hidet.ops.cast(hidet.ops.matmul(a, b), "float32")

    check_matmul_dynamic(a_shape, b_shape, bias_shape, torch_op, hidet_op)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("hexcute_matmul", ["enable", "disable"])
def test_matmul_relu_1(hexcute_matmul: bool):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4096, 6144)

        def forward(self, x):
            y = self.linear1(x)
            y = torch.relu(y)
            y = y + 1
            return y

    with hidet.option.context():
        hidet.option.hexcute_matmul(hexcute_matmul)
        # hidet.option.save_lower_ir(True)
        # hidet.option.debug_cache_tuning()
        backend = Backend(backend='hidet', mode='max-autotune-no-cudagraphs', dtype=torch.bfloat16)
        model = Model().cuda().to(torch.bfloat16).eval()

        with torch.inference_mode(True):
            compiled_model = backend.compile(model)

            input = torch.randn(8, 4096, dtype=torch.bfloat16, device='cuda')
            torch._dynamo.mark_dynamic(input, 0)
            compiled_model(input)

            j = 1024
            input = torch.randn(j, 4096, dtype=torch.bfloat16, device='cuda')
            y1 = compiled_model(input)
            y2 = model(input)
            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(
                actual=y1.to(torch.float32).cpu().numpy(), desired=y2.to(torch.float32).cpu().numpy(), rtol=5e-2
            )

            lat = bench_model(compiled_model, [input])
            print(lat)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("hexcute_matmul", ["enable", "disable"])
def test_matmul_matmul_add(hexcute_matmul: bool):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4096, 6144)
            self.linear2 = nn.Linear(6144, 4096)

        def forward(self, x):
            y = self.linear1(x)
            y = torch.relu(y)
            y = self.linear2(y)
            y = torch.relu(y + x)
            return y

    with hidet.option.context():
        hidet.option.hexcute_matmul(hexcute_matmul)
        # hidet.option.cache_dir("./222")
        # hidet.option.save_lower_ir(True)
        # hidet.option.debug_cache_tuning()
        backend = Backend(backend='hidet', mode='default', dtype=torch.bfloat16)
        model = Model().cuda().to(torch.bfloat16).eval()

        with torch.inference_mode(True):
            weight1 = torch.randint(low=0, high=3, size=(6144, 4096), dtype=torch.bfloat16, device='cuda') / 4096
            weight2 = torch.randint(low=0, high=3, size=(4096, 6144), dtype=torch.bfloat16, device='cuda') / 6144
            model.linear1.weight.copy_(weight1)
            model.linear2.weight.copy_(weight2)
            compiled_model = backend.compile(model)

            input = torch.randn(8, 4096, dtype=torch.bfloat16, device='cuda')
            torch._dynamo.mark_dynamic(input, 0)
            compiled_model(input)

            j = 1
            input = torch.randint(low=-3, high=3, size=(j, 4096), dtype=torch.bfloat16, device='cuda')
            y1 = compiled_model(input)
            y2 = model(input)
            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(
                actual=y1.to(torch.float32).cpu().numpy(), desired=y2.to(torch.float32).cpu().numpy(), rtol=5e-2
            )


@pytest.mark.requires_cuda
@pytest.mark.parametrize("hexcute_matmul", ["enable", "disable"])
def test_matmul_add_scalar(hexcute_matmul: bool):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4096, 6144)

        def forward(self, x):
            y = self.linear1(x)
            y = torch.relu(y)
            s = x.sum(0).sum(0)
            y = y + s
            return y

    with hidet.option.context():
        hidet.option.hexcute_matmul(hexcute_matmul)
        # hidet.option.cache_dir("./333")
        # hidet.option.save_lower_ir(True)
        # hidet.option.debug_cache_tuning()
        backend = Backend(backend='hidet', mode='default', dtype=torch.bfloat16)
        model = Model().cuda().to(torch.bfloat16).eval()

        with torch.inference_mode(True):
            compiled_model = backend.compile(model)

            input = torch.randn(8, 4096, dtype=torch.bfloat16, device='cuda')
            torch._dynamo.mark_dynamic(input, 0)
            compiled_model(input)

            j = 1
            input = torch.randn(j, 4096, dtype=torch.bfloat16, device='cuda')
            y1 = compiled_model(input)
            y2 = model(input)
            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(
                actual=y1.to(torch.float32).cpu().numpy(), desired=y2.to(torch.float32).cpu().numpy(), rtol=5e-2
            )
