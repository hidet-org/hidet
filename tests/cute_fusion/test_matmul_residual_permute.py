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
from typing import List, Any
import pytest

import numpy as np

import hidet
import torch
from hidet import ops
from hidet.ir.cute.layout import (
    TensorLayout,
    composition,
    ThrValAtom,
    Level,
    TiledTensorLayout,
    logical_divide,
    left_inverse,
)
from hidet.ir.cute.int_tuple import compact_col_major
from hidet.utils import initialize

from fusion_bench_utils import bench


pattern_tests = []


@initialize()
def initialize_tests():
    # hidet.option.cache_dir("./pattern_3")
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    problem = [4096, 4096, 4096, 1]
    head_size = 32
    relu = torch.nn.ReLU()

    import itertools
    from functools import partial

    for mode in ["default", "max-autotune-no-cudagraphs"]:
        for dims in itertools.permutations([0, 1, 2, 3]):

            def graph(dims, a, b, d):
                c = a @ b
                L, M, N = c.shape
                c = c.reshape(L, M, N // head_size, head_size)
                c = c.permute(*dims)
                c = c.reshape(L * N // head_size, M, head_size)
                d = d.reshape(L, M, N // head_size, head_size)
                d = d.permute(*dims)
                d = d.reshape(L * N // head_size, M, head_size)
                c = c + d
                c = relu(c)
                return c

            # we skip the max-autotune to save the ci time
            if mode == "default":
                pattern_tests.append((problem, partial(graph, dims), mode))


def data(M, N, K, L, dtype="float16", device="cuda"):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(L, M, K), dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(L, K, N), dtype=dtype, device=device)
    c = torch.randint(low=lo, high=hi, size=(L, M, N), dtype=dtype, device=device)

    return a, b, c


@pytest.mark.parametrize("args,graph,mode", pattern_tests)
def test_pattern(args, graph, mode):
    M, N, K, L = args
    graph_args = data(*args)

    import torch._dynamo as dynamo

    options = {"triton.cudagraphs": False, "epilogue_fusion": True, "max_autotune": True}
    D = graph(*graph_args)
    dynamo.reset()
    graph_opt = torch.compile(graph, options=options)
    D_opt = graph_opt(*graph_args)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=D_opt.cpu().numpy(), desired=D.cpu().numpy(), rtol=1e-2)

    torch_mean, torch_min, torch_max = bench(graph_opt, graph_args)
    print(f"baseline(torch.compile mode=max-autotune): {torch_mean} ms")

    with hidet.option.context():
        hidet.option.parallel_k(strategy="disabled")

        D = graph(*graph_args)
        dynamo.reset()
        graph_hidet = torch.compile(graph, backend="hidet", mode=mode)
        D_hidet = graph_hidet(*graph_args)

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=D_hidet.cpu().numpy(), desired=D.cpu().numpy(), rtol=1e-2)
    hidet_mean, hidet_min, hidet_max = bench(graph_hidet, graph_args)
    print(f"hidet(torch.compile): {hidet_mean} ms")
    return torch_mean, hidet_mean


def main():
    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k,l)", "dims", "max-autotune", "hidet", "speedup"]
    problem = [4096, 4096, 4096, 1]
    head_size = 32
    relu = torch.nn.ReLU()

    import itertools

    for dims in itertools.permutations([0, 1, 2, 3]):

        def graph(a, b, d):
            c = a @ b
            L, M, N = c.shape
            c = c.reshape(L, M, N // head_size, head_size)
            c = c.permute(*dims)
            c = c.reshape(L * N // head_size, M, head_size)
            d = d.reshape(L, M, N // head_size, head_size)
            d = d.permute(*dims)
            d = d.reshape(L * N // head_size, M, head_size)
            c = c + d
            c = relu(c)
            return c

        torch_time, hidet_time = test_pattern(problem, graph=graph, mode="max-autotune-no-cudagraphs")
        records.append([problem, dims, torch_time, hidet_time, (torch_time / hidet_time - 1.0) * 100.0])

    with open("results_pattern_3.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )