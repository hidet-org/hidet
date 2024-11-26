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
from hidet.option import OptionContext
from hidet.ir.cute.int_tuple import compact_col_major
from hidet.utils import initialize

from fusion_bench_utils import bench


pattern_tests = []


@initialize()
def initialize_tests():
    # hidet.option.cache_dir("./graph")
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    dim = 12 * 256
    dim0 = dim // 8
    dim3 = 32
    import itertools

    for dims in itertools.permutations([0, 1, 2, 3]):
        assert isinstance(dims, tuple)
        problem = [dim, 768, 320]
        pattern_tests.append((dim0, dim3, dims, problem))

    dim0 = 8
    dim3 = 32
    for dims in itertools.permutations([0, 1, 2, 3]):
        assert isinstance(dims, tuple)
        problem = [dim, 768, 320]
        if dims[3] == 0:
            continue
        pattern_tests.append((dim0, dim3, dims, problem))


def data(M, N, K, dtype="float16", device="cuda"):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(1, M, K), dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(1, K, N), dtype=dtype, device=device)

    return a, b


@pytest.mark.parametrize("dim0,dim3,dims,args", pattern_tests)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_epilogue_fusion(dim0, dim3, dims, args, dtype):
    M, N, K = args
    A, B = data(*args, dtype=dtype)

    import torch._dynamo as dynamo

    def graph(a, b):
        c = a @ b
        _, M, N = c.shape
        c = c.reshape(dim0, M // dim0, N // dim3, dim3)
        c = c.permute(dims)
        c = c.contiguous()
        return c

    options = {"triton.cudagraphs": False, "epilogue_fusion": True, "max_autotune": True}
    C = graph(A, B)
    dynamo.reset()
    graph_opt = torch.compile(graph, options=options)
    C_opt = graph_opt(A, B)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(
        actual=C_opt.to(torch.float32).cpu().numpy(), desired=C.to(torch.float32).cpu().numpy(), rtol=1e-2
    )

    torch_mean, torch_min, torch_max = bench(graph_opt, (A, B))
    print(f"baseline(torch.compile mode=max-autotune): {torch_mean} ms")

    with hidet.option.context():
        hidet.option.cache_dir("./graph")
        hidet.option.search_space(2)
        hidet.option.debug_cache_tuning()
        # commented out because parallel build failed when dumping lower ir
        # hidet.option.save_lower_ir(True)
        hidet.option.parallel_k(strategy='disabled')

        C = graph(A, B)
        dynamo.reset()
        graph_hidet = torch.compile(graph, backend="hidet", mode="max-autotune-no-cudagraphs")
        C_hidet = graph_hidet(A, B)

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(
        actual=C_hidet.to(torch.float32).cpu().numpy(), desired=C.to(torch.float32).cpu().numpy(), rtol=1e-2
    )
    hidet_mean, hidet_min, hidet_max = bench(graph_hidet, (A, B))
    print(f"hidet(torch.compile): {hidet_mean} ms")


def main():
    dim = 12 * 256
    dim0 = dim // 8
    dim3 = 32
    import itertools
    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k)", "dim0", "dim1", "dims", "max-autotune", "hidet", "speedup"]
    for dims in itertools.permutations([0, 1, 2, 3]):
        assert isinstance(dims, tuple)
        problem = [dim, 768, 320]
        torch_time, hidet_time = test_epilogue_fusion(dim0, dim3, dims, problem)
        records.append([problem, dim0, dim3, dims, torch_time, hidet_time, (torch_time / hidet_time - 1.0) * 100.0])

    dim0 = 8
    dim3 = 32
    for dims in itertools.permutations([0, 1, 2, 3]):
        assert isinstance(dims, tuple)
        problem = [dim, 768, 320]
        if dims[3] == 0:
            continue
        torch_time, hidet_time = test_epilogue_fusion(dim0, dim3, dims, problem)
        records.append([problem, dim0, dim3, dims, torch_time, hidet_time, (torch_time / hidet_time - 1.0) * 100.0])

    with open("results.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )
