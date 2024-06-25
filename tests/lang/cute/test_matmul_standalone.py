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


bench_suites = 'functional'


matmul_tests = []


@initialize()
def initialize_tests():
    hidet.option.cache_dir("./matmul_standalone")
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    # square
    matmul_tests.append((256, 256, 256, 1))
    if bench_suites == 'performance':
        matmul_tests.append((512, 512, 512, 1))
        matmul_tests.append((1024, 1024, 1024, 1))
        matmul_tests.append((2048, 2048, 2048, 1))

    if bench_suites == 'functional':
        matmul_tests.append((256, 256 + 3, 256, 1))
        matmul_tests.append((256, 256 + 2, 256, 1))
        matmul_tests.append((256, 256 - 4, 256, 1))
    elif bench_suites == 'performance':
        # LLMs
        for batchsize in [1, 16]:
            # Llama7B
            matmul_tests.append((batchsize, 4096 * 3, 4096, 1))
            matmul_tests.append((batchsize, 4096, 4096, 1))
            matmul_tests.append((batchsize, 2 * 10752, 4096, 1))
            matmul_tests.append((batchsize, 10752, 4096, 1))

            # Llama13B
            matmul_tests.append((batchsize, 5120, 3 * 5120, 1))
            matmul_tests.append((batchsize, 5120, 5120, 1))
            matmul_tests.append((batchsize, 5120, 2 * 13568, 1))
            matmul_tests.append((batchsize, 13568, 5120, 1))

            # Llama33B
            matmul_tests.append((batchsize, 6656, 3 * 6656, 1))
            matmul_tests.append((batchsize, 6656, 6656, 1))
            matmul_tests.append((batchsize, 6656, 2 * 17664, 1))
            matmul_tests.append((batchsize, 17664, 6656, 1))

            # Llama65B
            matmul_tests.append((batchsize, 8192, 3 * 8192, 1))
            matmul_tests.append((batchsize, 8192, 8192, 1))
            matmul_tests.append((batchsize, 8192, 2 * 217600, 1))
            matmul_tests.append((batchsize, 217600, 8192, 1))
    else:
        raise NotImplementedError()


def data(M, N, K, L, dtype="float16", device="cuda"):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(L, M, K), dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(L, K, N), dtype=dtype, device=device)

    return a, b


@pytest.mark.parametrize("M,N,K,L", matmul_tests)
def test_problem(M, N, K, L):
    def graph(a, b):
        c = a @ b
        return c

    graph_args = data(M, N, K, L)

    import torch._dynamo as dynamo

    options = {"triton.cudagraphs": False, "epilogue_fusion": True, "max_autotune": True}
    D = graph(*graph_args)
    graph_opt = torch.compile(graph, options=options)
    D_opt = graph_opt(*graph_args)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=D_opt.cpu().numpy(), desired=D.cpu().numpy(), rtol=1e-2)

    torch_mean, torch_min, torch_max = bench(graph_opt, graph_args)
    print(f"baseline(torch.compile mode=max-autotune): {torch_mean} ms")

    # hidet.option.cache_dir("./matmul_standalone")
    hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)
    hidet.torch.dynamo_config.reset()
    hidet.torch.dynamo_config.parallel_k(strategy="disabled")

    D = graph(*graph_args)
    dynamo.reset()
    graph_hidet = torch.compile(graph, backend="hidet", mode="max-autotune-no-cudagraphs")
    D_hidet = graph_hidet(*graph_args)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    # change the tolarence because we use fp16 as accumulators
    np.testing.assert_allclose(actual=D_hidet.cpu().numpy(), desired=D.cpu().numpy(), rtol=5e-2)
    hidet_mean, hidet_min, hidet_max = bench(graph_hidet, graph_args)
    print(f"hidet(torch.compile): {hidet_mean} ms")
    return torch_mean, hidet_mean


def main():
    from tabulate import tabulate

    records = []
    headers = ["problem(m,n,k,l)", "max-autotune", "hidet", "speedup"]

    for problem in matmul_tests:
        torch_time, hidet_time = test_problem(*problem)
        records.append([problem, torch_time, hidet_time, (torch_time - hidet_time) / torch_time * 100.0])

    with open("results_matmul.txt", "w") as f:
        f.write(
            tabulate(records, headers=headers, tablefmt="github", floatfmt=".3f", numalign="right", stralign="left")
        )


if __name__ == "__main__":
    main()
