import pytest
import argparse

import hidet
from gemm_quant_nbit import FpAIntBGemm
from hidet.utils.py import cdiv

import torch

from quant_utils import bench, data


@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "problem_size",
    (
        (8, 4096, 4096 * 3),
        (8, 4096 * 3, 4096),
        (16, 4096, 4096 * 3),
        (16, 4096 * 3, 4096),
        (32, 4096, 4096 * 3),
        (32, 4096 * 3, 4096),
    ),
)
@pytest.mark.parametrize("nbit", (1, 2, 3))
def test_gemm_quant_nbit(problem_size, nbit):
    hidet.option.search_space(2)

    m, n, k = problem_size
    group_size = 128

    a, b2, b1, _, bq2, bq1, bdq, scale, bias = data(m, n, k, return_hidet=True)

    linear = FpAIntBGemm(m, k, n, group_size, nbit)

    modules = linear.modules()
    quant_module = modules[0]
    module = quant_module.gemm_quant
    qmod_2b = quant_module.quant[0]
    qmod_1b = quant_module.quant[1]

    func = module.build()
    qfunc_2b = qmod_2b.build()
    qfunc_1b = qmod_1b.build()
    qfunc_2b(b2, bq2)
    qfunc_1b(b1, bq1)

    tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
    parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
    print(quant_module._tuning_kwargs)
    c_shape, _ = tiled_mma.c_tv_layout()
    block_m, block_n = c_shape
    gridm = cdiv(m, block_m)
    gridn = cdiv(n, block_n)

    def fn():
        c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float16, device="cuda")
        c = torch.empty((m, n), dtype=torch.float16, device="cuda")
        counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
        func(a, bq2, bq1, c, scale, bias, c_parallel_k_parts, counters)
        return c

    c = fn()
    if nbit == 3:
        b = b2 * 2.0 + b1
    elif nbit == 2:
        b = b2 * 2.0
    else:
        b = b1
    a = a.torch()
    b = b.torch()
    c2 = a @ b
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=5e-3)


def bench_3bit():
    hidet.option.cache_dir("./demo_gemm_quant_3bit")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        [8, 4096, 4096 * 3],
        [8, 4096 * 3, 4096],
        [16, 4096, 4096 * 3],
        [16, 4096 * 3, 4096],
        [32, 4096, 4096 * 3],
        [32, 4096 * 3, 4096],
    ]

    group_size = 128
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b2, b1, _, bq2, bq1, bdq, scale, bias = data(m, n, k, return_hidet=True)
        b = b2 * 2.0 + b1
        b = b.torch()

        linear = FpAIntBGemm(m, k, n, group_size, 3)

        best_time = None
        best_module = None
        modules = linear.modules()
        for quant_module in modules:
            module = quant_module.gemm_quant
            qmod_2b = quant_module.quant[0]
            qmod_1b = quant_module.quant[1]

            func = module.build()
            qfunc_2b = qmod_2b.build()
            qfunc_1b = qmod_1b.build()
            qfunc_2b(b2, bq2)
            qfunc_1b(b1, bq1)

            tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
            parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
            print(quant_module._tuning_kwargs)
            c_shape, _ = tiled_mma.c_tv_layout()
            block_m, block_n = c_shape
            gridm = cdiv(m, block_m)
            gridn = cdiv(n, block_n)

            def fn():
                c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float16, device="cuda")
                c = torch.empty((m, n), dtype=torch.float16, device="cuda")
                counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
                func(a, bq2, bq1, c, scale, bias, c_parallel_k_parts, counters)
                return c

            mean, min_lat, max_lat = bench(fn, ())
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            from hidet.utils.benchmark import do_bench

            mean = do_bench(fn, percentiles=None)
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            if best_time is None:
                best_time = mean
                best_module = module
            elif mean < best_time:
                best_time = mean
                best_module = module

            c = fn()
            c2 = a.torch() @ b
            import numpy as np

            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=5e-3)
        print(f"m, n, k: {m}, {n}, {k}")
        print(best_time)
        print(best_module)


def bench_2bit():
    hidet.option.cache_dir("./demo_gemm_quant_2bit")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        [8, 4096, 4096 * 3],
        [8, 4096 * 3, 4096],
        [16, 4096, 4096 * 3],
        [16, 4096 * 3, 4096],
        [32, 4096, 4096 * 3],
        [32, 4096 * 3, 4096],
    ]

    group_size = 128
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b2, b1, _, bq2, bq1, bdq, scale, bias = data(m, n, k, return_hidet=True)
        b = b2 * 2.0
        b = b.torch()

        linear = FpAIntBGemm(m, k, n, group_size, 2)

        best_time = None
        best_module = None
        modules = linear.modules()
        for quant_module in modules:
            module = quant_module.gemm_quant
            qmod_2b = quant_module.quant[0]
            qmod_1b = quant_module.quant[1]

            func = module.build()
            qfunc_2b = qmod_2b.build()
            qfunc_1b = qmod_1b.build()
            qfunc_2b(b2, bq2)
            qfunc_1b(b1, bq1)

            tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
            parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
            print(quant_module._tuning_kwargs)
            c_shape, _ = tiled_mma.c_tv_layout()
            block_m, block_n = c_shape
            gridm = cdiv(m, block_m)
            gridn = cdiv(n, block_n)

            def fn():
                c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float16, device="cuda")
                c = torch.empty((m, n), dtype=torch.float16, device="cuda")
                counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
                func(a, bq2, bq1, c, scale, bias, c_parallel_k_parts, counters)
                return c

            mean, min_lat, max_lat = bench(fn, ())
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            from hidet.utils.benchmark import do_bench

            mean = do_bench(fn, percentiles=None)
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            if best_time is None:
                best_time = mean
                best_module = module
            elif mean < best_time:
                best_time = mean
                best_module = module

            c = fn()
            c2 = a.torch() @ b
            import numpy as np

            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=5e-3)
        print(f"m, n, k: {m}, {n}, {k}")
        print(best_time)
        print(best_module)


def bench_1bit():
    hidet.option.cache_dir("./demo_gemm_quant_1bit")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        [8, 4096, 4096 * 3],
        [8, 4096 * 3, 4096],
        [16, 4096, 4096 * 3],
        [16, 4096 * 3, 4096],
        [32, 4096, 4096 * 3],
        [32, 4096 * 3, 4096],
    ]

    group_size = 128
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b2, b1, _, bq2, bq1, bdq, scale, bias = data(m, n, k, return_hidet=True)
        b = b1
        b = b.torch()

        linear = FpAIntBGemm(m, k, n, group_size, 1)

        best_time = None
        best_module = None
        modules = linear.modules()
        for quant_module in modules:
            module = quant_module.gemm_quant
            qmod_2b = quant_module.quant[0]
            qmod_1b = quant_module.quant[1]

            func = module.build()
            qfunc_2b = qmod_2b.build()
            qfunc_1b = qmod_1b.build()
            qfunc_2b(b2, bq2)
            qfunc_1b(b1, bq1)

            tiled_mma = quant_module._tuning_kwargs["tiled_mma"]
            parallel_k_parts = quant_module._tuning_kwargs["parallel_k_parts"]
            print(quant_module._tuning_kwargs)
            c_shape, _ = tiled_mma.c_tv_layout()
            block_m, block_n = c_shape
            gridm = cdiv(m, block_m)
            gridn = cdiv(n, block_n)

            def fn():
                c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float16, device="cuda")
                c = torch.empty((m, n), dtype=torch.float16, device="cuda")
                counters = torch.empty((gridm, gridn), dtype=torch.int32, device="cuda")
                func(a, bq2, bq1, c, scale, bias, c_parallel_k_parts, counters)
                return c

            mean, min_lat, max_lat = bench(fn, ())
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            from hidet.utils.benchmark import do_bench

            mean = do_bench(fn, percentiles=None)
            print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
            print(
                "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
                    mean, (m * n * 2 + m * k * 2 + 3 * k * n // 8) / (1e6 * mean)
                )
            )

            if best_time is None:
                best_time = mean
                best_module = module
            elif mean < best_time:
                best_time = mean
                best_module = module

            c = fn()
            c2 = a.torch() @ b
            import numpy as np

            np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
            np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=5e-3)
        print(f"m, n, k: {m}, {n}, {k}")
        print(best_time)
        print(best_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mixed input GEMM")
    parser.add_argument("--bit_width", type=int, default=3)

    args = parser.parse_args()
    if args.bit_width == 3:
        bench_3bit()
    elif args.bit_width == 2:
        bench_2bit()
    elif args.bit_width == 1:
        bench_1bit()
