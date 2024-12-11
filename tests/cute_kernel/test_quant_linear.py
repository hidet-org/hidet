import hidet
import torch

from quantized_linear import w4a16_linear, cast_u4_to_f16, cast_f16_to_u4, preprocess_weight


@torch.no_grad()
def bench(functor, args):
    warmup_iters = 10
    bench_iters = 400

    for _ in range(warmup_iters):
        functor(*args)

    latencies = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(bench_iters):
        functor(*args)

    end.record()
    end.synchronize()
    latencies.append(start.elapsed_time(end) / bench_iters)

    mean = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    return mean, min_lat, max_lat


def data(M, N, K, group_size=64, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    a = torch.randint(low=lo, high=hi, size=(M, K), dtype=dtype, device=device)
    b = torch.randint(low=0, high=3, size=(K, N), dtype=dtype, device=device)
    scale = torch.randint(low=0, high=2, size=(K // group_size, N), dtype=dtype, device=device)
    zeros = torch.randint(low=0, high=3, size=(K // group_size, N), dtype=dtype, device=device)

    ret = [a, b, scale, zeros]

    if return_hidet:
        ret = [hidet.from_torch(x) for x in ret]

    return tuple(ret)


if __name__ == "__main__":
    hidet.option.cache_dir("./demo_quant_linear")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    problem_sizes = [
        [1, 4096, 4096],
        #    [1, 4096, 12288],
        #    [1, 12288, 4096],
        #    [1, 4096, 22016],
        #    [1, 22016, 4096],
        #    [1, 11008, 4096],
        #    [1, 4096, 11008],
        #    [1, 5120, 5120],
        #    [1, 5120, 15360],
        #    [1, 15360, 5120],
        #    [1, 5120, 27648],
        #    [1, 27648, 5120],
        #    [1, 13824, 5120],
        #    [1, 5120, 13824],
        #    [1, 2*10752, 4096],
        #    [1, 4096, 10752],
        #    [8, 4096, 4096 * 3],
        #    [8, 4096 * 3, 4096],
        #    [8, 4096, 4096 * 4],
        #    [8, 4096 * 4, 4096],
        #    [16, 4096, 4096 * 3],
        #    [16, 4096 * 3, 4096],
        #    [32, 4096, 4096*3],
        #    [32, 4096 * 3, 4096],
        #    [1, 4096, 4096*3],
        #    [7, 4096*3, 4096],
        #    [5, 4096*3, 4096],
        #    [1, 4096*3, 4096],
    ]

    gsize = 128
    for m, n, k in problem_sizes:
        print(f"m, n, k: {m}, {n}, {k}")
        a, b, scale, zeros = data(m, n, k, group_size=gsize, return_hidet=False)
        bq = cast_f16_to_u4(b)
        bdq = cast_u4_to_f16(bq)
        linear = w4a16_linear("w4a16", k, n, gsize)
        bq4 = preprocess_weight(bq)
        from hidet.ffi import runtime_api

        runtime_api.set_symbol_value('m', m)

        def fn():
            c = linear(a, bq4, scale, zeros)
            return c

        from hidet.utils.benchmark import do_bench

        mean = do_bench(fn, percentiles=None)
        #        mean, _, _ = bench(linear, (a, bq4, scale, zeros))
        print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))
        print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, (m * n * 2 + m * k * 2 + k * n // 2) / (1e6 * mean)))

        c = linear(a, bq4, scale, zeros)
        # a = a.torch()
        # b = b.torch()
        # scale = scale.torch()
        # zeros = zeros.torch()
        b2 = b.view(k // gsize, gsize, n)
        b3 = scale.view(k // gsize, 1, n) * (b2 - zeros.view(k // gsize, 1, n))
        b = b3.view(k, n)
        c2 = a @ b
        import numpy as np

        # np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
        # np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=2e-3)

    #    from hidet.utils.benchmark import do_bench
    #
    #    cache = hidet.empty((4096*3//2, 4096), dtype=hidet.float32, device='cuda')
    #
    #    def fn():
    #        b = cache * hidet.float32(0)
    #        return b
    #
    #    mean = do_bench(fn, percentiles=None)
    #    print(
    #        "time={:.3f} ms, bandwidth={:.3f} GB/s".format(
    #            mean, (4096*3//4*4096*2) / (1e6 * mean)
    #        )
    #    )

    for m, n, k in problem_sizes:
        from test_vllm_gemm_quant import bench_vllm

        bench_vllm(m, n, k, gs=64)
        bench_vllm(m, n, k, gs=128)

        # from test_awq_gemm_quant import bench_awq

        # bench_awq(m, n, k, gs=64)
        # bench_awq(m, n, k, gs=128)
