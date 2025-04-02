import hidet
import torch
import pytest

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


@pytest.mark.parametrize("problem_size", [[1, 4096, 4096], [1, 4096, 12288], [1, 12288, 4096]])
@pytest.mark.parametrize("with_zp", [True, False])
def test_quant_linear(problem_size, with_zp):
    gsize = 128
    m, n, k = problem_size
    print(f"m, n, k: {m}, {n}, {k}")
    a, b, scale, zeros = data(m, n, k, group_size=gsize, return_hidet=False)
    bq = cast_f16_to_u4(b)
    bdq = cast_u4_to_f16(bq)
    linear = w4a16_linear("w4a16", k, n, gsize, with_zp=with_zp)
    bq4 = preprocess_weight(bq)
    from hidet.ffi import runtime_api

    runtime_api.set_symbol_value('m', m)

    def fn():
        c = linear(a, bq4, scale, zeros)
        return c

    c = linear(a, bq4, scale, zeros)
    b2 = b.view(k // gsize, gsize, n)
    if with_zp:
        b3 = scale.view(k // gsize, 1, n) * (b2 - zeros.view(k // gsize, 1, n))
    else:
        b3 = scale.view(k // gsize, 1, n) * b2
    b = b3.view(k, n)
    c2 = a @ b
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=2e-3)
