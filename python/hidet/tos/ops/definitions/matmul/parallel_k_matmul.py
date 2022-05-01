import warnings
from .matmul import matmul, Tensor, cuda
from hidet.utils.py import gcd, factor


def parallel_k_nparts(batch_size, m_size, n_size, k_size) -> int:
    estimate_thread_blocks = batch_size * ((m_size + 63) // 64) * ((n_size + 63) // 64)
    num_multi_processors = cuda.device_property(cuda.PropertyMultiProcessorCount)
    # we hope to run multiple waves of thread blocks (e.g., 5)
    if estimate_thread_blocks * 8 <= num_multi_processors * 5:
        nparts = 8
    elif estimate_thread_blocks * 4 <= num_multi_processors * 5:
        nparts = 4
    elif estimate_thread_blocks * 2 <= num_multi_processors * 5:
        nparts = 2
    else:
        nparts = 1
    return nparts


def parallel_k_batched_matmul(a: Tensor, b: Tensor, mma: str = 'default', nparts=None) -> Tensor:
    k_size = a.shape[-1]
    batch_size, m_size, n_size = a.shape[0], a.shape[1], b.shape[2]

    if nparts is None:
        nparts = parallel_k_nparts(batch_size, m_size, n_size, k_size)

    nparts = gcd(nparts, k_size)

    print('parallel_k of batched matmul {}x{}x{}x{} used factor {}'.format(batch_size, m_size, n_size, k_size, nparts))

    if nparts == 1:
        # warnings.warn('Parallel k matmul use nparts=1, fall back to direct matmul.')
        return matmul(a, b, algo='direct', mma=mma)
    else:
        a = a.reshape([batch_size, m_size, nparts, k_size // nparts]).rearrange([[0, 2], [1], [3]])  # [batch_size * nparts, m_size, k_size // nparts]
        b = b.reshape([batch_size, nparts, k_size // nparts, n_size]).rearrange([[0, 1], [2], [3]])  # [batch_size * nparts, k_size // nparts, n_size]
        c = matmul(a, b, algo='direct', mma=mma).reshape([batch_size, nparts, m_size, n_size]).sum(1)
        return c

    # if nparts is None:
    #     if use_parallel_k(batch_size, m_size, n_size, k_size):
    #         if nparts is None:
    #             nparts = parallel_k_nparts(batch_size, m_size, n_size, k_size)
    #         else:
    #             nparts = gcd(nparts, k_size)
    #         if nparts == 1:
    #             return batched_matmul(a, b, algo='direct', mma=mma)
    #         else:
    #     else:
    #         warnings.warn('Please use use_parallel_k to check whether we should use parallel_k matmul first. Falling back to direct algorithm.')
    #         return batched_matmul(a, b, algo='direct', mma=mma)
    #     pass
    # else:
    #


def parallel_k_batched_matmul_search(a: Tensor, b: Tensor, mma: str = 'default') -> Tensor:
    import numpy as np
    k_size = a.shape[-1]
    batch_size, m_size, n_size = a.shape[0], a.shape[1], b.shape[2]

    factors = [v for v in factor(k_size) if v <= 16]

    best_nparts = None
    best_nparts_latency = 1e9

    if len(factors) > 1:
        print('searching batch_matmul {}x{}x{}x{} parallel k factors: {}'.format(batch_size, m_size, n_size, k_size, factors))
        candidate_latencies = []
        for nparts in factors:
            num_trials = 1000
            if nparts == 1:
                c = matmul(a, b, algo='direct', mma=mma)
                latency = float(np.median(c.op.latency(number=num_trials)))
                if latency < best_nparts_latency:
                    best_nparts = nparts
                    best_nparts_latency = latency
                candidate_latencies.append(latency)
            else:
                aa = a.reshape([batch_size, m_size, nparts, k_size // nparts]).rearrange([[0, 2], [1], [3]])  # [batch_size * nparts, m_size, k_size // nparts]
                bb = b.reshape([batch_size, nparts, k_size // nparts, n_size]).rearrange([[0, 1], [2], [3]])  # [batch_size * nparts, k_size // nparts, n_size]
                cc = matmul(aa, bb, algo='direct', mma=mma)
                c1 = cc.reshape([batch_size, nparts, m_size, n_size])
                c2 = c1.sum(1)
                latency = cc.op.latency(number=num_trials) + c2.op.latency(number=num_trials)
                if latency < best_nparts_latency:
                    best_nparts = nparts
                    best_nparts_latency = latency
                candidate_latencies.append(latency)
        print('candidate latencies: {}, choose factor {}'.format(['{:.3f}'.format(v * 1000) for v in candidate_latencies], best_nparts))
    else:
        assert len(factors) == 1
        best_nparts = factors[0]
    return parallel_k_batched_matmul(a, b, mma, best_nparts)
