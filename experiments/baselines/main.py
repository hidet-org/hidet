import numpy as np
from hidet.runtime.value import randn, empty, scalar
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass


def main():
    repeat = 10
    workloads = [
        (1024, 1024, 1024),
        (1600, 768, 2304)
    ]
    baselines = [
        ('Reference', matmul_ref()),
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cuBLAS', matmul_cublas())
    ]
    print('Repeat = {}'.format(repeat))
    print()
    for N, M, K in workloads:
        A = randn([N, K], 'float32', 'global', seed=1)
        B = randn([K, M], 'float32', 'global', seed=3)
        C = empty([N, M], 'float32', 'global')
        print("Workload (N x M x K): {} x {} x {}".format(N, M, K))
        for name, func in baselines:
            latencies = func.profile(scalar(N), scalar(M), scalar(K), A, B, C, repeat=repeat)
            print('{:>13}: {:.3f} (std {:.3f}) ms'.format(name, np.mean(latencies), np.std(latencies)))
        print()


if __name__ == '__main__':
    main()
