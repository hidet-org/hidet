from typing import List, Optional
from hidet import Tensor
from .manual_kernels.gemm_mma_fp16 import gemm_mma_fp16
from .common import BenchResult


def bench_manual(args, out_dir: str) -> BenchResult:
    if args.model.startswith('op_gemm_') and args.precision == 'f16' and args.reduce_precision == 'f16':
        _, _, m, n, k = args.model.split('_')
        m, n, k = int(m), int(n), int(k)
        return gemm_mma_fp16(args.bs, m, n, k, args.warmup, args.number, args.repeat)
    else:
        raise NotImplementedError()
