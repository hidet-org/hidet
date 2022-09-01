from typing import List, Optional
from hidet import Tensor
from . import manual_kernels
from bench.common import BenchResult, get_onnx_model, benchmark_run
import hidet


def bench_manual(args, out_dir: str) -> BenchResult:
    onnx_path, input_names, input_tensors = get_onnx_model(name=args.model, batch_size=args.bs, precision=args.precision)
    if args.model.startswith('op_gemm_') and args.precision == 'f16' and args.reduce_precision == 'f16':
        _, _, m, n, k = args.model.split('_')
        bs = args.bs
        m, n, k = int(m), int(n), int(k)
        # func = gemm_mma_fp16_kernel(args.bs, m, n, k)
        if args.manual_config == 'default':
            func = manual_kernels.gemm_mma_fp16_kernel(args.bs, m, n, k)
        elif args.manual_config == 'cp_async':
            func = manual_kernels.gemm_mma_fp16_cp_async_kernel(args.bs, m, n, k)
        elif args.manual_config == 'cp_async_multi_stage':
            func = manual_kernels.gemm_mma_fp16_cp_async_multi_stage_kernel(args.bs, m, n, k)
        elif args.manual_config == 'ldmatrix':
            func = manual_kernels.gemm_mma_fp16_ldmatrix_kernel(args.bs, m, n, k)
        elif args.manual_config == 'cp_async_ldmatrix':
            func = manual_kernels.gemm_mma_fp16_cp_async_ldmatrix_kernel(args.bs, m, n, k)
        elif args.manual_config == 'cp_async_ldmatrix_opt':
            func = manual_kernels.gemm_mma_fp16_cp_async_ldmatrix_opt_kernel(args.bs, m, n, k)
        elif args.manual_config == 'all':
            func = manual_kernels.gemm_mma_fp16_all_kernel(args.bs, m, n, k)
        else:
            raise ValueError(args.manual_config)
        a = input_tensors[0]
        b = input_tensors[1]
        c = hidet.randn([bs, m, n], 'float16')
        run_func = lambda: func(a, b, c)
        return BenchResult(
            latencies=benchmark_run(run_func=run_func, warmup=args.warmup, number=args.number, repeat=args.repeat),
            outputs=[c],
            configs=args.manual_config
        )
    elif args.model.startswith('op_dwc_'):
        from .manual_kernels.depthwise_conv2d import dwc_kernel
        _, _, n, c, h, w, s, k = args.model.split('_')
        n, c, h, w, s, k = int(n), int(c), int(h), int(w), int(s), int(k)
        func = dwc_kernel(n, c, h, w, s, k)
        xx, ww = input_tensors
        yy = hidet.randn([n, c, h, w])

        run_func = lambda: func(xx, ww, yy)
        return BenchResult(
            latencies=benchmark_run(run_func=run_func, warmup=args.warmup, number=args.number, repeat=args.repeat),
            outputs=[yy],
            configs=args.manual_config
        )

    else:
        raise NotImplementedError()
