from hidet.ir.expr import Expr
from hidet.ir.stmt import BlackBoxStmt


def memcpy_async(dst: Expr, src: Expr, count: Expr, kind: str):
    from hidet.ir.primitives.runtime import get_cuda_stream

    kind_map = {
        'cpu_to_cpu': 'cudaMemcpyHostToHost',
        'cpu_to_cuda': 'cudaMemcpyHostToDevice',
        'cuda_to_cpu': 'cudaMemcpyDeviceToHost',
        'cuda_to_cuda': 'cudaMemcpyDeviceToDevice',
    }

    if kind not in kind_map:
        raise RuntimeError(f'Unsupported transfer from {src} to {dst}, candidate kinds are {list(kind_map.keys())}')

    return BlackBoxStmt(
        'cudaMemcpyAsync({}, {}, {}, {}, (cudaStream_t){});'.format(dst, src, count, kind_map[kind], get_cuda_stream())
    )
