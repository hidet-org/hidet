from __future__ import annotations
from typing import List, Dict, Optional
from hidet.ir.mapping import TaskMapping, row_spatial, col_spatial, repeat_map, row_repeat, col_repeat
from hidet.utils import initialize
from hidet.ir.type import ScalarType
from hidet.ir.expr import Var, Expr, Call, cast
from hidet.ir.stmt import AsmStmt, AssignStmt
from hidet.ir.dialects.lowlevel import PointerType, VoidType, void_pointer
from hidet.ir.builders import FunctionBuilder
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda


def resolve_name_cp_async(cp_size: int, cache_level: str = 'always', prefetch_bytes: int = 0) -> str:
    if prefetch_bytes:
        prefetch_part = '_l2_{}B'.format(prefetch_bytes)
    else:
        prefetch_part = ''
    cache_part = 'c' + cache_level[0]   # 'ca' or 'cg'
    return 'cp_async_size_{}_{}{}'.format(cp_size, cache_part, prefetch_part)


def resolve_name_async_wait_group() -> str:
    pass


@initialize()
def register_cp_async():
    for cp_size in [4, 8, 16]:
        for prefetch_bytes in [0, 64, 128, 256]:
            for cache_level in ['always', 'global']:
                if cache_level == 'global' and cp_size != 16:
                    continue
                func_name = resolve_name_cp_async(cp_size, cache_level, prefetch_bytes)
                with FunctionBuilder(name=func_name, kind='cuda_device') as fb:
                    dst = Var('dst', PointerType(VoidType()))
                    src = Var('src', PointerType(VoidType()))
                    src_size = Var('src_size', ScalarType('int32'))
                    fb.extend_params([dst, src, src_size])
                    template_string = 'cp.async.{}.shared.global{} [%0], [%1], %2, %3'.format(
                        {'always': 'ca', 'global': 'cg'}[cache_level],
                        '.L2::{}B'.format(prefetch_bytes) if prefetch_bytes else '',
                        cp_size
                    )
                    fb += AsmStmt(
                        template_string=template_string,
                        outputs=[],
                        inputs=[]
                    )


@initialize()
def register_cp_async_commit_group():
    pass


@initialize()
def register_cp_async_wait_group():
    pass


@initialize()
def register_cp_async_wait_all():
    pass


def cp_async(dst: Expr, src: Expr, cp_size: int, src_size: Optional[Expr], cache_level: str = 'always', prefetch_bytes: int = 0) -> Call:
    """
    Copy data from global memory to shared memory asynchronously.

    See also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    Parameters
    ----------
    dst: Expr
        The address of the destination in shared memory.
    src: Expr
        The address of the source in global memory.
    cp_size: int
        The number of bytes to be copied to the destination. Candidates: 4, 8 and 16.
    src_size: Optional[Expr]
        The number of bytes in the source to be copied. If src_size < cp_size, the remaining part of destination will be filled with 0.
    cache_level: str
        The cache level. Candidates: 'always' and 'global'. When cache_level is 'global', the cp_size must be 16.
    prefetch_bytes: int
        The number of bytes to be prefetched in L2 cache. Candidates: 0, 64, 128, 256.

    Returns
    -------
    ret: Call
        The call expression.
    """
    if not (isinstance(cp_size, int) and cp_size in [4, 8, 16]):
        raise ValueError('cp_size must be either 4, 8, or 16, got {}.'.format(cp_size))
    if not (isinstance(prefetch_bytes, int) and prefetch_bytes in [0, 64, 128, 256]):
        raise ValueError('prefetch_bytes must be either None, 64, 128 or 256, got {}.'.format(prefetch_bytes))
    if cache_level not in ['global', 'always']:
        raise ValueError('Cache level candidates: {}, got {}'.format(['always', 'global'], cache_level))
    if cache_level == 'global':
        if cp_size != 16:
            raise ValueError('When cache_level is global, the cp_size must be 16.')
    if src_size is None:
        src_size = cp_size
    func_name = resolve_name_cp_async(cp_size, cache_level, prefetch_bytes)
    return call_cuda(func_name, [src, dst, src_size])
