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
# pylint: disable=line-too-long
from typing import Union
from hidet.utils import initialize
from hidet.ir.type import PointerType, VoidType
from hidet.ir.expr import Expr, Call
from hidet.ir.stmt import asm
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda


def resolve_name_cp_async(cp_size: int, cache_level: str = 'always', prefetch_bytes: int = 0) -> str:
    if prefetch_bytes:
        prefetch_part = '_l2_{}B'.format(prefetch_bytes)
    else:
        prefetch_part = ''
    cache_part = 'c' + cache_level[0]  # 'ca' or 'cg'
    return 'cp_async_size_{}_{}{}'.format(cp_size, cache_part, prefetch_part)


def resolve_name_async_wait_group() -> str:
    pass


@initialize()
def register_cp_async():
    from hidet.lang import script, i32, attr
    from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

    for cp_size in [4, 8, 16]:
        for prefetch_bytes in [0, 64, 128, 256]:
            for cache_level in ['always', 'global']:
                if cache_level == 'global' and cp_size != 16:
                    # cache level 'global' only support copy size of 16 bytes.
                    continue
                func_name = 'cuda_' + resolve_name_cp_async(cp_size, cache_level, prefetch_bytes)
                template_string = 'cp.async.{cache_level}.shared.global{prefetch} [%0], [%1], %2, %3;'.format(
                    cache_level={'always': 'ca', 'global': 'cg'}[cache_level],
                    prefetch='.L2::{}B'.format(prefetch_bytes) if prefetch_bytes != 0 else '',
                )

                @script
                def cuda_cp_async(dst: PointerType(VoidType()), src: PointerType(VoidType()), src_size: i32):
                    attr.func_name = func_name
                    attr.func_kind = 'cuda_device'
                    dst_smem_ptr = cvta_generic_to_shared(dst)
                    asm(template=template_string, inputs=[dst_smem_ptr, src, cp_size, src_size])

                assert isinstance(cuda_cp_async, Function)
                register_primitive_function(name=cuda_cp_async.name, func_or_type=cuda_cp_async)


@initialize()
def register_cp_async_commit_group():
    from hidet.lang import script, attr

    @script
    def cuda_cp_async_commit_group():
        attr.func_name = 'cuda_cp_async_commit_group'
        attr.func_kind = 'cuda_device'
        asm('cp.async.commit_group;')

    assert isinstance(cuda_cp_async_commit_group, Function)
    register_primitive_function(cuda_cp_async_commit_group.name, cuda_cp_async_commit_group)


@initialize()
def register_cp_async_wait_group():
    from hidet.lang import script, attr

    for groups in range(10):
        func_name = 'cuda_cp_async_wait_group_{}'.format(groups)

        @script
        def cuda_cp_async_wait_group():
            attr.func_name = func_name
            attr.func_kind = 'cuda_device'
            asm('cp.async.wait_group {};'.format(groups))

        assert isinstance(cuda_cp_async_wait_group, Function)
        register_primitive_function(cuda_cp_async_wait_group.name, cuda_cp_async_wait_group)


@initialize()
def register_cp_async_wait_all():
    from hidet.lang import script, attr

    @script
    def cuda_cp_async_wait_all():
        attr.func_name = 'cuda_cp_async_wait_all'
        attr.func_kind = 'cuda_device'
        asm('cp.async.wait_all;')

    assert isinstance(cuda_cp_async_wait_all, Function)
    register_primitive_function(cuda_cp_async_wait_all.name, cuda_cp_async_wait_all)


def cp_async(dst: Expr, src: Expr, cp_size: int, src_size=None, cache_level='always', prefetch_bytes=0) -> Call:
    """
    Copy data from global memory to shared memory asynchronously.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    Parameters
    ----------
    dst: Expr
        The address of the destination in shared memory.
    src: Expr
        The address of the source in global memory.
    cp_size: int
        The number of bytes to be copied to the destination. Candidates: 4, 8 and 16.
    src_size: Optional[Union[Expr, int]]
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
    return call_cuda(func_name, [dst, src, src_size])


def cp_async_commit_group():
    """
    Commit all prior issued cp_async into a group.

    See Also
        https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group
    """
    return call_cuda('cp_async_commit_group', [])


def cp_async_wait_group(allow_on_fly_groups: Union[int, Expr]):
    """
    Wait the completion of prior asynchronous copy operations.

    See Also
       https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group

    Parameters
    ----------
    allow_on_fly_groups: Union[int, Expr]
        The maximum number of asynchronous copies that are allowed to be on-the-fly after this function.
        Can be a python integer or a hidet constant expression.
    """
    if isinstance(allow_on_fly_groups, Expr):
        from hidet.ir.functors.simplifier import simplify_to_int

        allow_on_fly_groups = simplify_to_int(allow_on_fly_groups)
    if not 0 <= allow_on_fly_groups < 10:
        raise ValueError('n out of bound')
    return call_cuda('cp_async_wait_group_{}'.format(allow_on_fly_groups), [])


def cp_async_wait_all():
    """
    Wait all prior asynchronous copy operations.

    See Also
       https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group
    """
    return call_cuda('cp_async_wait_all', [])
