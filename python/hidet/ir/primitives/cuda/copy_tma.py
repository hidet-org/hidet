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
from typing import Union, Optional
from hidet.utils import initialize
from hidet.ir.type import OpaqueType, PointerType, VoidType
from hidet.ir.expr import Expr, cast
from hidet.ir.stmt import asm
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda


@initialize()
def register_copy_bulk():
    from hidet.lang import script, u16, i32, u64, attrs
    from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

    # G2S
    # cache hint not enabled now
    func_name = 'cuda_copy_bulk_g2s_multicast'
    template_string = 'cp.async.bulk.{dst}.{src}.{completion_mechanism}{multicast} [%0], [%1], %2, [%3], %4;'.format(
        src='global',
        dst='shared::cluster',
        completion_mechanism='mbarrier::complete_tx::bytes',
        multicast='.multicast::cluster',
    )

    @script
    def cuda_copy_bulk_g2s_multicast(
        dst: PointerType(VoidType()), src: PointerType(VoidType()), size: i32, mbar: ~u64, multicastmask: u16
    ):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_int_ptr = cvta_generic_to_shared(dst)
        smem_int_mbar = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_int_ptr, src, size, smem_int_mbar, multicastmask])

    assert isinstance(cuda_copy_bulk_g2s_multicast, Function)
    register_primitive_function(name=cuda_copy_bulk_g2s_multicast.name, func_or_type=cuda_copy_bulk_g2s_multicast)

    func_name = 'cuda_copy_bulk_g2s'
    template_string = 'cp.async.bulk.{dst}.{src}.{completion_mechanism} [%0], [%1], %2, [%3];'.format(
        src='global', dst='shared::cluster', completion_mechanism='mbarrier::complete_tx::bytes'
    )

    @script
    def cuda_copy_bulk_g2s(dst: PointerType(VoidType()), src: PointerType(VoidType()), size: i32, mbar: ~u64):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_int_ptr = cvta_generic_to_shared(dst)
        smem_int_mbar = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_int_ptr, src, size, smem_int_mbar])

    assert isinstance(cuda_copy_bulk_g2s, Function)
    register_primitive_function(name=cuda_copy_bulk_g2s.name, func_or_type=cuda_copy_bulk_g2s)

    # S2S
    func_name = 'cuda_copy_bulk_s2s'
    template_string = 'cp.async.bulk.{dst}.{src}.{completion_mechanism} [%0], [%1], %2, [%3];'.format(
        src='shared::cta', dst='shared::cluster', completion_mechanism='mbarrier::complete_tx::bytes'
    )

    @script
    def cuda_copy_bulk_s2s(dst: PointerType(VoidType()), src: PointerType(VoidType()), size: i32, mbar: ~u64):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        dst_smem_int_ptr = cvta_generic_to_shared(dst)
        src_smem_int_ptr = cvta_generic_to_shared(src)
        smem_int_mbar = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[dst_smem_int_ptr, src_smem_int_ptr, size, smem_int_mbar])

    assert isinstance(cuda_copy_bulk_s2s, Function)
    register_primitive_function(name=cuda_copy_bulk_s2s.name, func_or_type=cuda_copy_bulk_s2s)

    # S2G
    func_name = 'cuda_copy_bulk_s2g'
    template_string = 'cp.async.bulk.{dst}.{src}.{completion_mechanism} [%0], [%1], %2;'.format(
        src='shared::cta', dst='global', completion_mechanism='bulk_group'
    )

    @script
    def cuda_copy_bulk_s2g(dst: PointerType(VoidType()), src: PointerType(VoidType()), size: i32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_int_ptr = cvta_generic_to_shared(src)
        asm(template=template_string, inputs=[dst, smem_int_ptr, size])

    assert isinstance(cuda_copy_bulk_s2g, Function)
    register_primitive_function(name=cuda_copy_bulk_s2g.name, func_or_type=cuda_copy_bulk_s2g)

    # prefetch
    func_name = 'cuda_copy_bulk_prefetch'
    template_string = 'cp.async.bulk.prefetch.L2.{src} [%0], %1;'.format(src='shared::cta')

    @script
    def cuda_copy_bulk_prefetch(src: PointerType(VoidType()), size: i32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string, inputs=[src, size])

    assert isinstance(cuda_copy_bulk_prefetch, Function)
    register_primitive_function(name=cuda_copy_bulk_prefetch.name, func_or_type=cuda_copy_bulk_prefetch)


@initialize()
def register_copy_tensor_nd():
    from hidet.lang import script, u16, i32, u64, attrs
    from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

    tensor_map_type = OpaqueType("CUtensorMap", "const")

    # G2S
    # multicast
    for dim in [1, 2, 3, 4, 5]:
        for load_mode in ['tile']:
            func_name = f"cuda_copy_tma_{dim}d_g2s_multicast"
            tensor_coords = ', '.join(f'%{i}' for i in range(4, 4 + dim))
            template_string = 'cp.async.bulk.tensor.{dim}.{dst}.{src}.{load_mode}.{completion_mechanism}{multicast} [%0], [%1, {{{tensor_coords}}}], [%2], %3;'.format(
                src='global',
                dst='shared::cluster',
                dim=f'{dim}d',
                completion_mechanism='mbarrier::complete_tx::bytes',
                load_mode=load_mode,
                multicast='.multicast::cluster',
                tensor_coords=tensor_coords,
            )

            if dim == 1:

                @script
                def cuda_copy_tensor_g2s_multicast_1d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    multicastmask: u16,
                    crd0: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(
                        template=template_string,
                        inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, multicastmask, crd0],
                    )

                assert isinstance(cuda_copy_tensor_g2s_multicast_1d, Function)
                register_primitive_function(
                    name=cuda_copy_tensor_g2s_multicast_1d.name, func_or_type=cuda_copy_tensor_g2s_multicast_1d
                )
            elif dim == 2:

                @script
                def cuda_copy_tensor_g2s_multicast_2d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    multicastmask: u16,
                    crd0: i32,
                    crd1: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(
                        template=template_string,
                        inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, multicastmask, crd0, crd1],
                    )

                assert isinstance(cuda_copy_tensor_g2s_multicast_2d, Function)
                register_primitive_function(
                    name=cuda_copy_tensor_g2s_multicast_2d.name, func_or_type=cuda_copy_tensor_g2s_multicast_2d
                )
            elif dim == 3:

                @script
                def cuda_copy_tensor_g2s_multicast_3d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    multicastmask: u16,
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(
                        template=template_string,
                        inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, multicastmask, crd0, crd1, crd2],
                    )

                assert isinstance(cuda_copy_tensor_g2s_multicast_3d, Function)
                register_primitive_function(
                    name=cuda_copy_tensor_g2s_multicast_3d.name, func_or_type=cuda_copy_tensor_g2s_multicast_3d
                )
            elif dim == 4:

                @script
                def cuda_copy_tensor_g2s_multicast_4d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    multicastmask: u16,
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                    crd3: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(
                        template=template_string,
                        inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, multicastmask, crd0, crd1, crd2, crd3],
                    )

                assert isinstance(cuda_copy_tensor_g2s_multicast_4d, Function)
                register_primitive_function(
                    name=cuda_copy_tensor_g2s_multicast_4d.name, func_or_type=cuda_copy_tensor_g2s_multicast_4d
                )
            elif dim == 5:

                @script
                def cuda_copy_tensor_g2s_multicast_5d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    multicastmask: u16,
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                    crd3: i32,
                    crd4: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(
                        template=template_string,
                        inputs=[
                            smem_int_ptr,
                            gmem_int_desc,
                            smem_int_mbar,
                            multicastmask,
                            crd0,
                            crd1,
                            crd2,
                            crd3,
                            crd4,
                        ],
                    )

                assert isinstance(cuda_copy_tensor_g2s_multicast_5d, Function)
                register_primitive_function(
                    name=cuda_copy_tensor_g2s_multicast_5d.name, func_or_type=cuda_copy_tensor_g2s_multicast_5d
                )

    # w/o multicast
    for dim in [1, 2, 3, 4, 5]:
        for load_mode in ['tile']:
            func_name = f"cuda_copy_tma_{dim}d_g2s"
            tensor_coords = ', '.join(f'%{i}' for i in range(3, 3 + dim))
            template_string = 'cp.async.bulk.tensor.{dim}.{dst}.{src}.{load_mode}.{completion_mechanism} [%0], [%1, {{{tensor_coords}}}], [%2];'.format(
                src='global',
                dst='shared::cluster',
                dim=f'{dim}d',
                completion_mechanism='mbarrier::complete_tx::bytes',
                load_mode=load_mode,
                tensor_coords=tensor_coords,
            )

            if dim == 1:

                @script
                def cuda_copy_tensor_g2s_1d(
                    dst: PointerType(VoidType()), src: PointerType(tensor_map_type), mbar: ~u64, crd0: i32
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(template=template_string, inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, crd0])

                assert isinstance(cuda_copy_tensor_g2s_1d, Function)
                register_primitive_function(name=cuda_copy_tensor_g2s_1d.name, func_or_type=cuda_copy_tensor_g2s_1d)
            elif dim == 2:

                @script
                def cuda_copy_tensor_g2s_2d(
                    dst: PointerType(VoidType()), src: PointerType(tensor_map_type), mbar: ~u64, crd0: i32, crd1: i32
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(template=template_string, inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, crd0, crd1])

                assert isinstance(cuda_copy_tensor_g2s_2d, Function)
                register_primitive_function(name=cuda_copy_tensor_g2s_2d.name, func_or_type=cuda_copy_tensor_g2s_2d)
            elif dim == 3:

                @script
                def cuda_copy_tensor_g2s_3d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(template=template_string, inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, crd0, crd1, crd2])

                assert isinstance(cuda_copy_tensor_g2s_3d, Function)
                register_primitive_function(name=cuda_copy_tensor_g2s_3d.name, func_or_type=cuda_copy_tensor_g2s_3d)
            elif dim == 4:

                @script
                def cuda_copy_tensor_g2s_4d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                    crd3: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(
                        template=template_string,
                        inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, crd0, crd1, crd2, crd3],
                    )

                assert isinstance(cuda_copy_tensor_g2s_4d, Function)
                register_primitive_function(name=cuda_copy_tensor_g2s_4d.name, func_or_type=cuda_copy_tensor_g2s_4d)
            elif dim == 5:

                @script
                def cuda_copy_tensor_g2s_5d(
                    dst: PointerType(VoidType()),
                    src: PointerType(tensor_map_type),
                    mbar: ~u64,
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                    crd3: i32,
                    crd4: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(src, u64)
                    smem_int_ptr = cvta_generic_to_shared(dst)
                    smem_int_mbar = cvta_generic_to_shared(mbar)
                    asm(
                        template=template_string,
                        inputs=[smem_int_ptr, gmem_int_desc, smem_int_mbar, crd0, crd1, crd2, crd3, crd4],
                    )

                assert isinstance(cuda_copy_tensor_g2s_5d, Function)
                register_primitive_function(name=cuda_copy_tensor_g2s_5d.name, func_or_type=cuda_copy_tensor_g2s_5d)

    # S2G
    for dim in [1, 2, 3, 4, 5]:
        for load_mode in ['tile']:
            func_name = f"cuda_copy_tma_{dim}d_s2g"
            tensor_coords = ', '.join(f'%{i}' for i in range(2, 2 + dim))
            template_string = 'cp.async.bulk.tensor.{dim}.{dst}.{src}.{load_mode}.{completion_mechanism} [%0, {{{tensor_coords}}}], [%1];'.format(
                src='shared::cta',
                dst='global',
                dim=f'{dim}d',
                completion_mechanism='bulk_group',
                load_mode=load_mode,
                tensor_coords=tensor_coords,
            )

            if dim == 1:

                @script
                def cuda_copy_tensor_s2g_1d(dst: PointerType(tensor_map_type), src: PointerType(VoidType()), crd0: i32):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(dst, u64)
                    smem_int_ptr = cvta_generic_to_shared(src)
                    asm(template=template_string, inputs=[gmem_int_desc, smem_int_ptr, crd0])

                assert isinstance(cuda_copy_tensor_s2g_1d, Function)
                register_primitive_function(name=cuda_copy_tensor_s2g_1d.name, func_or_type=cuda_copy_tensor_s2g_1d)
            elif dim == 2:

                @script
                def cuda_copy_tensor_s2g_2d(
                    dst: PointerType(tensor_map_type), src: PointerType(VoidType()), crd0: i32, crd1: i32
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(dst, u64)
                    smem_int_ptr = cvta_generic_to_shared(src)
                    asm(template=template_string, inputs=[gmem_int_desc, smem_int_ptr, crd0, crd1])

                assert isinstance(cuda_copy_tensor_s2g_2d, Function)
                register_primitive_function(name=cuda_copy_tensor_s2g_2d.name, func_or_type=cuda_copy_tensor_s2g_2d)
            elif dim == 3:

                @script
                def cuda_copy_tensor_s2g_3d(
                    dst: PointerType(tensor_map_type), src: PointerType(VoidType()), crd0: i32, crd1: i32, crd2: i32
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(dst, u64)
                    smem_int_ptr = cvta_generic_to_shared(src)
                    asm(template=template_string, inputs=[gmem_int_desc, smem_int_ptr, crd0, crd1, crd2])

                assert isinstance(cuda_copy_tensor_s2g_3d, Function)
                register_primitive_function(name=cuda_copy_tensor_s2g_3d.name, func_or_type=cuda_copy_tensor_s2g_3d)
            elif dim == 4:

                @script
                def cuda_copy_tensor_s2g_4d(
                    dst: PointerType(tensor_map_type),
                    src: PointerType(VoidType()),
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                    crd3: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(dst, u64)
                    smem_int_ptr = cvta_generic_to_shared(src)
                    asm(template=template_string, inputs=[gmem_int_desc, smem_int_ptr, crd0, crd1, crd2, crd3])

                assert isinstance(cuda_copy_tensor_s2g_4d, Function)
                register_primitive_function(name=cuda_copy_tensor_s2g_4d.name, func_or_type=cuda_copy_tensor_s2g_4d)
            elif dim == 5:

                @script
                def cuda_copy_tensor_s2g_5d(
                    dst: PointerType(tensor_map_type),
                    src: PointerType(VoidType()),
                    crd0: i32,
                    crd1: i32,
                    crd2: i32,
                    crd3: i32,
                    crd4: i32,
                ):
                    attrs.func_name = func_name
                    attrs.func_kind = 'cuda_internal'
                    gmem_int_desc = cast(dst, u64)
                    smem_int_ptr = cvta_generic_to_shared(src)
                    asm(template=template_string, inputs=[gmem_int_desc, smem_int_ptr, crd0, crd1, crd2, crd3, crd4])

                assert isinstance(cuda_copy_tensor_s2g_5d, Function)
                register_primitive_function(name=cuda_copy_tensor_s2g_5d.name, func_or_type=cuda_copy_tensor_s2g_5d)


@initialize()
def register_copy_bulk_commit_group():
    from hidet.lang import script, attrs

    @script
    def cuda_copy_bulk_commit_group():
        attrs.func_name = 'cuda_copy_bulk_commit_group'
        attrs.func_kind = 'cuda_internal'
        asm('cp.async.bulk.commit_group;')

    assert isinstance(cuda_copy_bulk_commit_group, Function)
    register_primitive_function(cuda_copy_bulk_commit_group.name, cuda_copy_bulk_commit_group)


@initialize()
def register_copy_bulk_wait_group():
    from hidet.lang import script, attrs

    for groups in range(10):
        for read in ['.read', '']:
            func_name = 'cuda_copy_bulk_wait_group{}_{}'.format(read.replace('.', '_'), groups)

            @script
            def cuda_copy_bulk_wait_group():
                attrs.func_name = func_name
                attrs.func_kind = 'cuda_internal'
                asm('cp.async.bulk.wait_group{} {};'.format(read, groups))

            assert isinstance(cuda_copy_bulk_wait_group, Function)
            register_primitive_function(cuda_copy_bulk_wait_group.name, cuda_copy_bulk_wait_group)


def copy_bulk_g2s(size: int, dst: Expr, src: Expr, mbar: Expr, multicastmask: Optional[Expr] = None):
    """
    Copy data from global memory to shared memory asynchronously.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

    Parameters
    ----------
    size: int
        The amount of memory to be copied, in terms of number of bytes.
    dst: Expr
        The address of the destination in shared memory.
    src: Expr
        The address of the source in global memory.
    mbar:
        The mbarrier object used in the mbarrier based completion mechanism
    multicastmask:
        The instruction modifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster.
        The multicast mask specifies the destination CTAs in the cluster such that each bit position in the 16-bit mask operand corresponds to the %ctaid of the destination CTA.

    WARNING: multicastmask may substantially decrease performance on sm_90.

    Returns
    -------
    ret: Call
        The call expression.
    """
    from hidet.lang import u16, u64

    assert size % 16 == 0, "Bulk copy size must be a multiple of 16"
    func_name = 'copy_bulk_g2s{multicast}'.format(multicast='_multicast' if multicastmask is not None else '')
    if multicastmask is None:
        return call_cuda(func_name, [dst, src, size, cast(mbar, ~u64)])
    else:
        return call_cuda(func_name, [dst, src, size, cast(mbar, ~u64), cast(multicastmask, u16)])


def copy_bulk_s2s(size: int, dst: Expr, src: Expr, mbar: Expr):
    """
    Copy data from shared memory to shared memory asynchronously.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

    Parameters
    ----------
    size: int
        The amount of memory to be copied, in terms of number of bytes.
    dst: Expr
        The address of the destination in shared memory.
    src: Expr
        The address of the source in shared memory.
    mbar:
        The mbarrier object used in the mbarrier based completion mechanism

    Returns
    -------
    ret: Call
        The call expression.
    """
    from hidet.lang import u64

    assert size % 16 == 0, "Bulk copy size must be a multiple of 16"
    func_name = 'copy_bulk_s2s'
    return call_cuda(func_name, [dst, src, size, cast(mbar, ~u64)])


def copy_bulk_s2g(size: int, dst: Expr, src: Expr):
    """
    Copy data from shared memory to global memory asynchronously.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

    Parameters
    ----------
    size: int
        The amount of memory to be copied, in terms of number of bytes.
    dst: Expr
        The address of the destination in global memory.
    src: Expr
        The address of the source in shared memory.

    Returns
    -------
    ret: Call
        The call expression.
    """
    assert size % 16 == 0, "Bulk copy size must be a multiple of 16"
    func_name = 'copy_bulk_s2g'
    return call_cuda(func_name, [dst, src, size])


def copy_bulk_prefetch(size: int, src: Expr):
    """
    Prefetch data from global memory to L2 cache with a non-blocking instruction

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch

    Parameters
    ----------
    size: int
        The amount of memory to be prefetched, in terms of number of bytes.
    src: Expr
        The address of the source in global memory.

    Returns
    -------
    ret: Call
        The call expression.
    """
    func_name = 'copy_bulk_prefetch'
    return call_cuda(func_name, [src, size])


def copy_tensor_g2s(
    dim: int, dst: Expr, tensor_map: Expr, mbar: Expr, *tensor_coords: Expr, multicastmask: Optional[Expr] = None
):
    """
    Copy data from global memory to shared memory asynchronously.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor

    Parameters
    ----------
    dim: int
        The dimension of the tensor data
    dst: Expr
        The address of the destination in shared memory.
    tensor_map: Expr
        An opaque tensor-map object of the source in global memory which resides either in .param space (kernel parameter buffer) or .const space (constant memory) or .global space (global memory)
    mbar:
        The mbarrier object used in the mbarrier based completion mechanism
    tensor_coords: Expr
        The starting coordinates in the tensor data in the global memory
    multicastmask:
        The instruction modifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster.
        The multicast mask specifies the destination CTAs in the cluster such that each bit position in the 16-bit mask operand corresponds to the %ctaid of the destination CTA

    Returns
    -------
    ret: Call
        The call expression.
    """
    from hidet.lang import u16, u64

    func_name = 'copy_tma_{dim}d_g2s{multicast}'.format(
        dim=dim, multicast='_multicast' if multicastmask is not None else ''
    )
    if multicastmask is None:
        return call_cuda(func_name, [dst, tensor_map, cast(mbar, ~u64), *tensor_coords])
    else:
        return call_cuda(func_name, [dst, tensor_map, cast(mbar, ~u64), cast(multicastmask, u16), *tensor_coords])


def copy_tensor_s2g(dim: int, src: Expr, tensor_map: Expr, *tensor_coords: Expr):
    """
    Copy data from shared memory to global memory asynchronously.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor

    Parameters
    ----------
    dim: int
        The dimension of the tensor data
    src: Expr
        The address of the source in shared memory.
    tensor_map: Expr
        An opaque tensor-map object of the source in global memory which resides either in .param space (kernel parameter buffer) or .const space (constant memory) or .global space (global memory)
    tensor_coords: Expr
        The starting coordinates in the tensor data in the global memory

    Returns
    -------
    ret: Call
        The call expression.
    """
    func_name = 'copy_tma_{dim}d_s2g'.format(dim=dim)
    return call_cuda(func_name, [tensor_map, src, *tensor_coords])


def copy_bulk_commit_group():
    """
    Create a new per-thread bulk async-group and batches all prior copy_{bulk|tensor} instructions to the new bulk-async-group

    See Also
        https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
    """
    return call_cuda('copy_bulk_commit_group', [])


def copy_bulk_wait_group(allow_on_fly_groups: Union[int, Expr], read: bool = False):
    """
    Wait the completion of prior bulk async-groups.

    See Also
       https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group

    Parameters
    ----------
    allow_on_fly_groups: Union[int, Expr]
        The maximum number of asynchronous copies that are allowed to be on-the-fly after this function.
        Can be a python integer or a hidet constant expression.
    read:
        Indicates that the waiting has to be don until all bulk async operations in the specified bulk async-group have completed reading from their source locations
    """
    if isinstance(allow_on_fly_groups, Expr):
        from hidet.ir.tools.simplifier import simplify_to_int

        allow_on_fly_groups = simplify_to_int(allow_on_fly_groups)
    if not 0 <= allow_on_fly_groups < 10:
        raise ValueError('n out of bound')
    return call_cuda('copy_bulk_wait_group{}_{}'.format('_read' if read else '', allow_on_fly_groups), [])
