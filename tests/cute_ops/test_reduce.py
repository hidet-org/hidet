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

import pytest

import hidet
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout
from hidet.ir.cute.layout import ThrValAtom, Level
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy, auto_copy
from hidet.ir.cute.ops import tensor_view, partition_src, partition_dst, mask, copy, reduce_sum, make_tensor
from hidet.lang.mapping import auto_map
from hidet.ir.expr import Expr
from hidet.utils import initialize


reduce_tests = []


@initialize()
def initialize_tests():
    threads = 128

    # ldsm.trans
    block_m = 128
    block_n = 128

    # block layout
    # row major layout
    memory_layout = TensorLayout((block_m, block_n), (block_n, 1))
    threads_map = auto_map(block_m, block_n // 8, workers=threads)
    spatial = threads_map.inner.task_shape
    repeat = threads_map.outer.task_shape
    copy_atom = CopyAtom("thread", (1, 8), TensorLayout(((1), (1, 8)), ((1), (1, 1))))
    thread_in_threadblock = Level(
        "thread", "thread_block", spatial, TensorLayout((spatial[1], spatial[0]), (spatial[0], 1)), repeat
    )
    tiled_copy = TiledCopy(copy_atom, [thread_in_threadblock])

    copy_atom = CopyAtom("thread", (1, 8), TensorLayout(((1), (1, 8)), ((1), (0, 1))))
    thread_in_threadblock = Level(
        "thread",
        "thread_block",
        spatial,
        TensorLayout((spatial[1], spatial[0]), (spatial[0], 0)),
        repeat,
        TensorLayout((repeat[0], repeat[1]), (0, repeat[0])),
    )
    # task mapping for copy a row vector to global memory
    tiled_copy_r = TiledCopy(copy_atom, [thread_in_threadblock])
    reduce_tests.append((memory_layout, tiled_copy, tiled_copy_r))

    # mma layout
    memory_layout = TensorLayout((block_m, block_n), (block_n, 1))
    copy_atom = CopyAtom(
        "warp",
        (16, 16),
        TensorLayout(((16, 2), (8)), ((1, 128), (16))),
        TensorLayout(((4, 8), (2, 2, 2)), ((2, 16), (1, 8, 128))),
    )
    warp_in_threadblock = Level(
        "warp", "thread_block", (2, 2), TensorLayout((2, 2), (1, 2)), (block_m // 32, block_n // 32)
    )
    # copy_atom.spatial(2, 2).repeat(4, 1)
    tiled_copy = TiledCopy(copy_atom, [warp_in_threadblock])

    copy_atom = CopyAtom("warp", (16, 16), TensorLayout(((4, 8), (2, 2, 2)), ((0, 16), (0, 0, 128))))
    warp_in_threadblock = Level(
        "warp",
        "thread_block",
        (2, 2),
        TensorLayout((2, 2), (0, 2)),
        (block_m // 32, block_n // 32),
        TensorLayout((block_m // 32, block_n // 32), (0, block_m // 32)),
    )
    tiled_copy_r = TiledCopy(copy_atom, [warp_in_threadblock])
    reduce_tests.append((memory_layout, tiled_copy, tiled_copy_r))


@pytest.mark.requires_cuda
@pytest.mark.parametrize("memory_layout,tiled_copy,tiled_copy_r", reduce_tests)
def test_reduce(memory_layout, tiled_copy, tiled_copy_r):
    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all

    block_m = memory_layout.shape[0]
    block_n = memory_layout.shape[1]

    gmem_layout_in = TensorLayout((block_m, block_n), (block_n, 1))
    gmem_layout_out = TensorLayout((block_m, block_n), (0, 1))

    atom = ThrValAtom("warp", tiled_copy.copy_atom.shape, tiled_copy.copy_atom.dst_thrval_layout)
    tiled_tensor_layout = TiledTensorLayout(atom, tiled_copy.levels)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(in_ptr: f16[block_m, block_n], out_ptr: f16[1, block_n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 0

            t_g_in = tensor_view(in_ptr, gmem_layout_in, "global")
            t_regs = make_tensor("float16", tiled_tensor_layout, "register")
            txgx_i = partition_src(t_g_in, tiled_copy)
            txrx_i = partition_dst(t_regs, tiled_copy)
            copy(tiled_copy, txgx_i, txrx_i)

            t_regs_o = reduce_sum(t_regs, axis=0)
            t_g_out = tensor_view(out_ptr, gmem_layout_out, "global")
            txrx_o = partition_src(t_regs_o, tiled_copy_r)
            txgx_o = partition_dst(t_g_out, tiled_copy_r)
            copy(tiled_copy_r, txrx_o, txgx_o)

    func = script_module.build()
    in_mem = hidet.empty([block_m, block_n], device="cuda")
    out_mem = hidet.empty([1, block_n], device="cuda")
    func(in_mem, out_mem)
