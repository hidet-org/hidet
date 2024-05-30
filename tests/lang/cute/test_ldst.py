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
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy
from hidet.ir.cute.ops import tiled_tensor_view, partition_src, partition_dst, mask, copy
from hidet.lang.mapping import auto_map
from hidet.utils import initialize


ldsm_tests = []


@initialize()
def initialize_tests():
    # hidet.option.cache_dir("./demo_ldsm")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    threads = 128

    # ldsm.trans
    block_k = 32
    block_n = 128
    # row major layout
    memory_layout = TensorLayout((block_k, block_n), (block_n, 1))
    threads_map = auto_map(block_k, block_n // 8, workers=threads)
    spatial = threads_map.inner.task_shape
    repeat = threads_map.outer.task_shape
    copy_atom = CopyAtom("thread", (1, 8), TensorLayout(((1), (1, 8)), ((1), (1, 1))))
    thread_in_threadblock = Level(
        "thread", "thread_block", spatial, TensorLayout((spatial[1], spatial[0]), (spatial[0], 1)), repeat
    )
    ldgsts_tiled_copy = TiledCopy(copy_atom, [thread_in_threadblock])

    copy_atom = CopyAtom(
        "warp",
        (16, 16),
        TensorLayout(((16, 2), (8)), ((1, 128), (16))),
        TensorLayout(((4, 8), (2, 2, 2)), ((2, 16), (1, 8, 128))),
    )
    warp_in_threadblock = Level(
        "warp", "thread_block", (2, 2), TensorLayout((2, 2), (1, 2)), (block_k // 32, block_n // 32)
    )
    # copy_atom.spatial(2, 2).repeat(4, 1)
    lds_tiled_copy = TiledCopy(copy_atom, [warp_in_threadblock])

    ldsm_tests.append((memory_layout, ldgsts_tiled_copy, lds_tiled_copy))

    threads = 128

    # ldsm
    block_m = 128
    block_k = 32
    # row major layout
    memory_layout = TensorLayout((block_m, block_k), (block_k, 1))
    threads_map = auto_map(block_m, block_k // 8, workers=threads)
    spatial = threads_map.inner.task_shape
    repeat = threads_map.outer.task_shape
    copy_atom = CopyAtom("thread", (1, 8), TensorLayout(((1), (1, 8)), ((1), (1, 1))))
    thread_in_warp = Level("thread", "warp", (8, 4), TensorLayout((4, 8), (8, 1)))
    warp_in_threadblock = Level("warp", "thread_block", (4, 1), TensorLayout((4, 1)), (4, 1))
    ldgsts_tiled_copy = TiledCopy(copy_atom, [thread_in_warp, warp_in_threadblock])

    copy_atom = CopyAtom(
        "warp",
        (16, 16),
        TensorLayout(((16, 2), (8)), ((1, 128), (16))),
        TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
    )
    warp_in_threadblock = Level(
        "warp", "thread_block", (2, 2), TensorLayout((2, 2), (1, 2)), (block_m // 32, block_k // 32)
    )
    # copy_atom.spatial(2, 2).repeat(4, 1)
    lds_tiled_copy = TiledCopy(copy_atom, [warp_in_threadblock])

    ldsm_tests.append((memory_layout, ldgsts_tiled_copy, lds_tiled_copy))

    # lds example
    threads = 128

    block_m = 128
    block_k = 32
    # row major layout
    memory_layout = TensorLayout((block_m, block_k), (block_k, 1))
    threads_map = auto_map(block_m, block_k // 8, workers=threads)
    spatial = threads_map.inner.task_shape
    repeat = threads_map.outer.task_shape
    copy_atom = CopyAtom("thread", (1, 8), TensorLayout(((1), (1, 8)), ((1), (1, 1))))
    thread_in_threadblock = Level(
        "thread", "thread_block", spatial, TensorLayout((spatial[1], spatial[0]), (spatial[0], 1)), repeat
    )
    ldgsts_tiled_copy = TiledCopy(copy_atom, [thread_in_threadblock])

    lds_tiled_copy = ldgsts_tiled_copy

    ldsm_tests.append((memory_layout, ldgsts_tiled_copy, lds_tiled_copy))


@pytest.mark.parametrize("memory_layout,ldgsts_tiled_copy,lds_tiled_copy", ldsm_tests)
def test_ldsm(memory_layout, ldgsts_tiled_copy, lds_tiled_copy):
    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all

    block_m = memory_layout.shape[0]
    block_n = memory_layout.shape[1]

    gmem_layout_in = memory_layout
    gmem_layout_out = memory_layout
    smem_layout = memory_layout

    atom = ThrValAtom("warp", lds_tiled_copy.copy_atom.shape, lds_tiled_copy.copy_atom.dst_thrval_layout)
    tiled_tensor_layout = TiledTensorLayout(atom, lds_tiled_copy.levels)
    nr_regs = tiled_tensor_layout.val_layout().size()

    copy_atom = CopyAtom("warp", tiled_tensor_layout.atom.shape, tiled_tensor_layout.atom.layout)
    stg_tiled_copy = TiledCopy(copy_atom, tiled_tensor_layout.levels)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(in_ptr: f16[block_m, block_n], out_ptr: f16[block_m, block_n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            smem = shared_tensor("float16", shape=[block_m, block_n])
            t_g_in = tiled_tensor_view(in_ptr, gmem_layout_in, "global")
            t_smem = tiled_tensor_view(smem, smem_layout, "shared")

            txgx_i = partition_src(t_g_in, ldgsts_tiled_copy)
            txsx_i = partition_dst(t_smem, ldgsts_tiled_copy)
            copy(ldgsts_tiled_copy, txgx_i, txsx_i)
            cp_async_wait_all()
            syncthreads()

            regs = register_tensor("float16", shape=[nr_regs])
            t_regs = tiled_tensor_view(regs, tiled_tensor_layout, "register")
            txsx_o = partition_src(t_smem, lds_tiled_copy)
            txrx = partition_dst(t_regs, lds_tiled_copy)
            copy(lds_tiled_copy, txsx_o, txrx)

            # we comment out the last copy operation because we don't have
            # stg instrcutions to support it. I will fix this in some later PRs.
            t_g_out = tiled_tensor_view(out_ptr, gmem_layout_out, "global")
            txgx_o = partition_dst(t_g_out, stg_tiled_copy)
            # copy(stg_tiled_copy, txrx, txgx_o)

    func = script_module.build()
    in_mem = hidet.empty([block_m, block_n], dtype="float16", device="cuda")
    out_mem = hidet.empty([block_m, block_n], dtype="float16", device="cuda")
    func(in_mem, out_mem)
