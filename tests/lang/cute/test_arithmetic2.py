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
from hidet.ir.cute.ops import tiled_tensor_view, partition_src, partition_dst, mask, copy, arithmetic
from hidet.lang.mapping import auto_map
from hidet.ir.expr import Expr
from hidet.utils import initialize


arithmetic_tests = []


@initialize()
def initialize_tests():
    # hidet.option.cache_dir("./demo_rearrange")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    threads = 128

    # ldsm.trans
    block_m = 128
    block_n = 128
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

    thread_in_threadblock = Level(
        "thread",
        "thread_block",
        spatial,
        TensorLayout((spatial[1], spatial[0]), (spatial[0], 0)),
        repeat,
        TensorLayout((repeat[1], repeat[0]), (repeat[0], 0)),
    )
    tiled_copy_r = TiledCopy(copy_atom, [thread_in_threadblock])

    arithmetic_tests.append((memory_layout, tiled_copy, tiled_copy_r))


@pytest.mark.parametrize("memory_layout,tiled_copy,tiled_copy_r", arithmetic_tests)
def test_mxn_1xn(memory_layout, tiled_copy, tiled_copy_r):
    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all

    block_m = memory_layout.shape[0]
    block_n = memory_layout.shape[1]

    gmem_layout_r = TensorLayout((block_m, block_n), (0, 1))
    gmem_layout_in = memory_layout
    gmem_layout_out = memory_layout

    atom = ThrValAtom("warp", tiled_copy.copy_atom.shape, tiled_copy.copy_atom.dst_thrval_layout)
    tiled_tensor_layout = TiledTensorLayout(atom, tiled_copy.levels)
    nr_regs_i = tiled_tensor_layout.val_layout().size()

    atom = ThrValAtom("warp", tiled_copy_r.copy_atom.shape, tiled_copy_r.copy_atom.dst_thrval_layout)
    tiled_tensor_layout_r = TiledTensorLayout(atom, tiled_copy_r.levels)
    nr_regs_r = tiled_tensor_layout_r.val_count()

    def op(x, y):
        return x + y

    with hidet.script_module() as script_module:

        @hidet.script
        def func(in_ptr: f16[block_m, block_n], r_ptr: f16[1, block_n], out_ptr: f16[block_m, block_n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            regs_r = register_tensor("float16", shape=[nr_regs_r])
            t_g_r = tiled_tensor_view(r_ptr, gmem_layout_r, "global")
            t_regs_r = tiled_tensor_view(regs_r, tiled_tensor_layout_r, "register")
            txgx_r = partition_src(t_g_r, tiled_copy_r)
            txrx_r = partition_dst(t_regs_r, tiled_copy_r)
            copy(tiled_copy_r, txgx_r, txrx_r)

            regs_i = register_tensor("float16", shape=[nr_regs_i])
            t_g_in = tiled_tensor_view(in_ptr, gmem_layout_in, "global")
            t_regs = tiled_tensor_view(regs_i, tiled_tensor_layout, "register")
            txgx_i = partition_src(t_g_in, tiled_copy)
            txrx_i = partition_dst(t_regs, tiled_copy)
            copy(tiled_copy, txgx_i, txrx_i)

            t_regs_o = arithmetic(t_regs_r, t_regs, op=op)
            t_g_out = tiled_tensor_view(out_ptr, gmem_layout_out, "global")
            txgx_o = partition_src(t_g_out, tiled_copy)
            txrx_o = partition_dst(t_regs_o, tiled_copy)
            copy(tiled_copy, txrx_o, txgx_o)

    func = script_module.build()
    in_mem = hidet.empty([block_m, block_n], dtype="float16", device="cuda")
    r_mem = hidet.empty([block_n], dtype="float16", device="cuda")
    out_mem = hidet.empty([block_m, block_n], dtype="float16", device="cuda")
    func(in_mem, r_mem, out_mem)
