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
from hidet.ir.cute.ops import tensor_view, rearrange
from hidet.lang.mapping import auto_map
from hidet.utils import initialize


rearrange_tests = []


@initialize()
def initialize_tests():
    # hidet.option.cache_dir("./demo_rearrange")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    from hidet.lang.cuda import MmaConfig

    mma_config = MmaConfig.m16n8k16_f16_f32()
    mma_m, mma_n, _ = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
    block_m, block_n, block_k = 64, 128, 64
    warp_m, warp_n, warp_k = 32, 64, 64
    warp_count_m, warp_count_n, warp_count_k = (block_m // warp_m, block_n // warp_n, block_k // warp_k)
    nr_threads = warp_count_m * warp_count_n * warp_count_k * 32
    atom_shape = (16, 8)
    atom = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    repeat_shape = (warp_m // mma_m, warp_n // mma_n)
    repeat_layout = TensorLayout((repeat_shape[1], repeat_shape[0]), (repeat_shape[0], 1))
    atom = ThrValAtom("warp", atom_shape, atom, repeat_shape, repeat_layout)
    warps_in_threadblock = Level(
        "warp",
        "thread_block",
        (warp_count_m, warp_count_n),
        TensorLayout((warp_count_n, warp_count_m), (warp_count_m, 1)),
    )
    src = TiledTensorLayout(atom, [warps_in_threadblock])

    store_c_map = auto_map(block_m, block_n // 8, workers=nr_threads)
    repeat = store_c_map.outer.task_shape
    spatial_shape = store_c_map.inner.task_shape
    atom_shape = (1, 8)
    atom = TensorLayout(((1), (1, 8)), ((1), (1, 1)))
    atom = ThrValAtom("thread", atom_shape, atom)
    threads_in_threadblock = Level(
        "thread",
        "thread_block",
        spatial_shape,
        TensorLayout((spatial_shape[1], spatial_shape[0]), (spatial_shape[0], 1)),
        repeat,
    )
    dst = TiledTensorLayout(atom, [threads_in_threadblock])
    rearrange_tests.append((src, dst))

    mma_config = MmaConfig.m16n8k16_f16_f32()
    mma_m, mma_n, _ = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
    block_m, block_n, block_k = 128, 128, 64
    warp_m, warp_n, warp_k = 64, 64, 64
    warp_count_m, warp_count_n, warp_count_k = (block_m // warp_m, block_n // warp_n, block_k // warp_k)
    nr_threads = warp_count_m * warp_count_n * warp_count_k * 32
    atom_shape = (16, 8)
    atom = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    repeat_shape = (warp_m // mma_m, warp_n // mma_n)
    repeat_layout = TensorLayout((repeat_shape[1], repeat_shape[0]), (repeat_shape[0], 1))
    atom = ThrValAtom("warp", atom_shape, atom, repeat_shape, repeat_layout)
    warps_in_threadblock = Level(
        "warp",
        "thread_block",
        (warp_count_m, warp_count_n),
        TensorLayout((warp_count_n, warp_count_m), (warp_count_m, 1)),
    )
    src = TiledTensorLayout(atom, [warps_in_threadblock])

    store_c_map = auto_map(warp_m, warp_n // 8, workers=32)
    repeat = store_c_map.outer.task_shape
    spatial_shape = store_c_map.inner.task_shape
    atom_shape = (1, 8)
    atom = TensorLayout(((1), (1, 8)), ((1), (1, 1)))
    atom = ThrValAtom("thread", atom_shape, atom)
    threads_in_warp = Level(
        "thread",
        "warp",
        spatial_shape,
        TensorLayout((spatial_shape[1], spatial_shape[0]), (spatial_shape[0], 1)),
        repeat,
        TensorLayout(repeat),
    )
    dst = TiledTensorLayout(atom, [threads_in_warp, warps_in_threadblock])
    rearrange_tests.append((src, dst))


@pytest.mark.requires_cuda
@pytest.mark.parametrize("src,dst", rearrange_tests)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_rearrange(src: TiledTensorLayout, dst: TiledTensorLayout, dtype):
    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang import register_tensor

    nr_threads = src.thr_layout().size()
    nr_regs = src.val_layout().size()

    with hidet.script_module() as script_module:

        @hidet.script
        def func():
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = nr_threads
            attrs.cuda.grid_dim = 1

            regs = register_tensor(dtype, shape=[nr_regs])
            ts = tensor_view(regs, src, "register")
            td = rearrange(ts, dst, "register")

    func = script_module.build()
    func()
