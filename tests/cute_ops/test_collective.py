import pytest

import hidet
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout
from hidet.ir.cute.layout import ThrValAtom, Level
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy
from hidet.ir.cute.ops import tensor_view, partition_src, partition_dst, copy, make_tensor
from hidet.ir.cute.collective import collective_store
from hidet.lang.mapping import auto_map


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ldgstg(dtype):
    # hidet.option.cache_dir("./demo_collective_store")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    threads = 128
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
    stg_tiled_copy = TiledCopy(copy_atom, [thread_in_threadblock])

    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import threadIdx, syncthreads, cp_async_wait_all

    block_m = memory_layout.shape[0]
    block_n = memory_layout.shape[1]

    atom = ThrValAtom(
        stg_tiled_copy.copy_atom.level, stg_tiled_copy.copy_atom.shape, stg_tiled_copy.copy_atom.src_thrval_layout
    )
    tiled_tensor_layout = TiledTensorLayout(atom, stg_tiled_copy.levels)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(in_ptr: f16[block_m, block_n], out_ptr: f16[block_m, block_n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            t_regs = make_tensor(dtype, tiled_tensor_layout, "register")
            t_g_in = tensor_view(in_ptr, memory_layout, "global")

            txgx = partition_src(t_g_in, stg_tiled_copy)
            txrx = partition_dst(t_regs, stg_tiled_copy)
            copy(stg_tiled_copy, txgx, txrx)

            collective_store(stg_tiled_copy, t_regs, out_ptr, [0, 0])

    func = script_module.build()
    in_mem = hidet.empty([block_m, block_n], dtype=dtype, device="cuda")
    out_mem = hidet.empty([block_m, block_n], dtype=dtype, device="cuda")
    func(in_mem, out_mem)
