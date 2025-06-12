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
from hidet.ir.cute.layout import TensorLayout, layout_auto
from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy
from hidet.ir.cute.ops import (
    tensor_view,
    make_tensor,
    partition_src,
    partition_dst,
    mask,
    copy,
    sub_tensor,
    make_mbarriers,
    mbarrier_arrive,
    mbarrier_try_wait,
    mbarrier_wait,
)
from hidet.utils.py import cdiv

from hidet.lang.types import u32, i32, f16, u4


@pytest.mark.requires_cuda_hopper
def test_mbarrier():
    from hidet.lang import attrs
    from hidet.lang.cuda import blockIdx, threadIdx

    m = 1024
    n = 1024
    block_m = 128
    block_n = 128
    grid_m = cdiv(m, block_m)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(tensor: f16[m, n], out1: f16[m, n], out2: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = grid_m, 1

            bidx = blockIdx.x
            bidy = blockIdx.y
            mbar_tma = make_mbarriers(2)
            mbarrier_arrive(mbar_tma[0], 128)
            mbarrier_try_wait(mbar_tma[0], False)

            mbarrier_arrive(mbar_tma[1], 128)
            mbarrier_try_wait(mbar_tma[1], False)

    mod = script_module.ir_module()
    print(mod)
    func = script_module.build()


@pytest.mark.requires_cuda_hopper
def test_tensor():
    from hidet.lang import attrs
    from hidet.lang.cuda import blockIdx, threadIdx

    m = 1024
    n = 1024
    block_m = 128
    block_n = 128
    grid_m = cdiv(m, block_m)

    copy_atom = CopyAtom("thread_block", (128, 128), TensorLayout(((128,), (128, 128)), ((0,), (1, 128))))
    tiled_copy = TiledCopy(copy_atom, [])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(tensor: f16[m, n], out1: f16[m, n], out2: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = grid_m, 1

            mbar_tma = make_mbarriers(1)
            bidx = blockIdx.x
            bidy = blockIdx.y
            tg = tensor_view(tensor, TensorLayout((m, n), (n, 1)), "global", (block_m, n), (bidx * block_m, 0))
            ts = make_tensor(f16, TensorLayout((block_m, block_n), (block_n, 1)), "shared")
            txgx = partition_src(tg, tiled_copy)
            txsx = partition_dst(ts, tiled_copy)

            # to1 = tensor_view(out1, TensorLayout((m, n), (n, 1)), "global", (block_m, n), (bidx * block_m, 0))
            # txsx1 = partition_src(ts, auto_copy())
            # txgo1 = partition_dst(to1, auto_copy())

            # to2 = tensor_view(out2[bidx * block_m :, :], TensorLayout((block_m, n), (n, 1)), "global")
            # txsx2 = partition_src(ts, auto_copy())
            # txgo2 = partition_dst(to2, auto_copy())

            phase = False

            for i in range(n // block_n):
                mbarrier_arrive(mbar_tma[0], block_m * block_n * f16.nbytes)
                copy(tiled_copy, txgx[:, :, i], txsx, mbarrier=mbar_tma[0])

                if mbarrier_try_wait(mbar_tma[0], phase):
                    mbarrier_wait(mbar_tma[0], phase)

                # copy(auto_copy((block_m, block_n)), txsx1, txgo1[:, :, i])
                # copy(auto_copy((block_m, block_n)), txsx2, txgo2[:, :, i])

            # ===>
            # tg = tensor_view()
            # bar = TmaBarrier(...)
            # txgx = ...
            # txsx = ...
            # TmaCopy(txgx, txsx, bar)

            # ===>
            #

    func = script_module.build()


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("dtype", [f16, u4])
def test_quantized_weight_tensor(dtype):
    from hidet.lang import attrs
    from hidet.lang.cuda import blockIdx, threadIdx

    m = 1024
    n = 1024
    block_m = 128
    block_n = 128
    grid_m = cdiv(m, block_m)

    copy_atom = CopyAtom("thread_block", (128, 128), TensorLayout(((128,), (128, 128)), ((0,), (1, 128))))
    tiled_copy = TiledCopy(copy_atom, [])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(tensor: dtype[m, n], out1: dtype[m, n], out2: dtype[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = grid_m, 1

            mbar_tma = make_mbarriers(1)
            bidx = blockIdx.x
            bidy = blockIdx.y
            tg = tensor_view(
                tensor,
                TensorLayout(((8, 2, m // 16), (2, 4, 2, n // 16)), ((4, 2, 16 * n), (1, 64, 32, 256))),
                "global",
                (block_m, n),
                (bidx * block_m, 0),
            )
            ts = make_tensor(
                dtype,
                TensorLayout(
                    ((8, 2, block_m // 16), (2, 4, 2, block_n // 16)), ((4, 2, 16 * block_n), (1, 64, 32, 256))
                ),
                "shared",
            )
            txgx = partition_src(tg, tiled_copy)
            txsx = partition_dst(ts, tiled_copy)

            # to1 = tensor_view(out1, TensorLayout((m, n), (n, 1)), "global", (block_m, n), (bidx * block_m, 0))
            # txsx1 = partition_src(ts, auto_copy())
            # txgo1 = partition_dst(to1, auto_copy())

            # to2 = tensor_view(out2[bidx * block_m :, :], TensorLayout((block_m, n), (n, 1)), "global")
            # txsx2 = partition_src(ts, auto_copy())
            # txgo2 = partition_dst(to2, auto_copy())

            phase = False

            for i in range(n // block_n):
                mbarrier_arrive(mbar_tma[0], block_m * block_n * f16.nbytes)
                copy(tiled_copy, txgx[:, :, i], txsx, mbarrier=mbar_tma[0])

                mbar_status = mbarrier_try_wait(mbar_tma[0], phase)
                if mbar_status:
                    mbarrier_wait(mbar_tma[0], phase)

                phase = phase ^ 1

                # copy(auto_copy((block_m, block_n)), txsx1, txgo1[:, :, i])
                # copy(auto_copy((block_m, block_n)), txsx2, txgo2[:, :, i])

    func = script_module.build()


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize(
    "gmem_layout,smem_layout",
    [
        [TensorLayout((1024, 1024), (1, 0)), TensorLayout((128, 128), (1, 0))],
        [TensorLayout((1024, 1024), (0, 1)), TensorLayout((128, 128), (0, 1))],
    ],
)
def test_row_and_column_tensor(gmem_layout: TensorLayout, smem_layout: TensorLayout):
    hidet.option.cache_dir("./tensor")
    hidet.option.save_lower_ir(True)
    hidet.option.debug_cache_tuning()

    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang.cuda import blockIdx, threadIdx

    m, n = gmem_layout.shape
    block_m, block_n = smem_layout.shape
    stride_m, stride_n = smem_layout.stride
    copied = block_m if stride_m == 1 else block_n
    grid_m = cdiv(m, block_m)

    copy_atom = CopyAtom(
        "thread_block", (block_m, block_n), TensorLayout(((128,), (block_m, block_n)), ((0,), (1, block_m)))
    )
    tiled_copy = TiledCopy(copy_atom, [])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(tensor: f16[m, n], out1: f16[m, n], out2: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = grid_m, 1

            mbar_tma = make_mbarriers(1)
            bidx = blockIdx.x
            bidy = blockIdx.y
            tg = tensor_view(tensor, gmem_layout, "global", (block_m, n), (bidx * block_m, 0))
            ts = make_tensor(f16, smem_layout, "shared")
            txgx = partition_src(tg, tiled_copy)
            txsx = partition_dst(ts, tiled_copy)

            phase = False

            for i in range(n // block_n):
                mbarrier_arrive(mbar_tma[0], copied * f16.nbytes)
                copy(tiled_copy, txgx[:, :, i], txsx, mbarrier=mbar_tma[0])

                if mbarrier_try_wait(mbar_tma[0], phase):
                    mbarrier_wait(mbar_tma[0], phase)

    func = script_module.build()


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("dtype,group", [(f16, 16), (u4, 4)])
def test_grouped_scale_tensor(dtype, group):
    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang.cuda import blockIdx, threadIdx

    m = 1024
    n = 1024
    block_m = 128
    block_n = 128
    grid_m = cdiv(m, block_m)
    copy_atom = CopyAtom("thread_block", (128, 128), TensorLayout(((128,), (128, 128)), ((0,), (1, 128))))
    tiled_copy = TiledCopy(copy_atom, [])

    if dtype.is_integer() and dtype.is_integer_subbyte():
        transaction_size = block_m * block_n * dtype.nbits // (8 * group)
    else:
        transaction_size = block_m * block_n * dtype.nbytes

    with hidet.script_module() as script_module:

        @hidet.script
        def func(tensor: dtype[n, m // group], out1: f16[m, n], out2: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = grid_m, 1

            mbar_tma = make_mbarriers(1)
            bidx = blockIdx.x
            bidy = blockIdx.y
            tg = tensor_view(
                tensor,
                TensorLayout(((group, m // group), n), ((0, 1), m // group)),
                "global",
                (block_m, n),
                (bidx * block_m, 0),
            )
            ts = make_tensor(
                dtype, TensorLayout(((group, block_m // group), block_n), ((0, 1), block_m // group)), "shared"
            )
            txgx = partition_src(tg, tiled_copy)
            txsx = partition_dst(ts, tiled_copy)

            phase = False

            for i in range(n // block_n):
                mbarrier_arrive(mbar_tma[0], transaction_size)
                copy(tiled_copy, txgx[:, :, i], txsx, mbarrier=mbar_tma[0])

                if mbarrier_try_wait(mbar_tma[0], phase):
                    mbarrier_wait(mbar_tma[0], phase)

    func = script_module.build()


if __name__ == "__main__":
    with hidet.option.context():
        hidet.option.cache_dir("./tensor")
        hidet.option.save_lower_ir(True)
        hidet.option.debug_cache_tuning()

        test_mbarrier()
        #        test_quantized_weight_tensor(u4)
        test_grouped_scale_tensor(u4, 64)
