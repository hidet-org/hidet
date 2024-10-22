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
import numpy as np
import pytest
from itertools import product

import hidet
from hidet.ir.module import IRModule
from hidet.ir.primitives.cuda import threadIdx
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.wgmma import WgmmaConfig, wgmma_async, wgmma_fence, wgmma_commit_group, wgmma_wait_group
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.type import data_type
from hidet.ir.dtypes import u64
from hidet.lang import attrs
from hidet.lang import register_tensor, as_tensor_pointer, shared_tensor, grid, tensor_pointer
from hidet.graph.tensor import from_numpy
from hidet.lang.layout import row_major, column_major
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared


def matmul_wgmma_tensor_core(
    config: WgmmaConfig, is_a_shared: bool, scale_d: int, scale_a: int, scale_b: int, trans_a: int, trans_b: int
) -> IRModule:
    with hidet.script_module() as script_module:

        @hidet.script
        def matmul_wgmma_tensor_core(
            a_ptr: ~data_type(config.a_input_dtype),
            b_ptr: ~data_type(config.b_input_dtype),
            c_ptr: ~data_type(config.output_dtype),
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1
            if not is_a_shared:
                attrs.cuda.dynamic_smem_bytes = config.n * config.k * data_type(config.b_input_dtype).nbytes
            else:
                attrs.cuda.dynamic_smem_bytes = (config.m * config.k) * data_type(
                    config.a_input_dtype
                ).nbytes + config.n * config.k * data_type(config.b_input_dtype).nbytes
            if not is_a_shared:
                a = as_tensor_pointer(a_ptr, data_type(config.a_input_dtype), [1, config.m, config.k])
            else:
                a = as_tensor_pointer(a_ptr, data_type(config.a_input_dtype), [1, config.m * config.k])

            b = as_tensor_pointer(b_ptr, data_type(config.b_input_dtype), [1, config.n * config.k])
            c = as_tensor_pointer(c_ptr, data_type(config.output_dtype), [1, config.m, config.n])

            regs_c = register_tensor(data_type(config.output_dtype), [config.c_elements])

            # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#register-fragments-and-shared-memory-matrix-layouts
            if trans_b == 0:
                b_layout = column_major(2, config.n // 8) * column_major(8, 8)
            else:
                b_layout = column_major(2, config.n // 8) * row_major(8, 8)

            b_sizeof_input = data_type(config.b_input_dtype).nbytes
            smem_b = tensor_pointer(dtype=config.b_input_dtype, layout=b_layout)
            smem_b = dynamic_shared_memory(byte_offset=0, dtype=config.b_input_dtype)

            smem_b_addr = cvta_generic_to_shared(smem_b)

            # build smem matrix descriptor
            # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
            b_matrix_start_addr = (smem_b_addr & 0x3FFFF) >> 4  # 14 bits

            b_leading_byte_offset: u64 = 64 * b_sizeof_input
            b_leading_byte_offset_encoded: u64 = (b_leading_byte_offset & 0x3FFFF) >> 4

            b_stride_byte_offset: u64 = 128 * b_sizeof_input
            b_stride_byte_offset_encoded: u64 = (b_stride_byte_offset & 0x3FFFF) >> 4

            matrix_base_offset = 0  # Set to 0 for no-swizzle mode

            swizzle_mode: u64 = 0  # 0: No swizzle

            b_desc: u64 = 0
            b_desc = b_desc | (b_matrix_start_addr) << 0
            b_desc = b_desc | (b_leading_byte_offset_encoded & 0xFFFFFFFFFFFFFFFF) << 16
            b_desc = b_desc | (b_stride_byte_offset_encoded & 0xFFFFFFFFFFFFFFFF) << 32

            if not is_a_shared:
                regs_a = register_tensor(data_type(config.a_input_dtype), [config.a_elements])

                # load reg_a
                p = 0
                for i, k in config.a_load_map.on(threadIdx.x):
                    regs_a[p] = a[0, i, k]
                    p += 1
            else:
                if trans_a == 0:
                    a_layout = row_major(8, 2) * row_major(8, 8)
                else:
                    a_layout = row_major(8, 2) * column_major(8, 8)

                smem_a = tensor_pointer(dtype=config.a_input_dtype, layout=a_layout)
                smem_a = dynamic_shared_memory(
                    byte_offset=config.n * config.k * b_sizeof_input, dtype=config.a_input_dtype
                )
                smem_a_addr = cvta_generic_to_shared(smem_a)
                a_sizeof_input = data_type(config.a_input_dtype).nbytes
                a_desc: u64 = 0
                a_matrix_start_addr = (smem_a_addr & 0x3FFF) >> 4
                a_desc = a_desc | (a_matrix_start_addr)

                a_leading_byte_offset: u64 = 128
                a_leading_byte_offset_encoded: u64 = (a_leading_byte_offset & 0x3FFFF) >> 4

                a_stride_byte_offset: u64 = 256
                a_stride_byte_offset_encoded: u64 = (a_stride_byte_offset & 0x3FFFF) >> 4
                a_desc = a_desc | (a_leading_byte_offset_encoded & 0xFFFFFFFFFFFFFFFF) << 16
                a_desc = a_desc | (a_stride_byte_offset_encoded & 0xFFFFFFFFFFFFFFFF) << 32
                for i in range(config.m * config.k):
                    row = i // config.k
                    col = i % config.k
                    smem_a[row, col] = a[0, i]

            for i in range(config.c_elements):
                regs_c[i] = 0.0

            # load reg_c
            p = 0
            for i, j in config.c_store_map.on(threadIdx.x):
                regs_c[p] = c[0, i, j]
                p += 1

            for i in range(config.n * config.k):
                row = i // config.n
                col = i % config.n
                smem_b[row, col] = b[0, i]

            syncthreads()
            wgmma_fence()
            if not is_a_shared:
                wgmma_async(
                    config,
                    regs_a,
                    regs_c,
                    b_desc,
                    scale_d=scale_d,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    trans_a=trans_a,
                    trans_b=trans_b,
                )
            else:
                wgmma_async(
                    config,
                    a_desc,
                    regs_c,
                    b_desc,
                    scale_d=scale_d,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    trans_a=trans_a,
                    trans_b=trans_b,
                )

            wgmma_commit_group()
            wgmma_wait_group(0)

            p = 0
            for i, j in config.c_store_map.on(threadIdx.x):
                c[0, i, j] = regs_c[p]
                p += 1

    ir_module = script_module.ir_module()
    return ir_module


configs = [*WgmmaConfig.all()]
is_a_smem = [True, False]
scale_d_values = [0, 1]
scale_a_values = [1, -1]
scale_b_values = [1, -1]
trans_a_values = [1, 0]
trans_b_values = [1, 0]

test_configs = []
for config, is_a_shared, scale_d, scale_a, scale_b, trans_b in product(
    configs, is_a_smem, scale_d_values, scale_a_values, scale_b_values, trans_b_values
):
    if is_a_shared:
        # Include `trans_a` when `is_a_shared` is True
        for trans_a in trans_a_values:
            test_configs.append((config, is_a_shared, scale_d, scale_a, scale_b, trans_a, trans_b))
    else:
        # Set `trans_a` to None when `is_a_shared` is False
        test_configs.append((config, is_a_shared, scale_d, scale_a, scale_b, None, trans_b))


@pytest.mark.parametrize("config, is_a_shared, scale_d, scale_a, scale_b, trans_a, trans_b", test_configs)

# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async
def test_wgmma(
    config: WgmmaConfig,
    is_a_shared: bool,
    scale_d: int,
    scale_a: int,
    scale_b: int,
    trans_a: int,  # transpose the core matrix in smem
    trans_b: int,
):
    if not (hidet.option.cuda.get_arch_pair() >= (9, 0) and hidet.option.cuda.get_arch_pair() <= config.required_arch):
        pytest.skip("tensor core is supported on device with sm90a")

    ir_module = matmul_wgmma_tensor_core(config, is_a_shared, scale_d, scale_a, scale_b, trans_a, trans_b)
    func = ir_module.build()
    m, n, k = config.m, config.n, config.k
    a = hidet.randint(3, shape=[1, m, k]).to(data_type(config.a_input_dtype).name).cuda()
    b = hidet.randint(3, shape=[1, k, n]).to(data_type(config.b_input_dtype).name).cuda()

    c_desire = hidet.ops.batch_matmul(a if scale_a == 1 else -a, b if scale_b == 1 else -b)

    if scale_d == 1:
        # test if c is accumulated correctly
        c = hidet.ones([1, m, n], dtype=data_type(config.output_dtype).name)
        func(a, b, c)
        np.testing.assert_allclose(
            actual=c.cpu().numpy(),
            desired=c_desire.cpu().numpy() + np.ones((1, m, n), dtype=data_type(config.output_dtype).name),
        )
    else:
        c = hidet.zeros([1, m, n], dtype=data_type(config.output_dtype).name)
        func(a, b, c)
        np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c_desire.cpu().numpy())
