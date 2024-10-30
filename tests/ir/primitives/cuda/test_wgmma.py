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
import torch
import pytest
import random
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
from hidet.graph.tensor import from_numpy, from_torch
from hidet.lang.layout import row_major, column_major, DataLayout
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

# layout use to prepare smem data for swizzling case
# use the global2local to prepare the data in main then copy over to smem
# https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
# https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/examples/cute/tutorial/wgmma_sm90.cu
# https://github.com/NVIDIA/cutlass/blob/08a49953a0954441ceaed836a89a853a597e67f3/include/cute/atom/mma_traits_sm90_gmma.hpp#L75
class test_layout(DataLayout):
    def __init__(self, bit_base: DataLayout, Datatype, BBits, MBase, SShift=None):
        from hidet.ir.type import data_type

        # self.base: DataLayout = base
        if SShift is None:
            SShift = BBits
        self.num_bits = BBits
        self.num_base = MBase
        self.num_shft = SShift

        assert self.num_base >= 0, "MBase must be positive."
        assert self.num_bits >= 0, "BBits must be positive."
        assert abs(self.num_shft) >= self.num_bits, "abs(SShift) must be more than BBits."

        self.bit_msk = (1 << self.num_bits) - 1
        self.yyy_msk = self.bit_msk << (self.num_base + max(0, self.num_shft))
        self.zzz_msk = self.bit_msk << (self.num_base - min(0, self.num_shft))
        self.msk_sft = self.num_shft

        self.swizzle_code = self.yyy_msk | self.zzz_msk

        self.dtype = Datatype
        self.num_bits = data_type(str(self.dtype)).nbytes * 8  # convert to bits
        self.bit_base = bit_base
        shape = bit_base.shape
        if shape[0] > shape[1]:
            self.base = row_major(shape[0] // self.num_bits, shape[1])
        else:
            self.base = column_major(shape[0], shape[1] // self.num_bits)

        super().__init__(shape=self.base.shape, size=self.base.size)

    def __str__(self):
        import numpy as np
        from hidet.ir.expr import Constant

        print('{}(shape={}, size={})'.format(self.__class__.__name__, self.shape, self.size))
        shape = [int(v) for v in self.shape]
        table = np.zeros(shape=shape, dtype=int)
        ranges = [range(v) for v in shape]
        for indices in product(*ranges):
            local_index = self.global2local(*indices)
            table[indices] = int(local_index)
        return np.array_str(table, max_line_width=120)

    def apply(self, offset):
        return offset ^ ((offset & self.yyy_msk) >> self.msk_sft)

    def global2local(self, *args: int) -> int:
        assert len(args) == len(self.shape)

        # Convert args to a list to make it mutable
        args = list(args)

        bit_shape = self.bit_base.shape
        if bit_shape[0] > bit_shape[1]:
            args[0] = args[0] * self.num_bits
        else:
            args[1] = args[1] * self.num_bits
        # Convert back to a tuple if needed by other parts of the code
        args = tuple(args)

        addr = self.bit_base.global2local(*args)
        addr = self.apply(addr)
        return addr // self.num_bits

    def global2cond(self, *args: int) -> bool:
        return self.base.global2cond(*args)


# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
def encode_matrix_descriptor(x: u64) -> u64:
    return (x & 0x3FFFF) >> 0x4


# build smem matrix descriptor without smem address
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
def make_wgmma_desc(lead_dim_offset: u64, stride_dim_offset: u64, layout_type: u64) -> int:
    desc: u64 = 0
    desc |= encode_matrix_descriptor(lead_dim_offset) << 16
    desc |= encode_matrix_descriptor(stride_dim_offset) << 32
    desc |= layout_type << 62
    return desc


def matmul_wgmma_tensor_core(
    config: WgmmaConfig,
    is_a_shared: bool,
    scale_d: int,
    scale_a: int,
    scale_b: int,
    trans_a: int,
    trans_b: int,
    a_desc_template: u64,
    b_desc_template: u64,
    a_size: int,
    b_size: int,
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
                attrs.cuda.dynamic_smem_bytes = b_size * data_type(config.b_input_dtype).nbytes
            else:
                attrs.cuda.dynamic_smem_bytes = (a_size) * data_type(config.a_input_dtype).nbytes + b_size * data_type(
                    config.b_input_dtype
                ).nbytes

            a = as_tensor_pointer(a_ptr, data_type(config.a_input_dtype), [a_size])
            b = as_tensor_pointer(b_ptr, data_type(config.b_input_dtype), [b_size])
            c = as_tensor_pointer(c_ptr, data_type(config.output_dtype), [1, config.m, config.n])

            regs_c = register_tensor(data_type(config.output_dtype), [config.c_elements])

            smem_b = tensor_pointer(dtype=config.b_input_dtype, shape=[b_size])

            smem_b = dynamic_shared_memory(byte_offset=0, dtype=config.b_input_dtype)

            smem_b_addr = cvta_generic_to_shared(smem_b)

            # build smem matrix descriptor
            # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-descriptor-format
            b_desc: u64 = b_desc_template
            b_matrix_start_addr = (smem_b_addr & 0x3FFFF) >> 4  # 14 bits
            b_desc = b_desc | (b_matrix_start_addr) << 0

            if not is_a_shared:
                regs_a = register_tensor(data_type(config.a_input_dtype), [config.a_elements])

                # load reg_a
                p = 0
                for i, k in config.a_load_map.on(threadIdx.x):
                    idx = i * config.k + k
                    regs_a[p] = a[idx]
                    p += 1
            else:
                b_sizeof_input = data_type(config.b_input_dtype).nbytes

                smem_a = tensor_pointer(dtype=config.a_input_dtype, shape=[a_size])

                smem_a = dynamic_shared_memory(byte_offset=b_size * b_sizeof_input, dtype=config.a_input_dtype)

                smem_a_addr = cvta_generic_to_shared(smem_a)
                a_desc: u64 = a_desc_template
                a_matrix_start_addr = (smem_a_addr & 0x3FFFF) >> 4
                a_desc = a_desc | (a_matrix_start_addr) << 0

                for i in range(a_size):
                    smem_a[i] = a[i]

            for i in range(config.c_elements):
                regs_c[i] = 0.0

            # load reg_c
            p = 0
            for i, j in config.c_store_map.on(threadIdx.x):
                regs_c[p] = c[0, i, j]
                p += 1

            for i in range(b_size):
                smem_b[i] = b[i]

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
swizzle_modes = ["SW128", "SW64", "SW32", "NOSW"]
trans_required_types = ["f16", "bf16"]  # wgmma.async transpose option only support f16 and bf16

test_configs = []

# full test
# for config, is_a_shared, scale_d, scale_a, scale_b, trans_b, swizzle_mode in product(
#     configs, is_a_smem, scale_d_values, scale_a_values, scale_b_values, trans_b_values, swizzle_modes
# ):
#     if config.a_input_dtype in trans_required_types:
#         if is_a_shared:
#             # Include `trans_a` when `is_a_shared` is True
#             for trans_a in trans_a_values:
#                 test_configs.append((config, is_a_shared, scale_d, scale_a, scale_b, trans_a, trans_b, swizzle_mode))
#         else:
#             # Set `trans_a` to None when `is_a_shared` is False
#             test_configs.append((config, is_a_shared, scale_d, scale_a, scale_b, None, trans_b, swizzle_mode))
#     else:
#         # If `config.type` is not in `trans_required_types`, set `trans_a` and `trans_b` to 0
#         test_configs.append((config, is_a_shared, scale_d, scale_a, scale_b, 0, 0, swizzle_mode))

# shorter tests with random config
for config in configs:
    # Randomly decide if `is_a_shared` is True or False
    is_a_shared = random.choice(is_a_smem)

    # Randomly choose other configurations
    scale_d = random.choice(scale_d_values)
    scale_a = random.choice(scale_a_values)
    scale_b = random.choice(scale_b_values)
    swizzle_mode = random.choice(swizzle_modes)

    # Check if `trans_a` and `trans_b` are needed based on `config.type`
    if config.a_input_dtype in trans_required_types:
        trans_a = random.choice(trans_a_values) if is_a_shared else None
        trans_b = random.choice(trans_b_values)
    else:
        trans_a = 0
        trans_b = 0

    # Append the generated configuration
    test_configs.append((config, is_a_shared, scale_d, scale_a, scale_b, trans_a, trans_b, swizzle_mode))


@pytest.mark.parametrize("config, is_a_shared, scale_d, scale_a, scale_b, trans_a, trans_b, swizzle_mode", test_configs)

# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async
def test_wgmma(
    config: WgmmaConfig,
    is_a_shared: bool,
    scale_d: int,
    scale_a: int,
    scale_b: int,
    trans_a: int,  # transpose the core matrix in smem
    trans_b: int,  # transpose the core matrix in smem
    swizzle_mode: str,
):
    if not (hidet.option.cuda.get_arch_pair() >= (9, 0) and hidet.option.cuda.get_arch_pair() <= config.required_arch):
        pytest.skip("tensor core is supported on device with sm90a")

    print(f"\n{swizzle_mode}, is_a_shared: {is_a_shared}")

    a_mode = swizzle_mode if trans_a == 0 else f"{swizzle_mode}_tp"
    b_mode = swizzle_mode if trans_b == 0 else f"{swizzle_mode}_tp"

    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout
    # take f16 matrix A as example
    # matrix is 64 * 16(M * K)
    # core matrix definition is 8 rows and the size of each row is 16 bytes.
    # then the 64 * 16 is expressed as 8 * 2 core matrices
    # the LBO is 128 which is the offset from the start of the core matrix to the start of the next core matrix
    # the SBO is 256 which is the offset from the start of the core matrix to the start of the next row in the same core matrix

    # In the document, the layout for swizzling core matrix is not very clear
    # follow the layout in the nvidia cutlass example
    # https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/examples/cute/tutorial/wgmma_sm90.cu
    # https://github.com/NVIDIA/cutlass/blob/08a49953a0954441ceaed836a89a853a597e67f3/include/cute/atom/mma_traits_sm90_gmma.hpp#L75
    # use print(test_layout()) to visualize the layout
    # The most notable change is the core matrix size is different in different swizzling mode
    # for SW32(f16), the core matrix dim is 8 * 16
    # for SW64(f16), the core matrix dim is 8 * 32
    # for SW128(f16), the core matrix dim is 8 * 64
    # notice the swizzling K dimension is equal or larger than K dimension of A matrix
    # thus LBO is not needed in this case, and the result smem layout is for multiple wgmma A operands
    # transpose mode transpose the core matrix
    # the way to tile core matrix over the smem does not matter as long as the LBO and SBO is set properly
    # for example row_major(8, 2) * row_major(8, 8) and column_major(8, 2) * row_major(8, 8) both work

    desc_template_dict = {}

    desc_template_dict["NOSW"] = make_wgmma_desc(128, 256, 0)
    desc_template_dict["SW32"] = make_wgmma_desc(
        1, 256, 3
    )  # the LBO for swizzling is unused in non-transpose mode, explanation below
    desc_template_dict["SW64"] = make_wgmma_desc(1, 512, 2)
    desc_template_dict["SW128"] = make_wgmma_desc(1, 1024, 1)
    desc_template_dict["NOSW_tp"] = make_wgmma_desc(128, 256, 0)
    desc_template_dict["SW32_tp"] = make_wgmma_desc(512, 256, 3)
    desc_template_dict["SW64_tp"] = make_wgmma_desc(1024, 512, 2)
    desc_template_dict["SW128_tp"] = make_wgmma_desc(2048, 1024, 1)

    m, n, k = config.m, config.n, config.k
    print(f"m: {m}, n: {n}, k: {k}")
    a_cpu = torch.randint(3, (1, m, k), dtype=torch.float16)
    b_cpu = torch.randint(3, (1, k, n), dtype=torch.float16)

    a_layout_dict = {}
    b_layout_dict = {}

    a_layout_dict["NOSW"] = row_major(8, 2) * row_major(8, 8)
    a_layout_dict["SW32"] = row_major(8, 1) * test_layout(row_major(8, 256), "f16", 1, 7, 3)
    a_layout_dict["SW64"] = row_major(8, 1) * test_layout(row_major(8, 512), "f16", 2, 7, 3)
    a_layout_dict["SW128"] = row_major(8, 1) * test_layout(row_major(8, 1024), "f16", 3, 7, 3)
    a_layout_dict["NOSW_tp"] = row_major(8, 2) * column_major(8, 8)  # tp means transpose
    a_layout_dict["SW32_tp"] = row_major(m // 16, 2) * test_layout(column_major(256, 8), "f16", 1, 7, 3)
    a_layout_dict["SW64_tp"] = row_major(m // 32, 2) * test_layout(column_major(512, 8), "f16", 2, 7, 3)
    a_layout_dict["SW128_tp"] = row_major(m // 64, 2) * test_layout(column_major(1024, 8), "f16", 3, 7, 3)

    b_layout_dict["NOSW"] = column_major(2, n // 8) * column_major(8, 8)
    b_layout_dict["SW32"] = column_major(1, n // 8) * test_layout(column_major(256, 8), "f16", 1, 7, 3)
    b_layout_dict["SW64"] = column_major(1, n // 8) * test_layout(column_major(512, 8), "f16", 2, 7, 3)
    b_layout_dict["SW128"] = column_major(1, n // 8) * test_layout(column_major(1024, 8), "f16", 3, 7, 3)
    b_layout_dict["NOSW_tp"] = column_major(2, n // 8) * row_major(8, 8)  # tp means transpose
    b_layout_dict["SW32_tp"] = column_major(2, (n + 16 - 1) // 16) * test_layout(row_major(8, 256), "f16", 1, 7, 3)
    b_layout_dict["SW64_tp"] = column_major(2, (n + 32 - 1) // 32) * test_layout(row_major(8, 512), "f16", 2, 7, 3)
    b_layout_dict["SW128_tp"] = column_major(2, (n + 64 - 1) // 64) * test_layout(row_major(8, 1024), "f16", 3, 7, 3)

    a_torch_1d = torch.zeros(a_layout_dict[a_mode].size, dtype=torch.float16)
    b_torch_1d = torch.zeros(b_layout_dict[b_mode].size, dtype=torch.float16)

    # prepare memory layout for a and b
    for i in range(m * k):
        row = i // k
        col = i % k
        idx = a_layout_dict[a_mode].global2local(row, col)
        a_torch_1d[idx] = a_cpu[0, row, col]

    for i in range(n * k):
        row = i // n
        col = i % n
        idx = b_layout_dict[b_mode].global2local(row, col)
        b_torch_1d[idx] = b_cpu[0, row, col]

    a_1d = from_torch(a_torch_1d).cuda()
    b_1d = from_torch(b_torch_1d).cuda()
    a = from_torch(a_cpu).cuda()
    b = from_torch(b_cpu).cuda()

    ir_module = matmul_wgmma_tensor_core(
        config,
        is_a_shared,
        scale_d,
        scale_a,
        scale_b,
        trans_a,
        trans_b,
        desc_template_dict[a_mode],
        desc_template_dict[b_mode],
        a_layout_dict[a_mode].size,
        b_layout_dict[b_mode].size,
    )

    func = ir_module.build()

    c_desire = hidet.ops.batch_matmul(a if scale_a == 1 else -a, b if scale_b == 1 else -b)

    input_a = a_1d if is_a_shared else a

    if scale_d == 1:
        # test if c is accumulated correctly
        c = hidet.ones([1, m, n], dtype=data_type(config.output_dtype).name)
        func(input_a, b_1d, c)
        np.testing.assert_allclose(
            actual=c.cpu().numpy(),
            desired=c_desire.cpu().numpy() + np.ones((1, m, n), dtype=data_type(config.output_dtype).name),
        )
    else:
        c = hidet.zeros([1, m, n], dtype=data_type(config.output_dtype).name)
        func(input_a, b_1d, c)
        np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c_desire.cpu().numpy())
