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
"""
Please refer to the following section in PTX manual for the details of MMA instructions:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions
"""
from typing import Dict, Tuple
from itertools import product

from hidet.ir.mapping import TaskMapping, row_spatial, col_spatial, row_repeat, col_repeat
from hidet.utils import initialize
from hidet.ir.type import ReferenceType, PointerType, data_type
from hidet.ir.expr import Expr, cast
from hidet.ir.stmt import asm
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.lang import uint64, attrs, script, u64, ref_u32
from hidet.ir.tools import infer_type

NUM_THREADS = 128  # num threads per warp group
ptx_dtype_names: Dict[str, str] = {
    "f8e4m3": "e4m3",
    "f8e5m2": "e5m2",
    "i32": "s32",
    "i8": "s8",
    "i16": "s16",
    "i64": "s64",
}


def num_regs(short_dtype: str, num_elements: int) -> int:
    num_bytes = data_type(short_dtype).nbytes * num_elements
    assert num_bytes % (4 * NUM_THREADS) == 0
    return num_bytes // (4 * NUM_THREADS)


class WgmmaConfig:
    def __init__(self, m, n, k, a_input_dtype, b_input_dtype, output_dtype, a_load_map, c_store_map, required_arch):
        self.m: int = m
        self.n: int = n
        self.k: int = k
        self.a_input_dtype: str = a_input_dtype
        self.b_input_dtype: str = b_input_dtype
        self.output_dtype: str = output_dtype
        self.a_load_map: TaskMapping = a_load_map  # for register, A can be stored in registers or smem
        self.c_store_map: TaskMapping = c_store_map
        self.a_elements: int = m * k // NUM_THREADS
        self.b_elements: int = k * n // NUM_THREADS
        self.c_elements: int = m * n // NUM_THREADS
        self.a_regs: int = num_regs(a_input_dtype, m * k)
        self.c_regs: int = num_regs(output_dtype, m * n)
        self.required_arch: Tuple[int, int] = (
            required_arch  # The arch should be sm_90a only. Currently, get_arch returns sm_90,
        )
        # and in the build process, it converts to sm_90a.
        self.scale_d: int = 1  # D = A*B when false, D = A*B+D when true
        self.scale_a: int = 1  # A = -A when -1
        self.scale_b: int = 1  # B = -B when -1
        self.trans_a: int = 0  # transpose
        self.trans_b: int = 0

    def inst_name(self) -> str:
        return "wgmma.mma_async.sync.aligned.m{}n{}k{}.{}.{}.{}".format(
            self.m,
            self.n,
            self.k,
            self.get_ptx_dtype_name(self.output_dtype),
            self.get_ptx_dtype_name(self.a_input_dtype),
            self.get_ptx_dtype_name(self.b_input_dtype),
        )

    @staticmethod
    def get_ptx_dtype_name(dtype: str) -> str:
        if dtype in ptx_dtype_names:
            return ptx_dtype_names[dtype]
        else:
            return dtype

    @staticmethod
    def get(m: int, n: int, k: int, a_input_dtype: str, b_input_dtype: str, output_dtype: str):
        return wgmma_configs[f"m{m}n{n}k{k}_{output_dtype}_{a_input_dtype}_{b_input_dtype}"]

    @staticmethod
    def all():
        return list(wgmma_configs.values())

    def __str__(self):
        return self.inst_name()


wgmma_configs: Dict[str, WgmmaConfig] = {}


@initialize()
def register_wgmma_configs():
    # nameing convention: m{m}n{n}k{k}_{output_dtype}_{a_input_dtype}_{b_input_dtype}
    # a_input_dtype, b_input_dtype can be different with fp8.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
    # input_dtype = f16
    for input_dtype in ["f16"]:
        for output_dtype in ["f16", "f32"]:
            for n_values in range(8, 257, 8):  # n values from 8 to 256 in steps of 8
                wgmma_configs.update(
                    {
                        f"m64n{n_values}k16_{output_dtype}_{input_dtype}_{input_dtype}": WgmmaConfig(
                            m=64,
                            n=n_values,
                            k=16,
                            a_input_dtype=input_dtype,
                            b_input_dtype=input_dtype,
                            output_dtype=output_dtype,
                            a_load_map=row_spatial(4, 1)
                            * col_repeat(2, 2, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 2, attrs="u+u+"),  # Calculation specified by the PTX doc
                            c_store_map=row_spatial(4, 1)
                            * col_repeat(2, n_values // 8, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 2, attrs="u+u+"),  # Calculation specified by the PTX doc
                            required_arch=(
                                9,
                                0,
                                'a',
                            ),  # The arch should be sm_90a only. Currently, get_arch returns sm_90,
                            # and in the build process, it converts to sm_90a.
                        )
                    }
                )

    # input_dtype = bf16
    for input_dtype in ["bf16"]:
        for n_values in range(8, 257, 8):  # n values from 8 to 256 in steps of 8
            for output_dtype in ["f32"]:
                wgmma_configs.update(
                    {
                        f"m64n{n_values}k16_{output_dtype}_{input_dtype}_{input_dtype}": WgmmaConfig(
                            m=64,
                            n=n_values,
                            k=16,
                            a_input_dtype=input_dtype,
                            b_input_dtype=input_dtype,
                            output_dtype=output_dtype,
                            a_load_map=row_spatial(4, 1)
                            * col_repeat(2, 2, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 2, attrs="u+u+"),  # Calculation specified by the PTX doc
                            c_store_map=row_spatial(4, 1)
                            * col_repeat(2, n_values // 8, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 2, attrs="u+u+"),  # Calculation specified by the PTX doc
                            required_arch=(
                                9,
                                0,
                                'a',
                            ),  # The arch should be sm_90a only. Currently, get_arch returns sm_90,
                            # and in the build process, it converts to sm_90a.
                        )
                    }
                )
    # f8e4m3, f8e5m2
    for input_dtype in ["f8e4m3", "f8e5m2"]:
        for output_dtype in ["f32", "f16"]:
            for n_value in range(8, 257, 8):
                wgmma_configs.update(
                    {
                        f"m64n{n_value}k32_{output_dtype}_{input_dtype}_{input_dtype}": WgmmaConfig(
                            m=64,
                            n=n_value,
                            k=32,
                            a_input_dtype=input_dtype,
                            b_input_dtype=input_dtype,
                            output_dtype=output_dtype,
                            a_load_map=col_spatial(4, 1)
                            * col_repeat(2, 2, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 4, attrs="u+u+"),
                            c_store_map=row_spatial(4, 1)
                            * col_repeat(2, n_values // 8, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 2, attrs="u+u+"),
                            required_arch=(9, 0, 'a'),
                        )
                    }
                )
    # input_dtype = tf32
    for input_dtype in ["tf32"]:
        for n_values in range(8, 257, 8):  # n values from 8 to 256 in steps of 8
            for output_dtype in ["f32"]:
                wgmma_configs.update(
                    {
                        f"m64n{n_values}k8_{output_dtype}_{input_dtype}_{input_dtype}": WgmmaConfig(
                            m=64,
                            n=n_values,
                            k=8,
                            a_input_dtype=input_dtype,
                            b_input_dtype=input_dtype,
                            output_dtype=output_dtype,
                            a_load_map=row_spatial(4, 1)
                            * col_repeat(2, 2, attrs="u+u+")
                            * row_spatial(8, 4),  # Calculation specified by the PTX doc
                            c_store_map=row_spatial(4, 1)
                            * col_repeat(2, n_values // 8, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 2, attrs="u+u+"),  # Calculation specified by the PTX doc
                            required_arch=(
                                9,
                                0,
                                'a',
                            ),  # The arch should be sm_90a only. Currently, get_arch returns sm_90,
                            # and in the build process, it converts to sm_90a.
                        )
                    }
                )
    # int8, uint8
    n_values_list = list(range(8, 40, 8)) + list(range(48, 272, 16))
    for input_dtype in ["i8", "u8"]:
        for output_dtype in ["i32"]:
            for n_values in n_values_list:  # n values from 8 to 256 in steps of 8
                wgmma_configs.update(
                    {
                        f"m64n{n_values}k32_{output_dtype}_{input_dtype}_{input_dtype}": WgmmaConfig(
                            m=64,
                            n=n_values,
                            k=32,
                            a_input_dtype=input_dtype,
                            b_input_dtype=input_dtype,
                            output_dtype=output_dtype,
                            a_load_map=row_spatial(4, 1)
                            * col_repeat(2, 2, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 4, attrs="u+u+"),  # Calculation specified by the PTX doc
                            c_store_map=row_spatial(4, 1)
                            * col_repeat(2, n_values // 8, attrs="u+u+")
                            * row_spatial(8, 4)
                            * row_repeat(1, 2, attrs="u+u+"),  # Calculation specified by the PTX doc
                            required_arch=(
                                9,
                                0,
                                'a',
                            ),  # The arch should be sm_90a only. Currently, get_arch returns sm_90,
                            # and in the build process, it converts to sm_90a.
                        )
                    }
                )


def register_wgmma_instructions_generic(config: WgmmaConfig):
    inst_name = config.inst_name()
    a_store_types = ["shared", "regs"]
    scale_d_values = [0, 1]
    non_fp_list = ["i8", "u8"]
    trans_required_types = ["f16", "bf16"]
    # Determine scale_a_values and scale_b_values based on data type
    if config.a_input_dtype in non_fp_list:
        scale_a_values = [1]  # Only 1 is valid
        scale_b_values = [1]
    else:
        scale_a_values = [1, -1]
        scale_b_values = [1, -1]

    # Determine trans_a_values and trans_b_values based on data type
    if config.a_input_dtype in trans_required_types:
        trans_a_values = [0, 1]
        trans_b_values = [0, 1]
    else:
        trans_a_values = [None]  # Not used
        trans_b_values = [0]  # Default value

    for a_store_type in a_store_types:
        if a_store_type == "shared":
            param_combinations = product(scale_d_values, scale_a_values, scale_b_values, trans_a_values, trans_b_values)
        else:
            # For "regs", exclude trans_a
            param_combinations = product(scale_d_values, scale_a_values, scale_b_values, trans_b_values)

        for params in param_combinations:
            if a_store_type == "shared":
                scale_d, scale_a, scale_b, trans_a, trans_b = params
            else:
                scale_d, scale_a, scale_b, trans_b = params
                trans_a = None  # Not used for "regs"

            scale_a_str = "n1" if scale_a == -1 else "1"
            scale_b_str = "n1" if scale_b == -1 else "1"

            # Construct func_name exactly as in your original code
            name = inst_name.replace(".", "_")
            name += "_{}".format(a_store_type)

            if config.a_input_dtype in trans_required_types:
                name += "_{}_{}_{}_{}_{}".format(
                    scale_d, scale_a_str, scale_b_str, trans_a if trans_a is not None else "", trans_b
                )
            elif config.a_input_dtype in non_fp_list:
                name += "_{}".format(scale_d)
            else:
                name += "_{}_{}_{}".format(scale_d, scale_a_str, scale_b_str)

            func_name = "cuda_" + name

            # Build the assembly template
            if a_store_type == "regs":
                template_sub_strings = [
                    inst_name,
                    "{{{}}},".format(", ".join([f"%{i}" for i in range(config.c_regs)])),
                    "{{{}}},".format(", ".join([f"%{i}" for i in range(config.c_regs, config.c_regs + config.a_regs)])),
                    "%{},".format(config.c_regs + config.a_regs),
                    "{},".format(scale_d),
                ]

                if config.a_input_dtype not in non_fp_list:
                    template_sub_strings.extend(["{},".format(scale_a), "{},".format(scale_b)])

                if config.a_input_dtype in trans_required_types:
                    template_sub_strings.append("{};".format(trans_b))
                else:
                    # Remove trailing comma and add semicolon
                    template_sub_strings[-1] = template_sub_strings[-1].rstrip(',') + ';'

                template_string = " ".join(template_sub_strings)
                rc_dtype = "float32" if config.output_dtype == "f32" else "uint32"

                @script
                def cuda_wgmma(a: ~data_type(config.a_input_dtype), c: ~data_type(config.output_dtype), b_desc: uint64):
                    attrs.func_kind = 'cuda_internal'
                    attrs.func_name = func_name

                    ra = cast(a, PointerType("uint32"))
                    rc = cast(c, PointerType(rc_dtype))

                    asm(
                        template=template_string,
                        output_inputs=[rc[i] for i in range(config.c_regs)],
                        inputs=[ra[i] for i in range(config.a_regs)] + [b_desc],
                        is_volatile=True,
                    )

                assert isinstance(cuda_wgmma, Function)
                register_primitive_function(name=func_name, func_or_type=cuda_wgmma)

            elif a_store_type == "shared":
                template_sub_strings = [
                    inst_name,
                    "{{{}}},".format(", ".join([f"%{i}" for i in range(config.c_regs)])),
                    "%{},".format(config.c_regs),
                    "%{},".format(config.c_regs + 1),
                    "{},".format(scale_d),
                ]

                if config.a_input_dtype not in non_fp_list:
                    template_sub_strings.extend(["{},".format(scale_a), "{},".format(scale_b)])

                if config.a_input_dtype in trans_required_types:
                    template_sub_strings.extend(["{},".format(trans_a), "{};".format(trans_b)])
                else:
                    # Remove trailing comma and add semicolon
                    template_sub_strings[-1] = template_sub_strings[-1].rstrip(',') + ';'

                template_string = " ".join(template_sub_strings)
                rc_dtype = "float32" if config.output_dtype == "f32" else "uint32"

                @script
                def cuda_wgmma(a_desc: uint64, c: ~data_type(config.output_dtype), b_desc: uint64):
                    attrs.func_kind = 'cuda_internal'
                    attrs.func_name = func_name

                    rc = cast(c, PointerType(rc_dtype))

                    asm(
                        template=template_string,
                        output_inputs=[rc[i] for i in range(config.c_regs)],
                        inputs=[a_desc, b_desc],
                        is_volatile=True,
                    )

                assert isinstance(cuda_wgmma, Function)
                register_primitive_function(name=func_name, func_or_type=cuda_wgmma)


def wgmma_async(
    config: WgmmaConfig,
    a_expr: Expr,
    c_addr: Expr,
    b_desc: Expr,
    scale_d: Expr = 1,
    scale_a: Expr = 1,
    scale_b: Expr = 1,
    trans_a: Expr = None,
    trans_b: Expr = 0,
):
    from hidet.ir.primitives import is_primitive_function

    # from hidet import uint64
    scale_d_values = [0, 1]
    scale_a_values = [1, -1]
    scale_b_values = [1, -1]
    trans_a_values = [0, 1, None]
    trans_b_values = [0, 1]
    assert scale_d in scale_d_values
    assert scale_a in scale_a_values
    assert scale_b in scale_b_values
    assert trans_a in trans_a_values
    assert trans_b in trans_b_values

    name = config.inst_name().replace(".", "_")
    a_expr_type = infer_type(a_expr)
    # resolve a_store_type automatically by the infered type of a_expr
    if a_expr_type == uint64:
        name = name + "_{}".format("shared")
    else:
        name = name + "_{}".format("regs")
        assert trans_a is None, "trans_a is not used for 'regs'"

    scale_a_str = "n1" if scale_a == -1 else "1"
    scale_b_str = "n1" if scale_b == -1 else "1"
    if config.a_input_dtype in ["f16", "bf16"]:
        name = name + "_{}_{}_{}_{}_{}".format(
            scale_d, scale_a_str, scale_b_str, trans_a if trans_a is not None else "", trans_b
        )
    else:
        if config.a_input_dtype in ["i8", "u8"]:
            name = name + "_{}".format(scale_d)
        else:
            name = name + "_{}_{}_{}".format(scale_d, scale_a_str, scale_b_str)

    if not is_primitive_function(f"cuda_{name}"):
        register_wgmma_instructions_generic(config)

    return call_cuda(func_name=name, args=[a_expr, c_addr, b_desc])


@initialize()
def register_wgmma_fence():
    func_name = "cuda_wgmma_fence"
    template_string = "wgmma.fence.sync.aligned;"

    @script
    def cuda_wgmma_fence():
        attrs.func_name = func_name
        attrs.func_kind = "cuda_internal"
        asm(template=template_string, is_volatile=True, memory_fence=True)

    assert isinstance(cuda_wgmma_fence, Function)
    register_primitive_function(name=cuda_wgmma_fence.name, func_or_type=cuda_wgmma_fence)


@initialize()
def register_wgmma_commit_group():
    func_name = "cuda_wgmma_commit_group"
    template_string = "wgmma.commit_group.sync.aligned;"

    @script
    def cuda_wgmma_commit_group():
        attrs.func_name = func_name
        attrs.func_kind = "cuda_internal"
        asm(template=template_string, is_volatile=True, memory_fence=True)

    assert isinstance(cuda_wgmma_commit_group, Function)
    register_primitive_function(name=cuda_wgmma_commit_group.name, func_or_type=cuda_wgmma_commit_group)


@initialize()
def register_wgmma_wait_group():
    for n in range(8):
        func_name = "cuda_wgmma_wait_group_{}".format(n)

        @script
        def cuda_wgmma_wait_group():
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            # assert N >= 0 and N <=7
            template = "wgmma.wait_group.sync.aligned {};".format(n)
            asm(template=template, is_volatile=True, memory_fence=True)

        assert isinstance(cuda_wgmma_wait_group, Function)
        register_primitive_function(name=cuda_wgmma_wait_group.name, func_or_type=cuda_wgmma_wait_group)


@initialize()
def register_wgmma_fence_operand():
    ref_f32 = ReferenceType(data_type("f32"))

    for dtype in [ref_u32, ref_f32]:
        func_name = f"cuda_wgmma_fence_operand_{dtype.base_type.short_name}"

        @script
        def cuda_wgmma_fence_operand(reg: dtype):
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            template = ""
            asm(template=template, output_inputs=[reg], is_volatile=True, memory_fence=True)

        register_primitive_function(name=cuda_wgmma_fence_operand.name, func_or_type=cuda_wgmma_fence_operand)


def wgmma_fence():
    name = "wgmma_fence"
    return call_cuda(func_name=name, args=[])


def wgmma_commit_group():
    name = "wgmma_commit_group"
    return call_cuda(func_name=name, args=[])


def wgmma_wait_group(N: Expr):
    name = "wgmma_wait_group_{}".format(N)
    assert 0 <= N <= 7
    return call_cuda(func_name=name, args=[])


def wgmma_fence_operand(reg: Expr):
    name = "wgmma_fence_operand_{}".format(infer_type(reg).short_name)
    return call_cuda(func_name=name, args=[reg])


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
