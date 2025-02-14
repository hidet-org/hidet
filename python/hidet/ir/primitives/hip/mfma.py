# Licensed under the Apache License,
# Version 2.0 (the "License");
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

# %%
from typing import List

from hidet.ir.mapping import TaskMapping, row_spatial, col_spatial, row_repeat, col_repeat
from hidet.utils import initialize
from hidet.ir.type import PointerType, data_type
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import BlackBoxStmt
from hidet.ir.builders import FunctionBuilder
from hidet.ir.layout import DataLayout, local_layout, row_major
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.func import call_primitive_func

# Support only gfx90a for now

# TODO: support only block=1 instructions and cbsz/abid=0, blgp=0 for now
# see [Matrix Instruction Calculator](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator)
#   for more details
class MfmaConfig:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        input_dtype: str,
        output_dtype: str,
        a_load_map: TaskMapping,
        b_load_map: TaskMapping,
        c_store_map: TaskMapping,
        regs_a_layout: DataLayout = None,
        regs_b_layout: DataLayout = None,
        regs_c_layout: DataLayout = None,
        blocks: int = 1,
        abid: int = 0,
        blgp: int = 0,
    ):
        self.m: int = m
        self.n: int = n
        self.k: int = k
        self.input_dtype: str = input_dtype
        self.output_dtype: str = output_dtype
        self.a_load_map: TaskMapping = a_load_map
        self.b_load_map: TaskMapping = b_load_map
        self.c_store_map: TaskMapping = c_store_map
        self.regs_a_layout: DataLayout = regs_a_layout
        self.regs_b_layout: DataLayout = regs_b_layout
        self.regs_c_layout: DataLayout = regs_c_layout

        self.blocks: int = blocks
        self.abid: int = abid
        self.blgp: int = blgp

        self.a_elements = m * k // 64
        self.b_elements = k * n // 64
        self.c_elements = m * n // 64

        num_regs = lambda dtype, num_elements: num_elements * data_type(dtype).nbytes // 4

        self.a_regs = num_regs(input_dtype, self.a_elements)
        self.b_regs = num_regs(input_dtype, self.b_elements)
        self.c_regs = num_regs(output_dtype, self.c_elements)

    def register_name(self) -> str:
        return f"hip_v_mfma_{self.output_dtype}_{self.m}x{self.n}x{self.k}{self.input_dtype}"

    @staticmethod
    def v_mfma_f32_32x32x2f32():
        return MfmaConfig(
            m=32,
            n=32,
            k=2,
            input_dtype="f32",
            output_dtype="f32",
            a_load_map=col_spatial(32, 2),
            b_load_map=row_spatial(2, 32),
            c_store_map=col_repeat(4, 1) * row_spatial(2, 1) * col_repeat(4, 1) * row_spatial(1, 32),
            regs_a_layout=local_layout(32, 2),
            regs_b_layout=local_layout(2, 32),
            regs_c_layout=row_major(4, 1) * local_layout(2, 1) * row_major(4, 1) * local_layout(1, 32),
        )

    @staticmethod
    def v_mfma_f32_16x16x4f32():
        return MfmaConfig(
            m=16,
            n=16,
            k=4,
            input_dtype="f32",
            output_dtype="f32",
            a_load_map=col_spatial(16, 4),
            b_load_map=row_spatial(4, 16),
            c_store_map=row_spatial(4, 1) * col_repeat(4, 1) * row_spatial(1, 16),
            regs_a_layout=local_layout(16, 4),
            regs_b_layout=local_layout(4, 16),
            regs_c_layout=local_layout(4, 1) * row_major(4, 1) * local_layout(1, 16),
        )

    @staticmethod
    def v_mfma_f32_16x16x16f16():
        return MfmaConfig(
            m=16,
            n=16,
            k=16,
            input_dtype="f16",
            output_dtype="f32",
            a_load_map=col_spatial(16, 4) * row_repeat(1, 4),
            b_load_map=row_spatial(16, 16) * col_repeat(4, 1),
            c_store_map=row_spatial(4, 1) * col_repeat(4, 1) * row_spatial(1, 16),
        )

    @staticmethod
    def v_mfma_f32_32x32x8f16():
        return MfmaConfig(
            m=32,
            n=32,
            k=8,
            input_dtype="f16",
            output_dtype="f32",
            a_load_map=col_spatial(32, 2) * row_repeat(1, 4),
            b_load_map=row_spatial(2, 32) * col_repeat(4, 1),
            c_store_map=row_repeat(4, 1) * row_spatial(2, 1) * col_repeat(4, 1) * row_spatial(1, 32),
        )


@initialize()
def register_mfma_instructions():
    configs: List[MfmaConfig] = [
        MfmaConfig.v_mfma_f32_32x32x2f32(),
        MfmaConfig.v_mfma_f32_16x16x4f32(),
        MfmaConfig.v_mfma_f32_16x16x16f16(),
        MfmaConfig.v_mfma_f32_32x32x8f16(),
    ]

    # pylint: disable=line-too-long
    for config in configs:
        fn_name = config.register_name()
        with FunctionBuilder(name=fn_name, kind='hip_internal') as fb:
            a = Var("a", PointerType(config.input_dtype))
            b = Var("b", PointerType(config.input_dtype))
            c = Var("c", PointerType(config.output_dtype))

            fb.extend_params([a, b, c])

            assert config.input_dtype in ('f16', 'f32')
            assert config.output_dtype in ('f32',)
            assert config.a_regs == config.b_regs
            ab_dtype = '_Float16' if config.input_dtype == 'f16' else 'float'
            c_dtype = 'float'
            if config.a_elements > 1:
                fb += BlackBoxStmt(
                    f"using ABDtype = __attribute__( (__vector_size__({config.a_elements} * sizeof({ab_dtype})) )) {ab_dtype};"
                )
            else:
                fb += BlackBoxStmt(f"using ABDtype = {ab_dtype};")
            if config.c_elements > 1:
                fb += BlackBoxStmt(
                    f"using CDtype = __attribute__( (__vector_size__({config.c_elements} * sizeof({c_dtype})) )) {c_dtype};"
                )
            else:
                fb += BlackBoxStmt(f"using CDtype = {c_dtype};")

            fb += BlackBoxStmt("ABDtype a_regs; ABDtype b_regs; CDtype c_regs;")

            # hopefully this could be optimized away, as the type required by __builtin_amdgcn_mfma_... is not simply float4 or float2, etc.

            if config.a_elements > 1:
                for i in range(config.a_elements):
                    fb += BlackBoxStmt(f"a_regs[{i}] = a[{i}];")
                    fb += BlackBoxStmt(f"b_regs[{i}] = b[{i}];")
            else:
                fb += BlackBoxStmt("a_regs = a[0];")
                fb += BlackBoxStmt("b_regs = b[0];")

            if config.c_elements > 1:
                for i in range(config.c_elements):
                    fb += BlackBoxStmt(f"c_regs[{i}] = c[{i}];")
            else:
                fb += BlackBoxStmt("c_regs = c[0];")

            fb += BlackBoxStmt(
                f"c_regs = __builtin_amdgcn_mfma_{config.output_dtype}_{config.m}x{config.n}x{config.k}{config.input_dtype}(a_regs, b_regs, c_regs, {config.abid}, {config.abid}, {config.blgp});"
            )

            if config.c_elements > 1:
                for i in range(config.c_elements):
                    fb += BlackBoxStmt(f"c[{i}] = c_regs[{i}];")
            else:
                fb += BlackBoxStmt("c[0] = c_regs;")

        register_primitive_function(name=fn_name, func_or_type=fb.func)


def mfma_sync(config: MfmaConfig, a_addr: Expr, b_addr: Expr, c_addr: Expr):
    return call_primitive_func(func_name=config.register_name(), args=[a_addr, b_addr, c_addr])
