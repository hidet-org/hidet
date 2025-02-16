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
from enum import Enum
from hidet.utils import initialize
from hidet.ir.type import PointerType, data_type
from hidet.ir.dtypes import int32, float32, float16
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import BlackBoxStmt
from hidet.ir.builders import FunctionBuilder
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.func import call_primitive_func

BUFFER_LOAD_TEMPLATE = """
__device__ {return_type}
    {ldg_inst_name}(int32x4_t srsrc,
    int voffset,
    int soffset,
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.{dtype_name}");
"""


class AmdBufferCoherence(Enum):
    DefaultCoherence = 0
    GLC = 1
    SLC = 2
    GLC_SLC = 3


@initialize()
def register_buffer_instructions(coherence: AmdBufferCoherence = AmdBufferCoherence.DefaultCoherence):
    # this is architecture specific, this one is for gfx90a
    BUFFER_RESOURCE_3RD_DWORD = 0x00020000

    for dtype in ['float32', 'float16', 'int32']:
        for vec_load in [1, 2, 4, 8]:
            data_nbytes = data_type(dtype).nbytes
            bytes_load = data_nbytes * vec_load
            if not bytes_load in (1, 2, 4, 8, 16):
                continue

            fn_name = 'hip_buffer_load_{}x{}'.format(dtype, vec_load)
            with FunctionBuilder(name=fn_name, kind='hip_internal') as fb:
                wave_ptr = Var("wave_ptr", PointerType(dtype))
                elem_space = Var("elem_space", int32)
                lane_offset = Var("lane_offset", int32)
                dst_ptr = Var("dst_ptr", PointerType(dtype))

                fb.extend_params([wave_ptr, elem_space, lane_offset, dst_ptr])

                fb += BlackBoxStmt("using int32x4_t = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;")
                fb += BlackBoxStmt("using int32x2_t = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;")

                if bytes_load == 1:
                    return_type = 'char'
                    ldg_inst_name = 'llvm_amdgcn_raw_buffer_load_i8'
                    dtype_name = 'i8'
                elif bytes_load == 2:
                    return_type = 'short'
                    ldg_inst_name = 'llvm_amdgcn_raw_buffer_load_i16'
                    dtype_name = 'i16'
                elif bytes_load == 4:
                    return_type = 'int'
                    ldg_inst_name = 'llvm_amdgcn_raw_buffer_load_i32'
                    dtype_name = 'i32'
                elif bytes_load == 8:
                    return_type = 'int32x2_t'
                    ldg_inst_name = 'llvm_amdgcn_raw_buffer_load_i64'
                    dtype_name = 'v2i32'
                elif bytes_load == 16:
                    return_type = 'int32x4_t'
                    ldg_inst_name = 'llvm_amdgcn_raw_buffer_load_i128'
                    dtype_name = 'v4i32'

                inst = BUFFER_LOAD_TEMPLATE.format(
                    return_type=return_type, ldg_inst_name=ldg_inst_name, dtype_name=dtype_name
                )
                fb += BlackBoxStmt(inst)

                fb += BlackBoxStmt("int32x4_t buf_resource;")
                fb += BlackBoxStmt("((int**)(&buf_resource))[0] = (int*) wave_ptr;")
                fb += BlackBoxStmt(f"((int*)(&buf_resource))[2] = elem_space * {data_nbytes};")
                fb += BlackBoxStmt(f"((int*)(&buf_resource))[3] = {BUFFER_RESOURCE_3RD_DWORD};")

                fb += BlackBoxStmt(
                    f"{return_type} result = {ldg_inst_name}(buf_resource, \
                    lane_offset * {data_nbytes}, 0, {int(coherence.value)});"
                )

                # from personal observation, the compiler does not vectorize stores, even if
                #  its from register to register (eg. ((char*) dst_ptr)[0] = ((char*) &result)[0];)
                #  maybe any loads and stores that are less than a dword is not vectorized, even
                #  though its contiguous
                fb += BlackBoxStmt(f"(({return_type}*) dst_ptr)[0] = result;")

            register_primitive_function(name=fn_name, func_or_type=fb.func)


def hip_buffer_load(wave_ptr: Expr, elem_space: Expr, lane_offset: Expr, dst_ptr: Expr, dtype: str, vec_load: int):
    """
    each wave holds an unique wave_ptr pointing to global memory, elem_space is the size of the segment of memory
        any element such that lane_offset < elem_space is zeroed
        writes to dst_ptr which should point to registers
    """
    dtype = data_type(dtype)
    assert dtype.nbytes * vec_load in (
        1,
        2,
        4,
        8,
        16,
    ), "vec_load * dtype.nbytes must be 1, 2, 4, 8, or 16. got {}".format(dtype.nbytes * vec_load)
    if dtype == int32:
        return call_primitive_func(
            f"hip_buffer_load_int32x{vec_load}", args=[wave_ptr, elem_space, lane_offset, dst_ptr]
        )
    elif dtype == float32:
        return call_primitive_func(
            f"hip_buffer_load_float32x{vec_load}", args=[wave_ptr, elem_space, lane_offset, dst_ptr]
        )
    elif dtype == float16:
        return call_primitive_func(
            f"hip_buffer_load_float16x{vec_load}", args=[wave_ptr, elem_space, lane_offset, dst_ptr]
        )
    else:
        raise NotImplementedError(f"hip_buffer_load for dtype {dtype} and vec_load {vec_load} is not implemented")
