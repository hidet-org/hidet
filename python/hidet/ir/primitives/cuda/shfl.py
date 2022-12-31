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
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import FuncType
from hidet.utils import initialize


@initialize()
def register_primitive_functions():
    functions = [
        ('cuda_activemask', '__activemask', FuncType([], 'int32')),
        # T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
        ('cuda_shfl_sync', '__shfl_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
        ('cuda_shfl_up_sync', '__shfl_up_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
        ('cuda_shfl_down_sync', '__shfl_down_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def shfl_sync(mask, var, src_lane, width=32):
    return call_primitive_func('cuda_shfl_sync', [mask, var, src_lane, width])


def shfl_up_sync(mask, var, delta, width=32):
    return call_primitive_func('cuda_shfl_up_sync', [mask, var, delta, width])


def shfl_down_sync(mask, var, delta, width=32):
    return call_primitive_func('cuda_shfl_down_sync', [mask, var, delta, width])


def shfl_xor_sync(mask, var, lane_mask, width=32):
    return call_primitive_func('cuda_shfl_down_sync', [mask, var, lane_mask, width])


def active_mask():
    return call_primitive_func('cuda_activemask', [])
