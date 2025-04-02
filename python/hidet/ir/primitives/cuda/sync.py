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
from typing import Union

from hidet.ir.expr import Expr, Call
from hidet.ir.func import Function
from hidet.ir.stmt import asm
from hidet.ir.type import FuncType, VoidType
from hidet.ir.primitives.func import is_primitive_function, register_primitive_function
from hidet.lang import attrs, script
from hidet.utils import initialize
from hidet.ir.primitives.func import call_primitive_func


@initialize()
def register_primitive_functions():
    functions = [
        ('cuda_syncthreads', '__syncthreads', FuncType([], VoidType())),
        ('cuda_syncthreads_count', '__syncthreads_count', FuncType(['int32'], 'int32')),
        ('cuda_syncthreads_and', '__syncthreads_and', FuncType(['int32'], 'int32')),
        ('cuda_syncthreads_or', '__syncthreads_or', FuncType(['int32'], 'int32')),
        ('cuda_syncwarp', '__syncwarp', FuncType([], VoidType())),
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def syncthreads() -> Call:
    return call_primitive_func('cuda_syncthreads', [])


def syncthreads_count(value: Expr) -> Call:
    return call_primitive_func('cuda_syncthreads_count', [value])


def syncthreads_and(cond: Union[Expr, int, bool]) -> Call:
    return call_primitive_func('cuda_syncthreads_and', [cond])


def syncthreads_or(cond: Expr) -> Call:
    return call_primitive_func('cuda_syncthreads_or', [cond])


def syncwarp() -> Call:
    return call_primitive_func('cuda_syncwarp', [])


def bar_sync(cooperative_threads: int) -> Call:
    """Synchronize threads in a warp group using bar.sync instruction.

    This function implements the CUDA bar.sync instruction which allows threads to arrive at a pre-computed barrier
    and wait for a fixed number of cooperating threads to arrive. It is commonly used for coordinating threads
    when they need to write to and read from shared memory.

    Args:
        cooperative_threads (int): The number of cooperating threads that must arrive at the barrier.
            Must be a multiple of 32.

    Returns:
        Call: A primitive function call that represents the bar.sync instruction.

    Example:
        ```cuda
        st.shared [r0],r1;     // write my result to shared memory
        bar.cta.sync  1, CNT1; // arrive, wait for others to arrive
        ld.shared r2,[r3];     // use shared results from other threads
        ```

    Raises:
        ValueError: If cooperative_threads is not a multiple of 32.

    Note:
        The bar.sync instruction ensures that all cooperating threads have completed their operations
        before any thread proceeds past the barrier. This is particularly useful for coordinating
        shared memory access patterns.
    """
    if cooperative_threads % 32 != 0:
        raise ValueError(f'cooperating threads in bar.cta.sync must be a multiple of 32, but got {cooperative_threads}')

    func_name = f'cuda_bar_sync_{cooperative_threads}'
    if not is_primitive_function(func_name):

        @script
        def cuda_bar_sync():
            attrs.func_name = func_name
            attrs.func_kind = 'cuda_internal'
            template = 'bar.cta.sync 1, {};'.format(cooperative_threads)
            asm(template=template)

        assert isinstance(cuda_bar_sync, Function)
        register_primitive_function(name=cuda_bar_sync.name, func_or_type=cuda_bar_sync)
    return call_primitive_func(func_name, [])
