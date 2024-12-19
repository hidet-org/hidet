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
from typing import Union, List

from hidet.ir.dtypes import f16, f16x2, bf16, f32, u8, u16, u32, u64
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, cast, deref
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import FuncType, data_type
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import script, attrs, asm

    i32 = data_type('int32')

    for sem in ['acq_rel', 'acquire', 'release']:
        for space in ['global', 'shared']:
            func_name = f'cuda_atomic_add_{sem}_{space}'
            # If the .scope qualifier is absent, .gpu scope is assumed by default.
            # If no sub-qualifier is specified with .shared state space, then ::cta is assumed by default.
            scope = 'gpu' if space == 'global' else 'cta'
            template = f'atom.{sem}.{scope}.{space}.add.s32 %0, [%1], %2;'

            @script
            def func(addr: ~i32, v0: i32) -> i32:
                attrs.func_kind = 'cuda_internal'
                attrs.func_name = func_name
                inputs = [addr, v0]
                ret = i32(0)
                asm(template, outputs=[ret], inputs=inputs, is_volatile=True)
                return ret

            register_primitive_function(name=func.name, func_or_type=func)
    register_primitive_function('cuda_atomic_add', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicAdd')
    register_primitive_function('cuda_atomic_sub', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicSub')
    register_primitive_function('cuda_atomic_min', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicMin')
    register_primitive_function('cuda_atomic_max', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicMax')
    register_primitive_function(
        'cuda_atomic_exchange', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicExch'
    )
    register_primitive_function(
        'cuda_atomic_cas', func_or_type=FuncType([~i32, i32, i32], i32), codegen_name='atomicCAS'
    )


def resolve_reduce_inst_name(dtype: Union[DataType, str], vec: int, scope: str, op: str) -> str:
    inst = 'red'

    assert scope in ['cta', 'gpu', 'sys']
    inst += '.{}'.format(scope)

    inst += '.global'

    assert op in ['add', 'min', 'max']
    inst += '.{}'.format(op)

    dtype = data_type(dtype)
    if dtype in [f16, f16x2, bf16]:
        inst += '.noftz'  # no flush to zero

    if vec > 1:
        assert vec * dtype.nbytes * 8 <= 128  # maximum 128 bits
        inst += '.v{}'.format(vec)

    inst += '.{}'.format(dtype.short_name)

    return inst


def resolve_reduce_func_name(dtype: Union[DataType, str], vec: int, scope: str, op: str) -> str:
    dtype = data_type(dtype)
    return 'cuda_{}'.format(resolve_reduce_inst_name(dtype, vec, scope, op).replace('.', '_'))


@initialize()
def register_reduce_functions():
    from hidet.lang import script, attrs, asm

    for dtype in [f16, f32, f16x2]:
        for vec in [1, 2, 4, 8]:
            for scope in ['gpu', 'cta']:
                if dtype.nbytes * vec * 8 > 128:
                    continue

                inst_name = resolve_reduce_inst_name(dtype, vec, scope, 'add')
                func_name = resolve_reduce_func_name(dtype, vec, scope, 'add')

                def erase_type(v):
                    erased_dtype = {8: u8, 16: u16, 32: u32, 64: u64}[dtype.nbytes * 8]
                    return deref(cast(~v, ~erased_dtype))

                if vec == 1:

                    @script
                    def reduce_op(addr: ~dtype, v0: dtype):
                        attrs.func_kind = 'cuda_internal'
                        attrs.func_name = func_name
                        template = inst_name + ' [%0], %1;'
                        inputs = [addr] + [erase_type(v) for v in [v0]]
                        asm(template, inputs=inputs, is_volatile=True)

                elif vec == 2:

                    @script
                    def reduce_op(addr: ~dtype, v0: dtype, v1: dtype):
                        attrs.func_kind = 'cuda_internal'
                        attrs.func_name = func_name
                        template = inst_name + ' [%0], {%1, %2};'
                        inputs = [addr] + [erase_type(v) for v in [v0, v1]]
                        asm(template, inputs=inputs, is_volatile=True)

                elif vec == 4:

                    @script
                    def reduce_op(addr: ~dtype, v0: dtype, v1: dtype, v2: dtype, v3: dtype):
                        attrs.func_kind = 'cuda_internal'
                        attrs.func_name = func_name
                        template = inst_name + ' [%0], {%1, %2, %3, %4};'
                        inputs = [addr] + [erase_type(v) for v in [v0, v1, v2, v3]]
                        asm(template, inputs=inputs, is_volatile=True)

                elif vec == 8:

                    @script
                    def reduce_op(
                        addr: ~dtype,
                        v0: dtype,
                        v1: dtype,
                        v2: dtype,
                        v3: dtype,
                        v4: dtype,
                        v5: dtype,
                        v6: dtype,
                        v7: dtype,
                    ):
                        attrs.func_kind = 'cuda_internal'
                        attrs.func_name = func_name
                        template = inst_name + ' [%0], {%1, %2, %3, %4};'
                        inputs = [addr] + [erase_type(v) for v in [v0, v1, v2, v3, v4, v5, v6, v7]]
                        asm(template, outputs=[addr], inputs=inputs, is_volatile=True)

                else:
                    raise NotImplementedError()

                register_primitive_function(name=func_name, func_or_type=reduce_op)


def reduce_add(dtype, addr: Expr, src_values: List[Expr], scope='gpu'):
    func_name = resolve_reduce_func_name(dtype, len(src_values), scope, 'add')
    return call_primitive_func(func_name, [addr] + src_values)


def atomic_add(addr: Expr, value: Expr, sem='relaxed', space='global'):
    """
    Atomic reduction operations for thread-to-thread communication.

    See Also:
    ---------
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=fence#parallel-synchronization-and-communication-instructions-atom


    Parameters:
    -----------
    addr: Expr
        Address of the memory location to be updated.
    value: Expr
        Value to be added to the memory location.
    sem: str
        Memory ordering semantics. One of ['acq_rel', 'relaxed', 'acquire', 'release'].

        The optional sem qualifier specifies a memory synchronizing effect as described in the Memory Consistency Model.
        If the sem qualifier is absent, "relaxed" is assumed by default.
    space: str
        Memory space. One of ['global', 'shared'].
    """
    if sem == 'relaxed':
        return call_primitive_func('cuda_atomic_add', [addr, value])
    else:
        return call_primitive_func(f'cuda_atomic_add_{sem}_{space}', [addr, value])


def atomic_sub(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_sub', [addr, value])


def atomic_min(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_min', [addr, value])


def atomic_max(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_max', [addr, value])


def atomic_exchange(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_exchange', [addr, value])


def atomic_cas(addr: Expr, compare: Union[Expr, int], value: Union[Expr, int]):
    return call_primitive_func('cuda_atomic_cas', [addr, compare, value])
