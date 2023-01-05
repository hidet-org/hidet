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
from typing import Dict, Tuple, Sequence, Union
from hidet.ffi.packedfunc import ArgType
from hidet.ir.func import Function, IRModule
from hidet.ir.type import DataType, TensorType, TensorPointerType, PointerType
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var, Call
from hidet.ir.stmt import Stmt, AssertStmt, LaunchKernelStmt, DeclareStmt
from hidet.ir.functors import simplify_to_int, rewrite, simplify
from hidet.ir.builders import StmtBuilder


def _normalize_dim3(dim3: Union[int, Expr, Sequence[Union[int, Expr]]]) -> Tuple[Expr, Expr, Expr]:
    if isinstance(dim3, int):
        return int32(dim3), int32(1), int32(1)
    elif isinstance(dim3, Expr):
        return simplify(dim3), int32(1), int32(1)
    elif isinstance(dim3, Sequence):
        dim3 = [simplify(int32(v)) for v in dim3]
        assert len(dim3) <= 3
        while len(dim3) < 3:
            dim3.append(int32(1))
        return dim3[0], dim3[1], dim3[2]
    else:
        raise TypeError('Unsupported type: {}'.format(type(dim3)))


def _rewrite_dim3(dim3: Tuple[Expr, Expr, Expr], param2arg: Dict[Expr, Expr]) -> Tuple[Expr, Expr, Expr]:
    return rewrite(dim3[0], param2arg), rewrite(dim3[1], param2arg), rewrite(dim3[2], param2arg)


def add_packed_func(ir_module: IRModule, func: Function, pack_func_name: str):
    from hidet import lang
    from hidet.lang import attr, i32, void_p, deref, cast
    from hidet.lang.cuda import set_kernel_max_dynamic_smem_bytes

    func_var = ir_module.lookup_var(func.name)

    def extract_params_and_call(num_args: Expr, arg_types: Expr, args: Expr) -> Stmt:
        sb = StmtBuilder()
        sb += AssertStmt(num_args == len(func.params), 'Expect {} arguments'.format(len(func.params)))
        param2arg: Dict[Var, Var] = {}
        for idx, param in enumerate(func.params):
            assert isinstance(param, Var)
            if isinstance(param.type, DataType):
                name2code = {'int32': ArgType.INT32, 'float32': ArgType.FLOAT32, 'float16': ArgType.FLOAT16}
                if param.type.name not in name2code:
                    raise NotImplementedError('Unsupported scalar type: {}'.format(param.type.name))
                code: ArgType = name2code[param.type.name]
                arg: Expr = deref(cast(args[idx], ~param.type))
                arg_var: Var = Var(param.hint, param.type)
            elif isinstance(param.type, TensorType):
                code: ArgType = ArgType.POINTER
                arg: Expr = cast(args[idx], ~param.type.dtype)
                arg_var: Var = Var(param.hint, ~param.type.dtype)
            elif isinstance(param.type, TensorPointerType):
                code: ArgType = ArgType.POINTER
                arg: Expr = cast(args[idx], ~param.type.tensor_type.dtype)
                arg_var: Var = Var(param.hint, ~param.type.tensor_type.dtype)
            elif isinstance(param.type, PointerType):
                code: ArgType = ArgType.POINTER
                arg: Expr = cast(args[idx], param.type)
                arg_var: Var = Var(param.hint, param.type)
            else:
                raise NotImplementedError('Unsupported type: {}'.format(param.type))
            param2arg[param] = arg_var
            sb += AssertStmt(arg_types[idx] == code.value, 'The {}-th argument should be {}'.format(idx, param.type))
            sb += DeclareStmt(arg_var, init=arg)

        if func.kind == 'cuda_kernel' and simplify_to_int(func.get_attr('cuda_dynamic_smem_bytes', 0)) > 48 * 1024:
            dynamic_smem_bytes = simplify_to_int(func.get_attr('cuda_dynamic_smem_bytes', 0))
            sb += set_kernel_max_dynamic_smem_bytes(func_var, dynamic_smem_bytes)
        if func.kind == 'cuda_kernel':
            sb += LaunchKernelStmt(
                func_var,
                [param2arg[param] for param in func.params],
                grid_dim=_rewrite_dim3(_normalize_dim3(func.get_attr('cuda_grid_dim')), param2arg),
                block_dim=_rewrite_dim3(_normalize_dim3(func.get_attr('cuda_block_dim')), param2arg),
                shared_mem=rewrite(int32(func.get_attr('cuda_dynamic_smem_bytes', 0)), param2arg),
            )
        elif func.kind == 'host_kernel':
            sb += Call(func_var, [param2arg[param] for param in func.params])
        else:
            raise NotImplementedError('Unsupported function kind: {}'.format(func.kind))
        return sb.finish()

    with lang.script_module():

        @lang.script
        def packed_func(num_args: i32, arg_types: ~i32, args: ~void_p):
            attr.func_name = pack_func_name
            attr.func_kind = 'packed_func'
            extract_params_and_call(num_args, arg_types, args)

    assert isinstance(packed_func, Function)
    packed_func.attrs['packed_func'] = ir_module.lookup_var(func.name)
    ir_module.add(packed_func.name, packed_func)
