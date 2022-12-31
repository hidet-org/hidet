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
from typing import List, Optional, Tuple

from hidet.ir.type import DataType
from hidet.ir.stmt import Stmt
from hidet.ir.expr import Call, Expr, BinaryOp, cast
from hidet.ir.func import Function
from hidet.ir.functors import StmtExprRewriter, infer_type, TypeInfer
from hidet.ir.primitives import is_primitive_function, lookup_primitive_function
from hidet.ir.primitives.math import registered_math_function_sets
from hidet.transforms import FunctionBodyPass
from hidet.utils.py import green


def resolve_dtype(arg_dtypes: List[DataType]) -> DataType:
    import hidet.ir.primitives.math

    return hidet.ir.primitives.math.type_infer_func(arg_dtypes)


def cast_args(args: List[Expr], arg_dtypes: List[DataType], target_dtype: DataType) -> List[Expr]:
    casted_args = []
    for arg, arg_dtype in zip(args, arg_dtypes):
        if arg_dtype.name != target_dtype.name:
            casted_args.append(cast(arg, target_dtype))
        else:
            casted_args.append(arg)
    return casted_args


class ResolveGenericPrimitiveFuncRewriter(StmtExprRewriter):
    def __init__(self, device: str):
        super().__init__()
        self.type_infer = TypeInfer()
        self.device: str = device

    def visit_Call(self, e: Call):
        if is_primitive_function(e.func_var.hint):
            entry = lookup_primitive_function(e.func_var.hint)
            if entry.generic:
                args = [self(arg) for arg in e.args]
                arg_types = [infer_type(arg) for arg in args]
                resolved_dtype: DataType = resolve_dtype(arg_types)
                generic, func_name = entry.name.split('_')  # such as 'generic_exp'
                assert generic == 'generic'
                dtype: str = resolved_dtype.name
                key: Tuple[str, str] = (self.device, dtype)
                if key not in registered_math_function_sets:
                    msg = 'Can not dispatch generic primitive function {} to device {} and dtype {}.\n'.format(
                        green(entry.name), green(self.device), green(dtype)
                    )
                    msg += 'Registered math function sets: {}'.format(list(registered_math_function_sets.keys())) + '\n'
                    raise NotImplementedError(msg)
                else:
                    func_set = registered_math_function_sets[key]
                    assert hasattr(func_set, func_name)
                    func = getattr(func_set, func_name)
                    try:
                        return func(*args)
                    except NotImplementedError as err:
                        msg = 'Math function {} for {} data has not been implemented for device {}'.format(
                            green(e.func_var.hint.replace('generic_', '')), green(dtype), green(self.device)
                        )
                        raise NotImplementedError(msg) from err

        return StmtExprRewriter.visit_Call(self, e)

    def visit_Binary(self, e: BinaryOp):
        lhs = self.visit(e.a)
        rhs = self.visit(e.b)
        lhs_dtype = self.type_infer(lhs)
        rhs_dtype = self.type_infer(rhs)
        if isinstance(lhs_dtype, DataType) and isinstance(rhs_dtype, DataType) and lhs_dtype.name != rhs_dtype.name:
            dtype = resolve_dtype([lhs_dtype, rhs_dtype])
            lhs, rhs = cast_args([lhs, rhs], [lhs_dtype, rhs_dtype], dtype)
            if lhs is e.a and rhs is e.b:
                return e
            else:
                return e.__class__(lhs, rhs)
        else:
            return StmtExprRewriter.visit_Binary(self, e)


class ResolveGenericPrimitiveFuncPass(FunctionBodyPass):
    def __init__(self):
        super().__init__()
        self.device: Optional[str] = None

    def process_func(self, func: Function) -> Function:
        func_kind_to_device = {'host_kernel': 'cpu', 'packed_func': 'cpu', 'cuda_kernel': 'cuda', 'cuda_device': 'cuda'}
        self.device = func_kind_to_device[func.kind]
        return FunctionBodyPass.process_func(self, func)

    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = ResolveGenericPrimitiveFuncRewriter(self.device)
        return rewriter.visit(stmt)


def resolve_primitive_func_pass():
    return ResolveGenericPrimitiveFuncPass()
