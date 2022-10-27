from typing import List

import hidet.ir.primitives.base.generic
from hidet.ir.type import ScalarType
from hidet.ir.stmt import Stmt
from hidet.ir.expr import Call, Expr, BinaryOp, cast
from hidet.ir.functors import StmtExprRewriter, infer_type, TypeInfer
from hidet.ir.primitives import is_primitive_function, lookup_primitive_function
from hidet.transforms import FunctionBodyPass
from hidet.utils.py import green


def resolve_dtype(arg_dtypes: List[ScalarType]) -> ScalarType:
    return hidet.ir.primitives.base.generic.type_infer_func(arg_dtypes)


def cast_args(args: List[Expr], arg_dtypes: List[ScalarType], target_dtype: ScalarType) -> List[Expr]:
    casted_args = []
    for arg, arg_dtype in zip(args, arg_dtypes):
        if arg_dtype.name != target_dtype.name:
            casted_args.append(cast(arg, target_dtype))
        else:
            casted_args.append(arg)
    return casted_args


class ResolveGenericPrimitiveFuncRewriter(StmtExprRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Call(self, e: Call):
        if is_primitive_function(e.func_var.hint):
            entry = lookup_primitive_function(e.func_var.hint)
            if entry.generic:
                args = [self(arg) for arg in e.args]
                arg_types = [infer_type(arg) for arg in args]
                resolved_dtype = resolve_dtype(arg_types)
                if resolved_dtype.name not in entry.dispatch_dtype_rules:
                    msg = 'Can not dispatch generic primitive function {} to dtype {}'.format(
                        green(entry.name), green(resolved_dtype)
                    )
                    raise NotImplementedError(msg)
                dispatched_func_key = entry.dispatch_dtype_rules[resolved_dtype.name]
                dispatched_func_entry = lookup_primitive_function(name=dispatched_func_key)
                casted_args = cast_args(args, arg_types, resolved_dtype)
                return Call(dispatched_func_entry.var, casted_args)

        return StmtExprRewriter.visit_Call(self, e)

    def visit_Binary(self, e: BinaryOp):
        lhs = self.visit(e.a)
        rhs = self.visit(e.b)
        lhs_dtype = self.type_infer(lhs)
        rhs_dtype = self.type_infer(rhs)
        if isinstance(lhs_dtype, ScalarType) and isinstance(rhs_dtype, ScalarType) and lhs_dtype.name != rhs_dtype.name:
            dtype = resolve_dtype([lhs_dtype, rhs_dtype])
            lhs, rhs = cast_args([lhs, rhs], [lhs_dtype, rhs_dtype], dtype)
            if lhs is e.a and rhs is e.b:
                return e
            else:
                return e.__class__(lhs, rhs)
        else:
            return StmtExprRewriter.visit_Binary(self, e)


class ResolveGenericPrimitiveFuncPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = ResolveGenericPrimitiveFuncRewriter()
        return rewriter.visit(stmt)


def resolve_primitive_func_pass():
    return ResolveGenericPrimitiveFuncPass()
