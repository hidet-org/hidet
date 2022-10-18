from typing import List
from hidet.ir import Function, Constant, TensorType, Var, var, tensor_var
from hidet.ir.stmt import DeclareStmt, Stmt, SeqStmt
from hidet.transforms.base import FunctionPass, Pass
from hidet.ir.functors import collect, rewrite


class NormalizeConstTensorPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        consts: List[Constant] = collect(func.body, Constant)
        if len(consts) == 0:
            return func

        tensor_consts: List[Constant] = [const for const in consts if isinstance(const.data_type, TensorType)]
        body: Stmt = func.body

        const_vars = [Var('const', tensor_const.data_type) for tensor_const in tensor_consts]
        declares = [DeclareStmt(const_var, tensor_const) for const_var, tensor_const in zip(const_vars, tensor_consts)]
        remap = {a: b for a, b in zip(tensor_consts, const_vars)}
        body = rewrite(body, remap)

        if isinstance(body, SeqStmt):
            body = SeqStmt(tuple(declares) + body.seq)
        else:
            body = SeqStmt(tuple(declares) + (body,))

        return Function(func.name, func.params, body, func.ret_type, kind=func.kind, extern_vars=func.extern_vars, attrs=func.attrs)


def normalize_const_tensor_pass() -> Pass:
    return NormalizeConstTensorPass()
