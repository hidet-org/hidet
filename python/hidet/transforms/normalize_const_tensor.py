from typing import List
from hidet.ir import Function, Constant, TensorType, Var, var, tensor_var
from hidet.transforms.base import FunctionPass, Pass
from hidet.ir.functors import collect, rewrite


class NormalizeConstTensorPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        consts: List[Constant] = collect(func.body, Constant)
        tensor_consts = [const for const in consts if isinstance(const.data_type, TensorType)]
        body = func.body
        local_const_vars = func.local_const_vars
        for tensor_const in tensor_consts:
            pair = (Var('const', tensor_const.data_type), tensor_const)
            body = rewrite(body, {pair[1]: pair[0]})
            local_const_vars.append(pair)
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, kind=func.kind, local_vars=func.local_vars, local_const_vars=local_const_vars,
                            extern_vars=func.extern_vars, attrs=func.attrs)


def normalize_const_tensor_pass() -> Pass:
    return NormalizeConstTensorPass()
