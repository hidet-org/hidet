from typing import Optional
from hidet.ffi import ArgType
from hidet.ir.type import ScalarType, TensorType
from hidet.ir.expr import Var, Call
from hidet.ir.stmt import AssertStmt, SeqStmt, EvaluateStmt
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import astext
from hidet.ir.dialects.lowlevel import VoidType, PointerType, Cast, Dereference
from hidet.ir.task import Grid, Host
from hidet.transforms import Pass


class SplitHostDevice(Pass):
    def __init__(self):
        super().__init__('SplitHostDevice')

    def __call__(self, ir_module: IRModule) -> IRModule:
        host_funcs = [self.add_host_func(func, ir_module.lookup_var(func.name)) for func in ir_module.functions.values()
                      if isinstance(func.get_attr('worker'), Grid)]
        for host_func in host_funcs:
            ir_module.functions[host_func.name] = host_func
        return ir_module

    def add_host_func(self, func: Function, func_global_var: Var) -> Optional[Function]:
        assert isinstance(func.get_attr('worker'), Grid)
        assert isinstance(func.ret_type, VoidType)
        # only check num_args and type_codes now, would add check of type_info in the future
        p_num_args = Var('num_args', ScalarType('int32'))
        p_arg_types = Var('arg_types', PointerType(ScalarType('int32')))
        p_args = Var('args', PointerType(PointerType(VoidType())))

        host_params = [p_num_args, p_arg_types, p_args]
        host_body = SeqStmt([])
        host_body.append(AssertStmt(p_num_args == len(func.params), "expect {} args".format(len(func.params))))
        grid_args = []
        # check parameters
        for idx, param in enumerate(func.params):
            assert isinstance(param, Var)
            if isinstance(param.type, ScalarType):
                if param.type.name == 'int32':
                    code = ArgType.INT32
                elif param.type.name == 'float32':
                    code = ArgType.FLOAT32
                else:
                    raise NotImplementedError()
                grid_args.append(Dereference(Cast(p_args[idx], PointerType(param.type))))
            elif isinstance(param.type, TensorType):
                code = ArgType.POINTER
                grid_args.append(Cast(p_args[idx], PointerType(param.type.scalar_type)))
            else:
                raise NotImplementedError()
            host_body.append(AssertStmt(p_arg_types[idx] == code, "The {} th arg should be {}".format(idx, astext(param.type))))
        # call
        host_body.append(EvaluateStmt(Call(func_global_var, grid_args)))

        host_func = Function(func.name.replace('.grid', '.host'), host_params, host_body, VoidType(), [], {'worker': Host()})
        return host_func


def split_host_device_pass():
    return SplitHostDevice()
