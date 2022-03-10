from typing import Optional
from hidet.ffi import ArgType
from hidet.ir.type import ScalarType, TensorType
from hidet.ir.expr import Var, Call, Equal
from hidet.ir.stmt import AssertStmt, SeqStmt, EvaluateStmt
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import astext
from hidet.ir.dialects.lowlevel import VoidType, PointerType, Cast, Dereference, TensorPointerType
from hidet.ir.task import Grid, Host
from hidet.transforms import Pass


class GeneratePackedFuncPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        new_ir_module = IRModule()
        for func in ir_module.functions.values():
            new_ir_module.add(func.name, func)
            if not isinstance(func.get_attr('worker', None), (Grid, Host)):
                # only generate packed func for entry function
                continue
            if func.get_attr('packed_func', None) is not None:
                # this function itself is a packed function
                continue
            if any(f.get_attr('packed_func', None) is func for f in ir_module.functions.values()):
                # the packed function for current function has existed, skip
                continue
            packed_func = self.generate_packed_func(func, ir_module.lookup_var(func.name))
            new_ir_module.add(packed_func.name, packed_func)
        return new_ir_module

    def generate_packed_func(self, func: Function, func_global_var: Var) -> Optional[Function]:
        assert isinstance(func.get_attr('worker'), (Grid, Host))
        assert isinstance(func.ret_type, VoidType)
        # only check num_args and type_codes now, would add check of type_info in the future
        p_num_args = Var('num_args', ScalarType('int32'))
        p_arg_types = Var('arg_types', PointerType(ScalarType('int32')))
        p_args = Var('args', PointerType(PointerType(VoidType())))

        host_params = [p_num_args, p_arg_types, p_args]
        packed_body = [AssertStmt(Equal(p_num_args, len(func.params)), "expect {} args".format(len(func.params)))]
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
            elif isinstance(param.type, (TensorPointerType, TensorType)):
                code = ArgType.POINTER
                if isinstance(param.type, TensorType):
                    dtype = param.type.scalar_type
                else:
                    dtype = param.type.tensor_type.scalar_type
                grid_args.append(Cast(p_args[idx], PointerType(dtype)))
            else:
                raise NotImplementedError()
            packed_body.append(AssertStmt(p_arg_types[idx] == code, "The {} th arg should be {}".format(idx, astext(param.type))))
        # call
        packed_body.append(EvaluateStmt(Call(func_global_var, grid_args)))

        assert isinstance(func.name, str) and (func.name.endswith('_grid') or func.name.endswith('_host'))
        packed_name = func.name[:-5]
        packed_func = Function(packed_name, host_params, SeqStmt(packed_body), VoidType(), [], [], {'worker': Host(), 'packed_func': func_global_var})
        return packed_func


def generate_packed_func_pass():
    return GeneratePackedFuncPass()
