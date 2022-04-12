from typing import Optional
from hidet.ffi import ArgType
from hidet.ir.type import ScalarType, TensorType
from hidet.ir.expr import Var, Call, Equal, Cast
from hidet.ir.stmt import AssertStmt, SeqStmt, EvaluateStmt
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import astext, simplify_to_int
from hidet.ir.dialects.lowlevel import VoidType, PointerType, Dereference, TensorPointerType
from hidet.ir.task import Grid, Host
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.transforms import Pass
from hidet.ir.primitives.func import set_kernel_max_dynamic_smem_bytes


class GeneratePackedFuncPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        new_ir_module = IRModule(task=ir_module.task)
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

    def generate_packed_func(self, func: Function, func_global_var: Var) -> Function:
        func_worker = func.get_attr('worker')
        assert isinstance(func_worker, (Grid, Host))
        assert isinstance(func.ret_type, VoidType)
        assert isinstance(func.name, str) and (func.name.endswith('_grid') or func.name.endswith('_host'))
        packed_name = func.name[:-5]
        with FunctionBuilder(name=packed_name, worker=Host(), attrs={'packed_func': func_global_var}) as fb:
            # params
            p_num_args = Var('num_args', ScalarType('int32'))
            p_arg_types = Var('arg_types', PointerType(ScalarType('int32')))
            p_args = Var('args', PointerType(PointerType(VoidType())))
            fb.extend_params([p_num_args, p_arg_types, p_args])

            # body
            sb = StmtBuilder()
            sb += AssertStmt(Equal(p_num_args, len(func.params)), "expect {} args".format(len(func.params)))
            func_args = []
            for idx, param in enumerate(func.params):
                assert isinstance(param, Var)
                if isinstance(param.type, ScalarType):
                    if param.type.name == 'int32':
                        code = ArgType.INT32
                    elif param.type.name == 'float32':
                        code = ArgType.FLOAT32
                    else:
                        raise NotImplementedError()
                    func_args.append(Dereference(Cast(p_args[idx], PointerType(param.type))))
                elif isinstance(param.type, (TensorPointerType, TensorType)):
                    code = ArgType.POINTER
                    if isinstance(param.type, TensorType):
                        dtype = param.type.scalar_type
                    else:
                        dtype = param.type.tensor_type.scalar_type
                    func_args.append(Cast(p_args[idx], PointerType(dtype)))
                else:
                    raise NotImplementedError()
                sb += AssertStmt(Equal(p_arg_types[idx], code), "The {} th arg should be {}".format(idx, astext(param.type)))
            if isinstance(func_worker, Grid) and simplify_to_int(func_worker.dynamic_smem_bytes) > 48 * 1024:
                sb += set_kernel_max_dynamic_smem_bytes(func_global_var, func_worker.dynamic_smem_bytes)
            sb += Call(func_global_var, func_args)
            fb.set_body(sb.finish())
        return fb.get()


def generate_packed_func_pass():
    return GeneratePackedFuncPass()
