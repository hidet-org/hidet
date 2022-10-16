from typing import List
from hidet.ffi.packedfunc import ArgType
from hidet.ir.func import Function, IRModule
from hidet.ir.type import ScalarType, TensorType, TensorPointerType, PointerType
from hidet.ir.expr import Expr, Var, Call
from hidet.ir.stmt import Stmt, AssertStmt
from hidet.ir.functors import simplify_to_int
from hidet.ir.builders import StmtBuilder
from hidet.ir.task import Task


def generate_packed_func(ir_module: IRModule, func: Function, pack_func_name: str) -> Function:
    from hidet import lang
    from hidet.lang import attr, u32, i32, void_p, deref, cast
    from hidet.lang.cuda import set_kernel_max_dynamic_smem_bytes

    func_var = ir_module.lookup_var(func.name)

    def extract_params_and_call(num_args: Expr, arg_types: Expr, args: Expr) -> Stmt:
        sb = StmtBuilder()
        sb += AssertStmt(num_args == len(func.params), 'Expect {} arguments'.format(len(func.params)))
        extracted_arguments: List[Expr] = []
        for idx, param in enumerate(func.params):
            assert isinstance(param, Var)
            if isinstance(param.type, ScalarType):
                name2code = {
                    'int32': ArgType.INT32,
                    'float32': ArgType.FLOAT32,
                    'float16': ArgType.FLOAT16
                }
                if param.type.name not in name2code:
                    raise NotImplementedError('Unsupported scalar type: {}'.format(param.type.name))
                code: ArgType = name2code[param.type.name]
                arg: Expr = deref(cast(args[idx], ~param.type))
            elif isinstance(param.type, TensorType):
                code: ArgType = ArgType.POINTER
                arg: Expr = cast(args[idx], ~param.type.scalar_type)
            elif isinstance(param.type, TensorPointerType):
                code: ArgType = ArgType.POINTER
                arg: Expr = cast(args[idx], ~param.type.tensor_type.scalar_type)
            elif isinstance(param.type, PointerType):
                code: ArgType = ArgType.POINTER
                arg: Expr = cast(args[idx], param.type)
            else:
                raise NotImplementedError('Unsupported type: {}'.format(param.type))
            extracted_arguments.append(arg)
            sb += AssertStmt(arg_types[idx] == code.value, 'The {}-th argument should be {} ({})'.format(idx, code.name, param.type))

        if func.kind == 'cuda_kernel' and simplify_to_int(func.get_attr('cuda_dynamic_smem_bytes', 0)) > 48 * 1024:
            dynamic_smem_bytes = simplify_to_int(func.get_attr('cuda_dynamic_smem_bytes', 0))
            sb += set_kernel_max_dynamic_smem_bytes(func_var, dynamic_smem_bytes)
        sb += Call(func_var, extracted_arguments)
        return sb.finish()

    with lang.script_module():
        @lang.script
        def packed_func(num_args: i32, arg_types: ~i32, args: ~void_p):
            attr.func_name = pack_func_name
            attr.func_kind = 'packed_func'
            extract_params_and_call(num_args, arg_types, args)

    assert isinstance(packed_func, Function)
    ir_module.add(packed_func.name, packed_func)
