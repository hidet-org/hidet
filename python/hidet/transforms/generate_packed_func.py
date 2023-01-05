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
from hidet.ffi import ArgType
from hidet.ir.type import DataType, TensorType, VoidType, PointerType, TensorPointerType, data_type
from hidet.ir.expr import Var, Call, Equal, Cast, Dereference
from hidet.ir.stmt import AssertStmt
from hidet.ir.func import IRModule, Function
from hidet.ir.functors import astext, simplify_to_int
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.transforms import Pass
from hidet.ir.primitives import set_kernel_max_dynamic_smem_bytes


class GeneratePackedFuncPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        new_ir_module = IRModule(task=ir_module.task)
        for func in ir_module.functions.values():
            new_ir_module.add(func.name, func)
            if func.kind not in ['cuda_kernel', 'host_kernel']:
                # only generate packed func for entry function
                continue
            if func.get_attr('packed_func', allow_missing=True) is not None:
                # this function itself is a packed function
                continue
            if any(
                f.get_attr('packed_func', allow_missing=True) is ir_module.lookup_var(func.name)
                for f in ir_module.functions.values()
            ):
                # the packed function for current function has existed, skip
                continue
            if not func.name.endswith('_grid') and not func.name.endswith('_host'):
                continue
            packed_func = self.generate_packed_func(func, ir_module.lookup_var(func.name))
            new_ir_module.add(packed_func.name, packed_func)
        return new_ir_module

    def generate_packed_func(self, func: Function, func_global_var: Var) -> Function:
        assert isinstance(func.ret_type, VoidType)
        assert isinstance(func.name, str) and (func.name.endswith('_grid') or func.name.endswith('_host'))
        packed_name = func.name[:-5]
        with FunctionBuilder(name=packed_name, kind='packed_func', attrs={'packed_func': func_global_var}) as fb:
            # params
            p_num_args = Var('num_args', data_type('int32'))
            p_arg_types = Var('arg_types', PointerType(data_type('int32')))
            p_args = Var('args', PointerType(PointerType(VoidType())))
            fb.extend_params([p_num_args, p_arg_types, p_args])

            # body
            sb = StmtBuilder()
            sb += AssertStmt(Equal(p_num_args, len(func.params)), "expect {} args".format(len(func.params)))
            func_args = []
            for idx, param in enumerate(func.params):
                assert isinstance(param, Var)
                if isinstance(param.type, DataType):
                    if param.type.name == 'int32':
                        code: ArgType = ArgType.INT32
                    elif param.type.name == 'float32':
                        code: ArgType = ArgType.FLOAT32
                    else:
                        raise NotImplementedError()
                    func_args.append(Dereference(Cast(p_args[idx], PointerType(param.type))))
                elif isinstance(param.type, (TensorPointerType, TensorType)):
                    code: ArgType = ArgType.POINTER
                    if isinstance(param.type, TensorType):
                        dtype = param.type.dtype
                    else:
                        dtype = param.type.tensor_type.dtype
                    func_args.append(Cast(p_args[idx], PointerType(dtype)))
                elif isinstance(param.type, PointerType):
                    code: ArgType = ArgType.POINTER
                    func_args.append(Cast(p_args[idx], param.type))
                else:
                    raise NotImplementedError()
                sb += AssertStmt(
                    Equal(p_arg_types[idx], code), "The {} th arg should be {}".format(idx + 1, astext(param.type))
                )
            if func.kind == 'cuda_kernel' and simplify_to_int(func.get_attr('cuda_dynamic_smem_bytes', 0)) > 48 * 1024:
                dynamic_smem_bytes = simplify_to_int(func.get_attr('cuda_dynamic_smem_bytes', 0))
                sb += set_kernel_max_dynamic_smem_bytes(func_global_var, dynamic_smem_bytes)
            sb += Call(func_global_var, func_args)
            fb.set_body(sb.finish())
        return fb.get()


def generate_packed_func_pass():
    return GeneratePackedFuncPass()
