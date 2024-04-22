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
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var
from hidet.ir.stmt import LaunchKernelStmt
from hidet.ir.tools import rewrite, simplify
from hidet.ir.builders import FunctionBuilder
from hidet.transforms import Pass


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


def add_launch_func(ir_module: IRModule, kernel_func: Function):
    with FunctionBuilder(name='launch', kind='public') as fb:
        params = [Var(param.hint, param.type) for param in kernel_func.params]
        param_remap = {a: b for a, b in zip(kernel_func.params, params)}
        fb.extend_params(params)

        func_var = ir_module.lookup_var(kernel_func.name)

        if kernel_func.kind == 'cuda_kernel':
            from hidet.lang.cuda import set_kernel_max_dynamic_smem_bytes

            shared_memory_bytes: Expr = rewrite(
                simplify(kernel_func.get_attr('cuda.dynamic_smem_bytes', int32(0))), param_remap
            )
            with fb.if_then(shared_memory_bytes > 48 * 1024):
                fb += set_kernel_max_dynamic_smem_bytes(func_var, shared_memory_bytes)
            fb += LaunchKernelStmt(
                func_var,
                params,
                grid_dim=rewrite(_normalize_dim3(kernel_func.get_attr('cuda.grid_dim')), param_remap),
                cluster_dim=rewrite(_normalize_dim3(kernel_func.get_attr('cuda.cluster_dim', default=1)), param_remap),
                block_dim=rewrite(_normalize_dim3(kernel_func.get_attr('cuda.block_dim')), param_remap),
                shared_mem=shared_memory_bytes,
            )
        elif kernel_func.kind == 'cpu_kernel':
            fb += func_var(*params)
        else:
            raise NotImplementedError('Unsupported function kind: {}'.format(kernel_func.kind))

    launch: Function = fb.func
    ir_module.add_function(launch.name, launch)


class GenerateLaunchFuncPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        if any(func.name.startswith('launch') for func in ir_module.functions.values() if func.kind == 'public'):
            # the launch function has already existed
            return ir_module
        kernel_functions: Dict[str, Function] = {
            name: func for name, func in ir_module.functions.items() if func.kind in ['cuda_kernel', 'cpu_kernel']
        }
        if len(kernel_functions) == 0:
            # no kernel function found in the module, do nothing
            return ir_module
        if len(kernel_functions) > 1:
            raise NotImplementedError('Can only handle one kernel function in a module')
        kernel_func = next(iter(kernel_functions.values()))
        add_launch_func(ir_module, kernel_func)
        return ir_module


def generate_launch_func(ir_module: IRModule) -> IRModule:
    return GenerateLaunchFuncPass().process_module(ir_module)


def generate_launch_func_pass() -> Pass:
    return GenerateLaunchFuncPass()
