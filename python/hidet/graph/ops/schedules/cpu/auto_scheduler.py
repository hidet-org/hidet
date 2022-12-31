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
from typing import List, Dict

from hidet.ir.builders import FunctionBuilder
from hidet.ir.compute import TensorNode, GridCompute
from hidet.ir.expr import Call, Expr, Var, convert
from hidet.ir.functors import collect, rewrite
from hidet.ir.stmt import Stmt, BufferStoreStmt, EvaluateStmt
from ..auto_scheduler import AutoScheduler, ComputeExprLower


class CpuAutoScheduler(AutoScheduler):
    def schedule_grid_compute(self, gc: GridCompute, node: TensorNode, node_map: Dict[TensorNode, Expr]) -> Stmt:
        # pylint: disable=too-many-locals, import-outside-toplevel, unnecessary-comprehension
        from hidet.ir.mapping import row_repeat, TaskMapping

        used_tensors: List[TensorNode] = collect(gc.value, TensorNode, stop_when_found=True)
        param_tensors: List[TensorNode] = used_tensors + [node]
        params: List[Var] = [Var(tensor.name, tensor.ttype) for tensor in param_tensors]

        with FunctionBuilder(name=f'compute_{node.name}', kind='host_kernel') as fb:
            # set function parameters
            fb.extend_params(params)

            mapping: TaskMapping = row_repeat(*gc.shape)
            iter_names = [f'i{i}' for i in range(len(gc.shape))]
            with fb.for_mapping(iter_names, mapping, convert(0)) as task_index:
                out_param: Var = params[-1]
                param_map: Dict[TensorNode, Expr] = {
                    tensor_node: param_var for tensor_node, param_var in zip(param_tensors, params)
                }
                compute_lower = ComputeExprLower(gc.value, param_map=param_map)
                stmts, value = compute_lower.lower()
                rmap = {axis: axis_value for axis, axis_value in zip(gc.axes, task_index)}
                stmts, value = [rewrite(stmt, rmap) for stmt in stmts], rewrite(value, rmap)
                fb += stmts
                fb += BufferStoreStmt(out_param, task_index, value)
        func = fb.get()
        func_var = self.add_function(func)
        return EvaluateStmt(Call(func_var, args=[node_map[param_tensor] for param_tensor in param_tensors]))
