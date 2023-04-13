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
from typing import List, Dict, Union

from hidet.ir.builders import FunctionBuilder
from hidet.ir.compute import TensorNode, GridCompute
from hidet.ir.expr import Expr, Var
from hidet.ir.tools import rewrite, simplify
from hidet.ir.stmt import Stmt, BufferStoreStmt, launch_kernel
from hidet.utils import prod
from ..auto_scheduler import AutoScheduler, ComputeExprLower


class CudaAutoScheduler(AutoScheduler):
    def schedule_grid_compute(
        self, node: GridCompute, node_map: Dict[TensorNode, Var], scalar_map: Dict[Var, Var]
    ) -> Stmt:
        # pylint: disable=unnecessary-comprehension, import-outside-toplevel
        from hidet.ir.primitives.cuda import threadIdx, blockIdx
        from hidet.ir.mapping import row_spatial, TaskMapping

        params: List[Var]
        param_map: Dict[Union[TensorNode, Var], Var]
        call_args: List[Expr]
        params, param_map, call_args = self.grid_compute_params_and_args(node, node_map, scalar_map)

        node_shape: List[Expr] = [simplify(rewrite(dim, param_map)) for dim in node.shape]

        block_dim = 512
        grid_dim: Expr = (prod(rewrite(node.shape, scalar_map)) + block_dim - 1) // block_dim

        with FunctionBuilder(
            name=f'compute_{node.name}', kind='cuda_kernel', grid_dim=grid_dim, block_dim=block_dim
        ) as fb:
            # set function parameters
            fb.extend_params(params)

            # calculate task indices assigned to current worker
            worker = blockIdx.x * block_dim + threadIdx.x

            mapping: TaskMapping = row_spatial(*node_shape)
            iter_names = [f'i{i}' for i in range(len(node_shape))]
            with fb.if_then(worker < mapping.num_workers):
                with fb.for_mapping(iter_names, mapping, worker) as task_index:
                    out_param: Var = params[-1]
                    compute_lower = ComputeExprLower(node.value, param_map=param_map)
                    stmts, value = compute_lower.lower()
                    rmap = {axis: axis_value for axis, axis_value in zip(node.axes, task_index)}
                    stmts, value = [rewrite(stmt, rmap) for stmt in stmts], rewrite(value, rmap)
                    fb += stmts
                    fb += BufferStoreStmt(out_param, task_index, value)
        func = fb.get()
        func_var = self.add_function(func)
        return launch_kernel(func_var, args=call_args, grid_dim=grid_dim, block_dim=block_dim)
