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
from typing import Dict

from hidet.transforms.base import Pass
from hidet.ir import ForMappingStmt, Expr, IfStmt, Function
from hidet.ir.functors import IRRewriter
from hidet.ir.mapping import mapping_2_list
from hidet.ir.analyzers import normalize_launch_dims, infer_bound, BoundInfo
from hidet.lang.cuda import threadIdx, blockIdx
from hidet.utils import prod


class TaskMappingBoundCheckRewriter(IRRewriter):
    def __init__(self, block_dims, grid_dims):
        super().__init__()
        self.block_dims = block_dims
        self.grid_dims = grid_dims

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        new_body = self.visit(stmt.body)
        if new_body is not stmt.body:
            new_stmt = ForMappingStmt(loop_vars=stmt.loop_vars, mapping=stmt.mapping, worker=stmt.worker, body=new_body)
        else:
            new_stmt = stmt

        mapping_list = mapping_2_list(new_stmt.mapping)
        task_size = prod([i.num_workers for i in mapping_list])

        worker = new_stmt.worker
        var4bound = {}
        if isinstance(self.block_dims[0], int):
            var4bound[threadIdx.x] = BoundInfo(min_value=0, max_value=self.block_dims[0] - 1)
        if isinstance(self.block_dims[1], int):
            var4bound[threadIdx.y] = BoundInfo(min_value=0, max_value=self.block_dims[1] - 1)
        if isinstance(self.block_dims[2], int):
            var4bound[threadIdx.z] = BoundInfo(min_value=0, max_value=self.block_dims[2] - 1)
        if isinstance(self.grid_dims[0], int):
            var4bound[blockIdx.x] = BoundInfo(min_value=0, max_value=self.grid_dims[0] - 1)
        if isinstance(self.grid_dims[1], int):
            var4bound[blockIdx.y] = BoundInfo(min_value=0, max_value=self.grid_dims[1] - 1)
        if isinstance(self.grid_dims[2], int):
            var4bound[blockIdx.z] = BoundInfo(min_value=0, max_value=self.grid_dims[2] - 1)

        bound_dict: Dict[Expr, BoundInfo] = infer_bound(worker, var4bound)
        worker_max = bound_dict[worker].possible_max_value()

        is_need_bound_check = True
        if worker_max is not None:
            num_workers = worker_max + 1
            assert (
                task_size <= num_workers
            )  # task size must be not greater than number of workers. Usually they are equal
            if num_workers == task_size:
                is_need_bound_check = False

        if is_need_bound_check:
            cond = worker < task_size
            new_stmt = IfStmt(cond=cond, then_body=new_stmt)

        return new_stmt


class TaskMappingBoundCheck(Pass):
    def process_func(self, func: Function) -> Function:
        if 'cuda.block_dim' not in func.attrs or 'cuda.grid_dim' not in func.attrs:
            return func
        block_dims = normalize_launch_dims(func.attrs['cuda.block_dim'])
        grid_dims = normalize_launch_dims(func.attrs['cuda.grid_dim'])
        rewriter = TaskMappingBoundCheckRewriter(block_dims, grid_dims)
        return rewriter.rewrite(func)


# Check worker bound in task mapping
# Example:
# To task mapping loops
# ```
# # cuda.block_dim: 1024
# ...
# for i in spatial(128) on threadIdx.x
#   a[i] = 0.0f
# ````
# add bound check
# ```
#  # cuda.block_dim: 1024
# ...
# if threadIdx.x < 128
#    for i in spatial(128) on threadIdx.x
#        a[i] = 0.0f
# ```
# See details in CentML/hidet/issues/478
def task_mapping_bound_check():
    return TaskMappingBoundCheck()
