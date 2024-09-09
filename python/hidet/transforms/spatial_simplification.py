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
from hidet.transforms.base import FunctionPass
from hidet.ir import ForMappingStmt
from hidet.ir.dtypes import int32
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.mapping import SpatialTaskMapping
from hidet.ir.tools import MapBasedRewriter
from hidet.lang import spatial


class SpatialSimplificationRewriter(IRRewriter):
    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        body = self.visit(stmt.body)
        changed = False
        if isinstance(stmt.mapping, SpatialTaskMapping):
            map = {}
            shape = []
            loop_vars = []
            for i, size in enumerate(stmt.mapping.task_shape):
                if size == 1:
                    map[stmt.loop_vars[i]] = int32.zero
                    changed = True
                else:
                    shape.append(stmt.mapping.task_shape[i])
                    loop_vars.append(stmt.loop_vars[i])
            if len(map) != 0:
                rewriter = MapBasedRewriter(map)
                body = rewriter.rewrite(body)

        if body is stmt.body and not changed:
            return stmt
        else:
            return ForMappingStmt(loop_vars=loop_vars, mapping=spatial(*shape), worker=stmt.worker, body=body)


class SpatialSimplificationPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = SpatialSimplificationRewriter()
        return rewriter.rewrite(func)


# This pass transform mapping spacial(1, 47) to spacial (47)
# I.e. remove dims with size 1
# It helps to simplify indexes calculation
def spatial_simplification_pass():
    return SpatialSimplificationPass()
