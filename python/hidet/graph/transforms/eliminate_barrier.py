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
from hidet.graph.ir import FlowGraph, Operator, GraphRewriter
from hidet.graph.transforms import GraphPass

from .utils import is_barrier


class EliminateBarrierRewriter(GraphRewriter):
    def visit_Operator(self, op: Operator):
        inputs = [self(x) for x in op.inputs]

        if is_barrier(op):
            outputs = inputs
            for original, updated in zip(op.outputs, outputs):
                self.memo[original] = updated
            return None
        else:
            return GraphRewriter.visit_Operator(self, op)


class EliminateBarrierPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewriter = EliminateBarrierRewriter()
        return rewriter(graph)


def eliminate_barrier_pass():
    return EliminateBarrierPass()
