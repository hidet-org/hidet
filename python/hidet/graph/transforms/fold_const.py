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
from hidet import utils


class FoldConstantRewriter(GraphRewriter):
    def visit_Operator(self, op: Operator):
        inputs = [self(input) for input in op.inputs]
        if all(input.storage is not None for input in inputs):
            outputs = Operator.imperative_run(op, inputs)
            for original, updated in zip(op.outputs, outputs):
                self.memo[original] = updated
            return None
        else:
            if utils.same_list(inputs, op.inputs):
                return None
            else:
                updated_outputs = op.reforward(inputs)
                for original, updated in zip(op.outputs, updated_outputs):
                    self.memo[original] = updated
                return None


class FoldConstantPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        rewriter = FoldConstantRewriter()
        return rewriter(graph)


def fold_const_pass():
    return FoldConstantPass()
