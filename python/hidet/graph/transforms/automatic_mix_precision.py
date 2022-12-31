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
from typing import List, Union
from hidet.graph.ir.functors import GraphRewriter
from hidet.graph.ir.flow_graph import FlowGraph, Operator, Tensor
from hidet.graph import ops
from hidet.ir.type import DataType, data_type
from hidet.utils import strict_zip, same_list
from .base import GraphPass


class DefaultTransformStrategy:
    always = [
        # defs.Conv2dOp, defs.MatmulOp, defs.AddOp, defs.SubOp, defs.MultiplyOp, defs.DivideOp, defs.PadOp
    ]
    never = [
        # defs.ErfOp, defs.PowOp,
        # defs.ReduceSumOp, defs.ReduceMeanOp  # disable because we can accumulate with higher precision
    ]


class AutoMixPrecisionRewriter(GraphRewriter):
    def __init__(self, target_dtype: Union[str, DataType]):
        super().__init__()
        self.target_dtype: DataType = data_type(target_dtype)
        self.policy = DefaultTransformStrategy()

        assert self.target_dtype.is_float()

    @staticmethod
    def cast_float(x: Tensor, target_dtype: DataType) -> Tensor:
        if x.dtype.is_float():
            return ops.cast(x, target_dtype)
        return x

    def visit_Operator(self, op: Operator):
        recv_inputs: List[Tensor] = [self(v) for v in op.inputs]

        # if type(op) in self.policy.always:
        #     decision = 'always'
        if type(op) in self.policy.never:
            decision = 'never'
        else:
            decision = 'always'

        casted_inputs = []
        for orig_input, recv_input in strict_zip(op.inputs, recv_inputs):
            if decision == 'always':
                casted_inputs.append(self.cast_float(recv_input, self.target_dtype))
            elif decision == 'never':
                casted_inputs.append(self.cast_float(recv_input, orig_input.dtype))
            else:
                casted_inputs.append(recv_input)
        if same_list(casted_inputs, op.inputs):
            return op
        else:
            updated_outputs = op.reforward(casted_inputs)
            for original, updated in zip(op.outputs, updated_outputs):
                self.memo[original] = updated
            return None


class AutoMixPrecisionPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        target_dtype: Union[str, DataType] = self.current_context().configs['precision']
        if target_dtype is None:
            return graph
        else:
            rewriter = AutoMixPrecisionRewriter(target_dtype)
            graph = rewriter(graph)
            return graph


def automatic_mix_precision_pass() -> GraphPass:
    return AutoMixPrecisionPass()
