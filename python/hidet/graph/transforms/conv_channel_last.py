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

from typing import List, Dict, Callable, Tuple, Iterable

from hidet.graph.operator import Operator, Tensor
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import GraphPass
from hidet.graph.graph_utils.functors import analyze_usage


class ConvChannelLastPass(GraphPass):

    def span_from_nodes(
        self, usage: Dict[Tensor, List[Tuple[Operator, int]]], seeds: List[Operator],
    ) -> List[Operator]:
        # span from the seed operators, return all operators that connect to the seed operators through
        # the operators that satisfy the condition f_cond
        def connected_operators(op: Operator) -> Iterable[Operator]:
            for x in op.inputs:
                if x.trace is not None:
                    yield x.trace[0]
            for y in op.outputs:
                for user, idx in usage[y]:
                    yield user
        pass

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        from hidet.graph.ops.conv2d import Conv2dOp
        nodes: List[Operator] = graph.nodes
        # Start from all conv2d operators as seeds
        seeds = [node for node in nodes if isinstance(node, Conv2dOp)]

        # Only use this pass if there is convolution in the graph
        if len(seeds) == 0:
            return graph

        # Get the usage of each Tensor
        usage: Dict[Tensor, List[Tuple[Operator, int]]] = analyze_usage(graph)
        # Use the usage to trace through all operators spanning from the seeds
        scope_nodes = self.span_from_nodes(usage=usage, seeds=seeds)
        return graph

        #
        perm: Dict[Tensor, List[int]] = {}
        tensor_map: Dict[Tensor, Tensor] = {}
        
        for node in nodes:
            if node in scope_nodes:
                # prepare inputs
                updated_inputs: List[Tensor] = ...
                # op transform
                # batch norm, activation, ...
                # update tensor_map
                updated_outputs = ...
                for original, updated in zip(node.outputs, updated_outputs):
                    tensor_map[original] = updated
            else:
                # prepare inputs by converting back
                # reforward
                # update tensor_map
                pass
        
        graph_inputs = [tensor_map[input] for input in graph.inputs]
        graph_outputs = [tensor_map[output] for output in graph.outputs]
        return FlowGraph(graph_outputs, graph_inputs)


def conv_channel_last_pass() -> GraphPass:
    return ConvChannelLastPass()
