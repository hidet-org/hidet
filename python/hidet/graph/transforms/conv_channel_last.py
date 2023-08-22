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

from typing import List, Dict, Callable, Tuple, Iterable, Optional

from hidet.graph.operator import Operator, Tensor
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import GraphPass
from hidet.graph.graph_utils.functors import analyze_usage


class PermutedOp:
    def __init__(self, op: Operator) -> None:
        self.op = op

    def reforward(self, tensor_map) -> None:
        from hidet.graph.ops.conv2d import Conv2dOp
        from hidet.graph.ops.transform import transpose
        node = self.op
        new_inputs: List[Tensor] = []
        update_attributes = {}
        for x in node.inputs:
            if x in tensor_map:
                current_x, current_perm = tensor_map[x]
                if current_perm is not None:
                    new_x = current_x
                else:
                    new_perm = [0, 2, 3, 1]
                    new_x = transpose(current_x, new_perm)
                    tensor_map[x] = (new_x, new_perm)
                new_inputs.append(new_x)
            else:
                new_perm = [0, 2, 3, 1]
                new_x = transpose(x, new_perm)
                tensor_map[x] = (new_x, new_perm)
                new_inputs.append(new_x)
        outputs = node.reforward(new_inputs, update_attributes)
        for idx, y in enumerate(node.outputs):
            tensor_map[y] = (outputs[idx], new_perm)
    
    @staticmethod
    def get_permuted_op(op: Operator):
        from hidet.graph.ops.conv2d import Conv2dOp
        from hidet.graph.ops.arithmetic import AddOp
        if isinstance(op, Conv2dOp):
            return PermutedConv2dOp(op)
        elif isinstance(op, AddOp):
            return PermutedOp(op)
        else:
            raise RuntimeError("PermutedOp.get_permuted_op() got invalid Operator.")


class PermutedConv2dOp(PermutedOp):
    def __init__(self, op: Operator) -> None:
        super().__init__(op)
    
    def reforward(self, tensor_map) -> None:
        from hidet.graph.ops.transform import transpose
        node = self.op
        new_inputs: List[Tensor] = []
        update_attributes = {}
        for x in node.inputs:
            if x in tensor_map:
                current_x, current_perm = tensor_map[x]
                if current_perm is not None:
                    new_x = current_x
                else:
                    new_perm = [0, 2, 3, 1]
                    new_x = transpose(current_x, new_perm)
                    tensor_map[x] = (new_x, new_perm)
                new_inputs.append(new_x)
            else:
                new_perm = [0, 2, 3, 1]
                new_x = transpose(x, new_perm)
                tensor_map[x] = (new_x, new_perm)
                new_inputs.append(new_x)
        outputs = node.reforward(new_inputs, update_attributes)
        for idx, y in enumerate(node.outputs):
            tensor_map[y] = (outputs[idx], new_perm)


class ConvChannelLastPass(GraphPass):
    """
    For a graph with convolution, convert all image tensors to channel last whereever possible.
    This is accomplished by:
        1. Find a list of operators within scope (see within_scope() below)
        2. For operators within scope, transform its input to channel last and record this permutation.
        3. For operators out of scope, convert its input back to channel first based on recorded permuations.
    """

    def within_scope(self, op: Operator) -> bool:
        """
        Given input tensor x[n, c, h, w], an operator Op is within scope if
            1. x is an input to Op, and the output tensor is also of shape [n, c, h, w]
            2. P is the set of attributes of Op; x_chnlast = x.transpose[0, 2, 3, 1];
               and there exists another set of attributes P', such that 
               Op(x_chnlast, P')).transpose([0, 3, 1, 2]) == Op(x, P)
        """
        from hidet.graph.ops.arithmetic import AddOp
        if isinstance(op, AddOp):
            return True
        return False

    def span_from_nodes(
        self, usage: Dict[Tensor, List[Tuple[Operator, int]]], seeds: List[Operator],
    ) -> List[Operator]:
        # span from the seed operators, return all operators that are connect to the
        # seed operators and also within the predefined scope of this pass. See within_scope()
        def connected_operators(op: Operator) -> Iterable[Operator]:
            for x in op.inputs:
                if x.trace is not None:
                    yield x.trace[0]
            for y in op.outputs:
                for user, idx in usage[y]:
                    yield user
        
        scope_nodes = set()
        for op in seeds:
            scope_nodes.add(op)
            for connected in connected_operators(op):
                if self.within_scope(connected):
                    scope_nodes.add(connected)
        return scope_nodes

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        from hidet.graph.ops.conv2d import Conv2dOp
        from hidet.graph.ops.transform import transpose
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

        # Map a Tensor from itself to its current instance and permutation
        tensor_map: Dict[Tensor, Tuple[Tensor, Optional[List[int]]]] = {}
        
        # Iterate through nodes in topological order
        for node in nodes:
            if node in scope_nodes:
                # reforward node with op-specific changes and update tensor map
                permuted_node = PermutedOp.get_permuted_op(node)
                permuted_node.reforward(tensor_map)
            else:
                # Node is not within scope. If its inputs are permuted,
                # need to convert back, reforward, and update mappings
                need_to_reforward = False
                new_inputs: List[Tensor] = []
                update_attributes = {}
                for x in node.inputs:
                    if x in tensor_map:
                        need_to_reforward = True
                        current_x, current_perm = tensor_map[x]
                        if current_perm is not None:
                            to_orig_perm = [current_perm.index(i) for i in range(len(current_perm))]
                            new_x = transpose(current_x, to_orig_perm)
                        else:
                            new_x = current_x 
                        tensor_map[x] = (new_x, None)
                        new_inputs.append(new_x)
                    else:
                        new_inputs.append(x)
                if need_to_reforward:
                    outputs = node.reforward(new_inputs, update_attributes)
                    for idx, y in enumerate(node.outputs):
                        tensor_map[y] = (outputs[idx], None)
        
        graph_inputs = [tensor_map[input_tensor][0]
                        if input_tensor in tensor_map else input_tensor for input_tensor in graph.inputs]
        graph_outputs = [tensor_map[output_tensor][0]
                         if output_tensor in tensor_map else output_tensor for output_tensor in graph.outputs]
        ret =  FlowGraph(graph_outputs, graph_inputs)
        print(ret)
        return ret


def conv_channel_last_pass() -> GraphPass:
    return ConvChannelLastPass()
