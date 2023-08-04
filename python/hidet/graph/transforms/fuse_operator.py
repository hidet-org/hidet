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
from typing import List, Sequence, Dict, Tuple, Optional, Set, Union

import hidet
from hidet.graph.flow_graph import FlowGraph, Operator, Tensor
from hidet.graph.ops.special import BarrierOp
from hidet.graph.ops.transfer import TransferOp
from hidet.graph.ops.distributed import AllReduceOp, AllGatherOp, ReduceScatterOp
from hidet.graph.graph_utils.functors import analyze_usage
from hidet.graph.transforms.base import GraphPass
from hidet.utils.structure import DirectedGraph
from hidet.utils.doc import Doc, Text, NewLine, doc_join
from hidet.utils.namer import Namer
from hidet.utils.py import unique


# the following operators are not fusible
NOT_FUSIBLE = {BarrierOp, TransferOp, AllReduceOp, AllGatherOp, ReduceScatterOp}


class FusibleGraph:
    def __init__(self, anchor: Operator):
        self.anchor: Operator = anchor
        self.operators: List[Operator] = [anchor]
        self.input_tensors: List[Tensor] = unique(anchor.inputs)  # remove duplicates
        self.output_tensors: List[Tensor] = list(anchor.outputs)

    def __str__(self):
        namer = Namer()
        args_doc = doc_join([namer(t) for t in self.input_tensors], ', ')
        head = Text('FusibleGraph(') + args_doc + Text(') {')
        body = Doc()
        for op in self.operators:
            output_doc = doc_join([namer(t) for t in op.outputs], ', ')
            input_doc = doc_join([namer(t) for t in op.inputs], ', ')
            body += NewLine() + output_doc + ' = ' + op.name + '(' + input_doc + ')'
        body += NewLine() + 'return ' + doc_join([namer(t) for t in self.output_tensors], ', ')
        tail = NewLine() + '}'
        return str(head + body.indent() + tail)

    def flow_graph_and_anchor(self) -> Tuple[FlowGraph, int]:
        from hidet.graph import symbol_like

        graph_inputs: List[Tensor] = [symbol_like(x, shape=x.shape, layout=x.layout) for x in self.input_tensors]

        remap: Dict[Tensor, Tensor] = {x: y for x, y in zip(self.input_tensors, graph_inputs)}
        updated_anchor: Optional[Operator] = None
        for op in self.operators:
            inputs = [remap[x] for x in op.inputs]
            outputs = op.reforward(inputs)
            if op is self.anchor:
                updated_anchor = outputs[0].op
            for x, y in zip(op.outputs, outputs):
                remap[x] = y
        graph_outputs = [remap[x] for x in self.output_tensors]
        flow_graph = hidet.trace_from(graph_outputs, graph_inputs)

        assert updated_anchor is not None
        anchor_idx = flow_graph.nodes.index(updated_anchor)
        return flow_graph, anchor_idx


Usage = Dict[Tensor, List[Tuple[Operator, int]]]


def fuse_epilogue_operators(anchors: Sequence[Operator], usage: Usage, belong: Dict[Operator, FusibleGraph]):
    for anchor in anchors:
        if not anchor.task.allow_epilogue():
            # this anchor operator does not allow epilogue fusion, skip
            continue

        if type(anchor) in NOT_FUSIBLE:
            # for operator are not fusible, skip
            continue

        sub_graph: FusibleGraph = belong[anchor]

        while True:
            found_output_tensor: Optional[Tensor] = None

            for output_tensor in sub_graph.output_tensors:
                use: List[Tuple[Operator, int]] = usage[output_tensor]
                if len(use) != 1:
                    # expect used only once, skip
                    continue

                user: Operator = use[0][0]
                if user is None:
                    # this tensor is an output of flow graph, skip
                    continue

                if type(user) in NOT_FUSIBLE:
                    # for operator are not fusible, skip
                    continue

                if user in belong:
                    # this tensor has been fused, skip
                    continue

                input_index: int = use[0][1]
                if user.task.inputs[input_index] not in user.task.inverse_map:
                    # this tensor is not invertible towards its output, skip
                    continue

                if len(user.inputs) > 1:
                    other_inputs: List[Tensor] = [tensor for tensor in user.inputs if tensor is not output_tensor]
                    if any(any(tensor is v for v in sub_graph.output_tensors) for tensor in other_inputs):
                        # the user operator has other inputs that read the output of the sub_graph, skip
                        continue

                # The tensor is invertible, used by only one time, and not fused yet, we found an output to fuse.
                # Because the fusion will update sub_graph and its output_tensors that we are iterating,
                # we break here and fuse out of the for loop.
                found_output_tensor = output_tensor
                break

            if found_output_tensor is None:
                break
            user: Operator = usage[found_output_tensor][0][0]
            sub_graph.operators.append(user)
            sub_graph.output_tensors.remove(found_output_tensor)
            sub_graph.output_tensors.append(user.outputs[0])
            sub_graph.input_tensors.extend(
                [
                    tensor
                    for tensor in unique(user.inputs)
                    if (all(tensor is not v for v in sub_graph.input_tensors) and tensor is not found_output_tensor)
                ]
            )
            belong[user] = sub_graph
            # continue the while loop to fuse more operators


def fuse_prologue_operators(anchors: Sequence[Operator], usage: Usage, belong: Dict[Operator, FusibleGraph]):
    for anchor in anchors:
        if not anchor.task.allow_prologue():
            # this anchor operator does not allow prologue fusion, skip
            continue

        if type(anchor) in NOT_FUSIBLE:
            # for operator are not fusible, skip
            continue

        sub_graph: FusibleGraph = belong[anchor]

        while True:
            found_input_tensor: Optional[Tensor] = None

            for input_tensor in sub_graph.input_tensors:
                used_ops: List[Operator] = [use[0] for use in usage[input_tensor]]
                if any(op not in belong or belong[op] is not sub_graph for op in used_ops):
                    # used by other operators not in current sub-graph, skip
                    continue

                if input_tensor.trace is None:
                    # this tensor is an input of FlowGraph, skip
                    continue

                producer: Operator = input_tensor.trace[0]
                if producer in belong:
                    # this tensor has been fused, skip
                    continue

                if type(producer) in NOT_FUSIBLE:
                    # for operator are not fusible, skip
                    continue

                # The tensor is used by only one time, and not fused yet, we found an input to fuse.
                # Same as the epilogue fusion, we break here and fuse out of the for loop.
                found_input_tensor = input_tensor
                break

            if found_input_tensor is None:
                break
            producer: Operator = found_input_tensor.trace[0]
            sub_graph.operators.insert(0, producer)
            sub_graph.input_tensors = [v for v in sub_graph.input_tensors if v is not found_input_tensor]
            sub_graph.input_tensors.extend(
                [tensor for tensor in unique(producer.inputs) if all(tensor is not v for v in sub_graph.input_tensors)]
            )
            belong[producer] = sub_graph
            # continue the while loop to fuse more operators


def topological_order(
    anchors: Sequence[Operator], usage: Usage, belong: Dict[Operator, FusibleGraph]
) -> List[FusibleGraph]:
    dag = DirectedGraph()
    sub_graphs: List[FusibleGraph] = [belong[anchor] for anchor in anchors]
    for src in sub_graphs:
        dag.add_node(src)
        for tensor in src.output_tensors:
            for user, _ in usage[tensor]:
                if user is None:
                    # graph output usage
                    continue
                dst = belong[user]
                dag.add_edge(src, dst)
    return dag.topological_order()


def sanity_check_partition(graph: FlowGraph, partition: List[FusibleGraph], belong: Dict[Operator, FusibleGraph]):
    # first, check if
    # 1. all operators are in the partition, and
    # 2. the partition contains all operators
    partition_union: Set[Operator] = set()
    for sub_graph in partition:
        partition_union.update(sub_graph.operators)
    assert len(partition_union.symmetric_difference(graph.nodes)) == 0, "partition is not a partition"

    # second, check if the edge relation is correct, see the assertions for details
    for consumer_node in graph.nodes:
        for input_tensor in consumer_node.inputs:
            if input_tensor.trace is None:
                # graph input
                continue
            producer_node, idx = input_tensor.trace  # pylint: disable=unused-variable
            producer_subgraph: FusibleGraph = belong[producer_node]
            consumer_subgraph: FusibleGraph = belong[consumer_node]
            if producer_subgraph is consumer_subgraph:
                # it is an intra-sub-graph edge
                sub_graph = producer_subgraph
                assert all(
                    input_tensor is not v for v in sub_graph.input_tensors
                ), "intra-sub-graph edge should not be an input tensor of its sub-graph"
                assert all(
                    input_tensor is not v for v in sub_graph.output_tensors
                ), "intra-sub-graph edge should not be an output tensor of its sub-graph"
            else:
                # it is an inter-sub-graph edge
                assert any(
                    input_tensor is v for v in producer_subgraph.output_tensors
                ), "inter-sub-graph edge does not start in producer subgraph's output_tensors"
                assert any(
                    input_tensor is v for v in consumer_subgraph.input_tensors
                ), "inter-sub-graph edge does not end in consumer subgraph's input_tensors"


def partition_graph(graph: FlowGraph, usage: Usage) -> List[FusibleGraph]:
    if graph.nodes is None:
        # graph.nodes is the cache of operators in the traced graph of graph.outputs.
        # some pass will invalidate the cache, so we need to re-trace the graph.
        graph.update_nodes()

    belong: Dict[Operator, FusibleGraph] = {}

    # first, we find all non-injective operators as the anchor operators,
    # and create a sub-graph for each such operator.
    anchors: List[Operator] = [op for op in graph.nodes if not op.task.is_injective()]
    for anchor in anchors:
        belong[anchor] = FusibleGraph(anchor)

    # fuse epilogue operators.
    fuse_epilogue_operators(anchors, usage, belong)

    # fuse prologue operators.
    fuse_prologue_operators(anchors, usage, belong)

    # there may be some injective operators left and stay together.
    # Assign the end of such connected area and apply prologue fusion.
    for node in reversed(graph.nodes):
        if node in belong:
            continue
        belong[node] = FusibleGraph(node)
        anchors.append(node)
        fuse_prologue_operators([node], usage, belong)

    # we have partitioned the graph into sub-graphs, each operator belongs to a sub-graph.
    # we return the sub-graphs in a topological order.
    partition: List[FusibleGraph] = topological_order(anchors, usage, belong)

    # sanity check that it is a partition
    sanity_check_partition(graph, partition, belong)

    return partition


def operator_from_sub_graph(sub_graph: FusibleGraph, tensor_remap: Dict[Tensor, Tensor]) -> Operator:
    if len(sub_graph.operators) == 1:
        # if there is only one operator in the sub-graph, we just update its inputs.
        origin_op = sub_graph.operators[0]
        updated_inputs: List[Tensor] = [
            tensor_remap[tensor] if tensor in tensor_remap else tensor for tensor in origin_op.inputs
        ]
        outs: List[Tensor] = origin_op.reforward(updated_inputs)
        updated_op = outs[0].trace[0]
        return updated_op
    else:
        # otherwise, create a new operator from the sub-graph.
        from hidet.graph.ops.fusion.fused_operator import fused_operator

        fused_graph, anchor = sub_graph.flow_graph_and_anchor()
        updated_inputs: List[Tensor] = [
            tensor_remap[tensor] if tensor in tensor_remap else tensor for tensor in sub_graph.input_tensors
        ]
        outs: Union[List[Tensor], Tensor] = fused_operator(*updated_inputs, fused_graph=fused_graph, anchor=anchor)
        if isinstance(outs, Tensor):
            outs = [outs]
        return outs[0].trace[0]


def construct_fused_graph(graph: FlowGraph, sub_graphs: Sequence[FusibleGraph]) -> FlowGraph:
    graph_input_tensors: List[Tensor] = graph.inputs
    graph_nodes: List[Operator] = []
    tensor_remap: Dict[Tensor, Tensor] = {}

    for sub_graph in sub_graphs:
        op = operator_from_sub_graph(sub_graph, tensor_remap)
        tensor_remap.update({a: b for a, b in zip(sub_graph.output_tensors, op.outputs)})
        for original, updated in zip(sub_graph.output_tensors, op.outputs):
            tensor_remap[original] = updated
        graph_nodes.append(op)

    graph_output_tensors: List[Tensor] = [
        tensor_remap[tensor] if tensor in tensor_remap else tensor for tensor in graph.outputs
    ]
    return FlowGraph(graph_output_tensors, inputs=graph_input_tensors, nodes=graph_nodes)


class FuseOperatorPass(GraphPass):
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        usage: Usage = analyze_usage(graph)
        partition: List[FusibleGraph] = partition_graph(graph, usage)
        fused_graph = construct_fused_graph(graph, partition)
        fused_graph.update_nodes()
        return fused_graph


def fuse_operator_pass() -> GraphPass:
    return FuseOperatorPass()
