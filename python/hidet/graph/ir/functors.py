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
from typing import Union, Type, Dict, List, Tuple, Optional
from collections import defaultdict
from hidet.graph.ir.flow_graph import FlowGraph, Operator, Tensor
from hidet.utils import same_list


class GraphVisitor:
    def __init__(self):
        self.memo = {}

    def __call__(self, obj):
        return self.visit(obj)

    def visit(self, obj: Union[FlowGraph, Operator, Tensor, list, tuple]):
        key = obj if not isinstance(obj, list) else id(obj)
        if self.memo is not None and obj in self.memo:
            return
        if isinstance(obj, FlowGraph):
            self.visit_FlowGraph(obj)
        elif isinstance(obj, Operator):
            self.visit_Operator(obj)
        elif isinstance(obj, Tensor):
            self.visit_Tensor(obj)
        elif isinstance(obj, (list, tuple)):
            self.visit_Sequence(obj)
        else:
            raise ValueError(type(obj))
        if self.memo is not None:
            self.memo[key] = None

    def visit_FlowGraph(self, graph: FlowGraph):
        for output in graph.outputs:
            self(output)

    def visit_Operator(self, op: Operator):
        for inp in op.inputs:
            self(inp)

    def visit_Tensor(self, tensor: Tensor):
        if tensor.trace is None:
            return
        self(tensor.trace[0])

    def visit_Sequence(self, seq: Union[list, tuple]):
        for obj in seq:
            self(obj)


class GraphRewriter:
    def __init__(self):
        self.memo = {}

    def __call__(self, obj):
        return self.visit(obj)

    def visit(self, obj: Union[FlowGraph, Operator, Tensor, list, tuple]):
        key = obj if not isinstance(obj, list) else id(obj)
        if self.memo is not None and obj in self.memo:
            return self.memo[key]
        if isinstance(obj, FlowGraph):
            ret = self.visit_FlowGraph(obj)
        elif isinstance(obj, Operator):
            ret = self.visit_Operator(obj)  # pylint: disable=assignment-from-none
        elif isinstance(obj, Tensor):
            ret = self.visit_Tensor(obj)
        elif isinstance(obj, (list, tuple)):
            ret = self.visit_Sequence(obj)
        else:
            raise ValueError(type(obj))
        if self.memo is not None:
            self.memo[key] = ret
        return ret

    def visit_FlowGraph(self, graph: FlowGraph):
        outputs = [self.visit(output) for output in graph.outputs]
        if same_list(outputs, graph.outputs):
            return graph
        else:
            return FlowGraph(outputs, graph.inputs)

    def visit_Operator(self, op: Operator):
        inputs = [self(input) for input in op.inputs]
        if same_list(inputs, op.inputs):
            return
        else:
            updated_outputs = op.clone(inputs)
            for original, updated in zip(op.outputs, updated_outputs):
                self.memo[original] = updated

    def visit_Tensor(self, tensor: Tensor):
        if tensor.trace is None:
            # input
            return tensor
        self(tensor.trace[0])
        if tensor in self.memo:
            # the operator has been updated
            return self.memo[tensor]
        else:
            return tensor

    def visit_Sequence(self, seq: Union[list, tuple]):
        return seq.__class__([self(obj) for obj in seq])


class GraphCloneRewriter(GraphRewriter):
    def visit_FlowGraph(self, graph: FlowGraph):
        outputs = [self.visit(output) for output in graph.outputs]
        return FlowGraph(outputs, graph.inputs)

    def visit_Operator(self, op: Operator):
        inputs = [self(x) for x in op.inputs]
        updated_outputs = op.clone(inputs)
        for original, updated in zip(op.outputs, updated_outputs):
            self.memo[original] = updated

    def visit_Tensor(self, tensor: Tensor):
        if tensor.trace is None:
            # keep the input tensor the same
            return tensor
        else:
            self(tensor.trace[0])
            return self.memo[tensor]


class GraphUsageAnalyzer(GraphVisitor):
    def __init__(self):
        super().__init__()
        self.usage: Dict[Tensor, List[Tuple[Optional[Operator], int]]] = defaultdict(list)

    def analyze(self, graph: FlowGraph):
        self.usage = defaultdict(list)
        self.visit(graph)
        return self.usage

    def visit_FlowGraph(self, graph: FlowGraph):
        for idx, output in enumerate(graph.outputs):
            self(output)
            self.usage[output].append((None, idx))
        GraphVisitor.visit_FlowGraph(self, graph)

    def visit_Operator(self, op: Operator):
        for idx, inp in enumerate(op.inputs):
            self.usage[inp].append((op, idx))
        GraphVisitor.visit_Operator(self, op)


def analyze_usage(graph: FlowGraph) -> Dict[Tensor, List[Tuple[Operator, int]]]:
    analyzer = GraphUsageAnalyzer()
    return analyzer.analyze(graph)


def clone(graph: FlowGraph):
    return GraphCloneRewriter().visit(graph)


def graph_collect(obj: Union[FlowGraph, Operator, Tensor], cls: Type[Union[Operator, Tensor]]):
    visitor = GraphVisitor()
    visitor.visit(obj)
    return [v for v in visitor.memo if isinstance(v, cls)]
