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
from typing import Union, Type, Dict, List, Tuple, Optional, Any, Sequence
from collections import defaultdict
from hidet.graph.ir.flow_graph import FlowGraph, Operator, Tensor
from hidet.graph.operator import Device, SizeVar
from hidet.utils import same_list
from hidet.ir.expr import Var, Expr
from hidet.ir.functors import IRRewriter


class AttributeRewriter(IRRewriter):
    def __init__(self, remap: Optional[Dict[Var, Expr]] = None):
        super().__init__()
        if remap is not None:
            self.update_remap(remap)

    def update_remap(self, remap: Dict[Var, Expr]):
        self.memo.update(remap)

    def visit_NotDispatched(self, n: Any):
        if isinstance(n, (FlowGraph, Device)):
            return n
        return super().visit_NotDispatched(n)

    def update_with_outputs(self, original_outputs: Sequence[Tensor], updated_outputs: Sequence[Tensor]):
        assert len(original_outputs) == len(updated_outputs)
        for original, updated in zip(original_outputs, updated_outputs):
            for original_dim, updated_dim in zip(original.shape, updated.shape):
                if isinstance(original_dim, Var):
                    self.memo[original_dim] = updated_dim


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
        self.attr_rewriter = AttributeRewriter()

    def __call__(self, obj):
        return self.visit(obj)

    def update_outputs(self, original_outputs: Sequence[Tensor], updated_outputs: Sequence[Tensor]):
        if len(original_outputs) != len(updated_outputs):
            raise RuntimeError("Length of original outputs and updated outputs are not equal.")
        for original, updated in zip(original_outputs, updated_outputs):
            self.memo[original] = updated
            for o_dim, u_dim in zip(original.shape, updated.shape):
                if isinstance(o_dim, Var):
                    self.attr_rewriter.update_remap({o_dim: u_dim})

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
        attrs = {k: self.attr_rewriter(v) for k, v in op.attrs.items()}
        if same_list(inputs, op.inputs) and same_list(attrs.values(), op.attrs.values()):
            return
        else:
            updated_outputs = op.reforward(inputs, attrs)
            self.update_outputs(op.outputs, updated_outputs)

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
        attrs = {k: self.attr_rewriter(v) for k, v in op.attrs.items()}
        updated_outputs = op.reforward(inputs, attrs)
        self.update_outputs(op.outputs, updated_outputs)

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


class GraphFreeSizeVarCollector(GraphVisitor):
    def __init__(self):
        super().__init__()
        self.defined_size_vars = set()
        self.free_size_vars = set()

    def collect(self, graph: FlowGraph):
        self.free_size_vars = set()
        self.visit(graph)
        return self.free_size_vars

    def visit_Operator(self, op: Operator):
        from hidet.ir.tools import collect

        GraphVisitor.visit_Operator(self, op)
        size_vars = set(collect(op.attrs, SizeVar))
        self.free_size_vars.update(size_vars - self.defined_size_vars)


def analyze_usage(graph: FlowGraph) -> Dict[Tensor, List[Tuple[Operator, int]]]:
    """
    Analyze the usage status of each tensor.

    Parameters
    ----------
    graph: FlowGraph
        The graph to be analyzed.

    Returns
    -------
    ret: Dict[Tensor, List[Tuple[Operator, int]]]
        The usage status of each tensor. ret[tensor] is a list of (operator, index) pairs, indicating that the tensor
        is the index-th input of the operator (when index >= 0), or the tensor is dependent by the operator
        (when `index == -1`).

    """
    analyzer = GraphUsageAnalyzer()
    return analyzer.analyze(graph)


def clone(graph: FlowGraph):
    return GraphCloneRewriter().visit(graph)


def graph_collect(obj: Union[FlowGraph, Operator, Tensor], cls: Type[Union[Operator, Tensor]]):
    visitor = GraphVisitor()
    visitor.visit(obj)
    return [v for v in visitor.memo if isinstance(v, cls)]


def collect_free_size_vars(graph: FlowGraph) -> List[SizeVar]:
    return list(GraphFreeSizeVarCollector().collect(graph))
