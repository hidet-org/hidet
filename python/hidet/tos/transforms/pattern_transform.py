from typing import List, Optional, Dict, Any, Union, Tuple, Type, Set
from hidet.tos.ir.graph import FlowGraph, Operator, Tensor
from hidet.tos.transforms import GraphPass
# from hidet.tos import ops
from hidet.tos import operators as ops
from hidet import utils
from hidet import tos

from .common import analyze_usage, graph_collect

from .fold_const import fold_const_pass


class TensorPattern:
    def __init__(self, is_const=False, is_symbolic=False, trace=None):
        self.is_const: bool = is_const
        self.is_symbolic: bool = is_symbolic
        assert not (is_const and is_symbolic), 'Can not be const and symbolic at the same time'
        self.trace: Optional[Tuple[OperatorPattern, int]] = trace

    def __repr__(self):
        if self.trace is None:
            if self.is_const:
                return 'c'
            if self.is_symbolic:
                return 's'
            return 'v'
        else:
            op, idx = self.trace
            op_str = str(op)
            if len(op.outputs) == 1:
                return op_str
            else:
                return '{}[{}]'.format(op_str, idx)

    def __add__(self, other):
        return OperatorPattern(ops.basic.arithmatic.AddOp, inputs=[self, other]).outputs[0]

    def __sub__(self, other):
        return OperatorPattern(ops.basic.arithmatic.SubOp, inputs=[self, other]).outputs[0]

    def __mul__(self, other):
        return OperatorPattern(ops.basic.arithmatic.MultiplyOp, inputs=[self, other]).outputs[0]

    def __neg__(self):
        return OperatorPattern(ops.basic.arithmatic.NegOp, inputs=[self]).outputs[0]

    @staticmethod
    def tensor(is_const=False, is_symbolic=False):
        return TensorPattern(is_const, is_symbolic)

    @staticmethod
    def tensors(num, is_const=False, is_symbolic=False):
        return [TensorPattern(is_const, is_symbolic) for _ in range(num)]


class OperatorPattern:
    def __init__(self, op_cls, inputs, num_outputs=1):
        self.op_cls = op_cls
        self.inputs: List[TensorPattern] = inputs
        self.outputs = [TensorPattern(is_symbolic=True, trace=(self, idx)) for idx in range(num_outputs)]

    def __repr__(self):
        input_items = [str(v) for v in self.inputs]
        unary_ops = {
            ops.basic.arithmatic.NegOp: '-'
        }
        binary_ops = {
            ops.basic.arithmatic.AddOp: '+',
            ops.basic.arithmatic.SubOp: '-',
            ops.basic.arithmatic.MultiplyOp: '*'
        }
        if self.op_cls in unary_ops:
            return '({}{})'.format(unary_ops[self.op_cls], input_items[0])
        elif self.op_cls in binary_ops:
            return '({} {} {})'.format(input_items[0], binary_ops[self.op_cls], input_items[1])
        else:
            return '{}({})'.format(self.op_cls.__class__[:-2], ', '.join(input_items))


def conv2d_pattern(x: TensorPattern, w: TensorPattern) -> TensorPattern:
    return OperatorPattern(ops.nn.conv.Conv2dOp, [x, w]).outputs[0]


class NotMatchedException(Exception):
    pass


class PatternMatcher:
    def __init__(self):
        self.matched = {}

    def check(self, cond: bool, msg=""):
        if not cond:
            raise NotMatchedException(msg)

    def match(self, pattern, target):
        key = pattern if not isinstance(pattern, list) else id(pattern)
        if key in self.matched:
            self.check(target is self.matched[key], 'tried to match a pattern to two different objects')
            # pattern has been matched to a different target
            return
        self.matched[key] = target
        if isinstance(pattern, (list, tuple)):
            self.match_Sequence(pattern, target)
        elif isinstance(pattern, TensorPattern):
            self.match_TensorPattern(pattern, target)
        elif isinstance(pattern, OperatorPattern):
            self.match_OperatorPattern(pattern, target)
        else:
            raise NotImplementedError()

    def match_Sequence(self, pattern, target):
        self.check(isinstance(target, (list, tuple)), 'target should be tuple or list')
        self.check(len(pattern) == len(target), 'sequence length does not match')
        for a, b in zip(pattern, target):
            self.match(a, b)

    def match_TensorPattern(self, pattern: TensorPattern, target):
        self.check(isinstance(target, Tensor), "expect target with type 'Tensor'")
        if pattern.is_const:
            self.check(target.storage is not None, 'requires const tensor')
        if pattern.is_symbolic:
            self.check(target.storage is None, 'requires symbolic tensor')
        if pattern.trace:
            self.check(target.trace is not None)
            self.check(pattern.trace[1] == target.trace[1])
            self.match(pattern.trace[0], target.trace[0])

    def match_OperatorPattern(self, pattern: OperatorPattern, target: Operator):
        self.check(isinstance(target, pattern.op_cls), "expect target with type 'Operator'")
        self.check(pattern.op_cls is target.__class__, 'operator cls does not match')
        assert len(pattern.inputs) == len(target.inputs) and len(pattern.outputs) == len(target.outputs)
        for a, b in zip(pattern.inputs, target.inputs):
            self.match(a, b)
        for a, b in zip(pattern.outputs, target.outputs):
            self.match(a, b)


def match(pattern, target) -> Optional[Dict]:
    # short-cut for early stop
    if pattern.trace is None:
        if target.trace is not None:
            return None
        if (pattern.is_const and target.storage is None) or (pattern.is_symbolic and target.storage is not None):
            return None
        return {pattern: target}
    if pattern.trace and target.trace and pattern.trace[0].op_cls is not target.trace[0].__class__:
        return None
    matcher = PatternMatcher()
    try:
        matcher.match(pattern, target)
        return matcher.matched
    except NotMatchedException:
        return None


class GraphPattern:
    def source(self) -> TensorPattern:
        raise NotImplementedError()

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        raise NotImplementedError()


class GraphConstructor:
    def __init__(self, matched):
        self.memo = {}
        self.matched = matched
        self.new_operators = []

    def visit(self, obj: Union[TensorPattern, OperatorPattern]):
        if obj in self.memo:
            return self.memo[obj]
        if isinstance(obj, OperatorPattern):
            ret = self.visit_OperatorPattern(obj)
        elif isinstance(obj, TensorPattern):
            ret = self.visit_TensorPattern(obj)
        else:
            raise ValueError()
        self.memo[obj] = ret
        return ret

    def visit_TensorPattern(self, t: TensorPattern) -> Tensor:
        if t.trace is None:
            # input in pattern
            return self.matched[t]
        else:
            op, idx = t.trace
            return self.visit(op).get_output(idx)

    def visit_OperatorPattern(self, t: OperatorPattern) -> Operator:
        inputs = [self.visit(input) for input in t.inputs]
        op = t.op_cls(*inputs)
        self.new_operators.append(op)
        return op


class SimpleGraphPattern(GraphPattern):
    def __init__(self, source, target):
        self.src = source
        self.tgt = target

    @staticmethod
    def all() -> List[GraphPattern]:
        # tensors can be used as pattern inputs
        x, y, z = TensorPattern.tensors(3, is_symbolic=True)  # can not be const
        a, b, c = TensorPattern.tensors(3, is_const=True)  # can not be symbolic

        # (source, target) pattern pairs
        pairs = [
            [a + x, x + a],
            [x - a, x + (-a)],
            [(x + a) * b, x * b + a * b],
            [(x + a) + (y + b), (x + y) + (a + b)],
        ]
        return [SimpleGraphPattern(src, tgt) for src, tgt in pairs]

    def source(self) -> TensorPattern:
        return self.src

    def target(self, matched: Dict) -> Optional[TensorPattern]:
        constructor = GraphConstructor(matched)
        return constructor.visit(self.tgt)


class ConvMultiplyPattern(GraphPattern):
    def __init__(self):
        x = TensorPattern.tensor()
        w, scale = TensorPattern.tensors(2, is_const=True)
        conv = conv2d_pattern(x, w)
        self.x = x
        self.w = w
        self.scale = scale
        self.conv_op = conv.trace[0]
        self.src = conv * scale

    def source(self) -> TensorPattern:
        return self.src

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        x, w, scale = matched[self.x], matched[self.w], matched[self.scale]
        conv_attrs = matched[self.conv_op].attributes
        if not (scale.shape[0] == scale.shape[2] == scale.shape[3] == 1):
            # we can only fuse the scale on channels
            return None
        ww = w * scale.squeeze((0, 2, 3)).unsqueeze((1, 2, 3))
        return ops.conv2d(x, ww, **conv_attrs)


def all_patterns() -> List[GraphPattern]:
    simple_patterns = SimpleGraphPattern.all()
    other_patterns = [
        ConvMultiplyPattern()
    ]
    return simple_patterns + other_patterns


class PatternTransformPass(GraphPass):
    """
    A pattern transform can be conducted only if
    1. The pattern source matched the actual tensor and its spanned subregion.
    2. The intermediate tensor in the matched region should not be used.
        For example, if pattern a -> b -> c matched x -> y -> z. We need to make sure y has not been
        used by other operators in the original graph.

    Time complexity of this implementation: O(num_applies * num_operators * num_patterns * pattern_size).
    """
    max_num_transforms = 1000

    # @utils.line_profile()
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        graph = tos.ir.functors.clone(graph)
        fold_const = fold_const_pass()
        for t in range(self.max_num_transforms):
            updated, graph = self.try_transform(graph)
            graph = fold_const.process_graph(graph)
            if not updated:
                return graph
        print('Exceeded maximum number of transforms {}, stop early.'.format(self.max_num_transforms))
        return graph

    # @utils.line_profile()
    def try_transform(self, graph: FlowGraph) -> Tuple[bool, FlowGraph]:
        patterns: List[GraphPattern] = all_patterns()
        usage: Dict[Tensor, List[Tuple[Optional[Operator], int]]] = analyze_usage(graph)
        all_tensors: List[Tensor] = graph_collect(graph, Tensor)

        for actual_tensor in all_tensors:
            for graph_pattern in patterns:
                source_pattern = graph_pattern.source()
                # condition 1
                matched = match(source_pattern, target=actual_tensor)
                if matched is None:
                    continue
                # condition 2
                rmap: Dict[Tensor, TensorPattern] = {v: k for k, v in matched.items() if isinstance(v, Tensor)}  # actual tensor -> pattern tensor
                inner_tensors: List[Tensor] = [t for t in rmap if (rmap[t].trace is not None            # not input tensor
                                                                   and rmap[t] is not source_pattern)]  # not output tensor
                matched_operators: Set[Operator] = {v for v in matched.values() if isinstance(v, Operator)}
                if any(any(use[0] not in matched_operators for use in usage[t]) for t in inner_tensors):
                    # tensor t is used by an operator not matched by the pattern graph, which violates condition 2.
                    continue
                # apply this transform
                target_tensor = graph_pattern.target(matched)
                if target_tensor is None:
                    continue
                for use in usage[actual_tensor]:
                    op, idx = use
                    assert isinstance(idx, int)
                    if op is None:
                        graph.outputs[idx] = target_tensor
                    else:
                        op.inputs[idx] = target_tensor
                return True, graph
        return False, graph


def pattern_transform_pass() -> GraphPass:
    return PatternTransformPass()
