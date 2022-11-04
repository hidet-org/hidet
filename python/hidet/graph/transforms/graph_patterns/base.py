from __future__ import annotations
from typing import List, Optional, Dict, Union, Tuple, Type
from hidet.graph.ir.flow_graph import Operator, Tensor
from hidet.graph import ops


class TensorPattern:
    def __init__(self, is_const=False, is_symbolic=False, trace=None):
        self.is_const: bool = is_const
        self.is_symbolic: bool = is_symbolic
        assert not (is_const and is_symbolic), 'Can not be const and symbolic at the same time'
        self.trace: Optional[Tuple[OperatorPattern, int]] = trace
        self.uses: List[Tuple[OperatorPattern, int]] = []

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
        return OperatorPattern(ops.definitions.arithmetic.AddOp, inputs=[self, other]).outputs[0]

    def __sub__(self, other):
        return OperatorPattern(ops.definitions.arithmetic.SubOp, inputs=[self, other]).outputs[0]

    def __mul__(self, other):
        return OperatorPattern(ops.definitions.arithmetic.MultiplyOp, inputs=[self, other]).outputs[0]

    def __neg__(self):
        return OperatorPattern(ops.definitions.arithmetic.NegOp, inputs=[self]).outputs[0]

    def op(self) -> Optional[OperatorPattern]:
        if self.trace is None:
            return None
        else:
            return self.trace[0]

    def add_use(self, op: OperatorPattern, idx: int):
        self.uses.append((op, idx))

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

        for idx, input_tensor in enumerate(self.inputs):
            input_tensor.add_use(self, idx)

    def __repr__(self):
        input_items = [str(v) for v in self.inputs]
        unary_ops = {ops.definitions.arithmetic.NegOp: '-'}
        binary_ops = {
            ops.definitions.arithmetic.AddOp: '+',
            ops.definitions.arithmetic.SubOp: '-',
            ops.definitions.arithmetic.MultiplyOp: '*',
        }
        if self.op_cls in unary_ops:
            return '({}{})'.format(unary_ops[self.op_cls], input_items[0])
        elif self.op_cls in binary_ops:
            return '({} {} {})'.format(input_items[0], binary_ops[self.op_cls], input_items[1])
        else:
            return '{}({})'.format(self.op_cls.__name__[:-2], ', '.join(input_items))


MatchDict = Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]


class GraphPattern:
    def __init__(self, name):
        self.name = name

    def source(self) -> List[TensorPattern]:
        """
        The output tensors in the source template graph to match in the computation graph.
        """
        raise NotImplementedError()

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        """
        The output tensors in the target sub-graph used to replace the matched pattern.
        Return None means failed to generate the target sub-graph, and we should not do the transformation.
        """
        raise NotImplementedError()


def op_pattern(
    op_cls: Type[Operator], input_patterns: List[TensorPattern], num_outputs=1
) -> Union[TensorPattern, List[TensorPattern]]:
    op = OperatorPattern(op_cls, input_patterns, num_outputs)
    if num_outputs == 1:
        return op.outputs[0]
    else:
        return op.outputs


Usage = Dict[Tensor, List[Tuple[Optional[Operator], int]]]


class NotMatchedException(Exception):
    pass


class PatternMatcher:
    """
    PatternMatcher matches a pattern to a subgraph in a larger graph.

    It starts from a tensor, or an operator, and tries to match the subgraph spanned from the start point.

    The spanning rules:
        1. A tensor spans to its producing operator and its consuming operators (i.e., uses).
        2. An operator spans to its input and output tensors.

    The matching rules:
        1. For tensor:
            a) check the storage requirement (e.g., constant and symbolic)
            b) check the output index in the producer's output array
        2. For operator:
            a) check the operator type.

    Because the operator also spans to its outputs, as long as the pattern is connected, we only need to start
    from a single tensor or operator.
    """

    def __init__(self, usage: Usage):
        self.matched = {}
        self.reverse_matched = {}
        self.usage: Usage = usage

    @staticmethod
    def check(cond: bool, msg=""):
        if not cond:
            raise NotMatchedException(msg)

    def match(self, pattern, target):
        key = pattern if not isinstance(pattern, list) else id(pattern)
        if key in self.matched:
            self.check(target is self.matched[key], 'tried to match a pattern to two different objects')
            # pattern has been matched to a different target
            return
        self.matched[key] = target
        self.reverse_matched[target] = key
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
            return
        if pattern.is_symbolic:
            self.check(target.storage is None, 'requires symbolic tensor')

        # spans to its inputs
        if pattern.trace:
            self.check(target.trace is not None)
            self.check(pattern.trace[1] == target.trace[1])
            self.match(pattern.trace[0], target.trace[0])

        # spans to its uses
        desire_uses: List[Tuple[OperatorPattern, int]] = pattern.uses
        actual_uses: List[Tuple[Optional[Operator], int]] = self.usage[target]
        for desire_use in desire_uses:
            desire_operator, desire_index = desire_use  # pylint: disable=unused-variable
            if desire_operator in self.matched:
                # this desire operator in pattern has been spanned
                continue
            spanned = False
            for actual_use in actual_uses:
                actual_operator, actual_index = actual_use  # pylint: disable=unused-variable
                if actual_operator in self.reverse_matched:
                    # this actual operator has been matched
                    continue
                if type(actual_operator) != desire_operator.op_cls:  # pylint: disable=unidiomatic-typecheck
                    continue
                self.match(desire_operator, actual_operator)
                spanned = True
                break
            self.check(spanned, "A usage of input tensor has not been spanned.")

    def match_OperatorPattern(self, pattern: OperatorPattern, target: Operator):
        self.check(isinstance(target, pattern.op_cls), "expect target with type 'Operator'")
        self.check(pattern.op_cls is target.__class__, 'operator cls does not match')
        assert len(pattern.inputs) == len(target.inputs) and len(pattern.outputs) == len(target.outputs)
        for a, b in zip(pattern.inputs, target.inputs):
            self.match(a, b)
        for a, b in zip(pattern.outputs, target.outputs):
            self.match(a, b)


def graph_pattern_match(pattern: TensorPattern, target: Tensor, usage: Usage) -> Optional[MatchDict]:
    # peek for early stop, only for performance
    if pattern.trace is None:
        if target.trace is not None:
            return None
        if (pattern.is_const and target.storage is None) or (pattern.is_symbolic and target.storage is not None):
            return None
        return {pattern: target}
    if pattern.trace and target.trace and pattern.trace[0].op_cls is not target.trace[0].__class__:
        return None

    # formal match
    matcher = PatternMatcher(usage)
    try:
        matcher.match(pattern, target)
        return matcher.matched
    except NotMatchedException:
        return None
