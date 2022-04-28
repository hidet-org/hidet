from typing import List, Optional, Dict, Tuple, Set

from functools import lru_cache
from hidet import tos
from hidet.tos.ir.graph import FlowGraph, Operator, Tensor
from hidet.tos.transforms import GraphPass, PassContext
from .common import analyze_usage, graph_collect
from .fold_const import fold_const_pass
from .graph_patterns import GraphPattern, TensorPattern, OperatorPattern
from .graph_patterns import basic_patterns, conv2d_patterns, matmul_patterns


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


@lru_cache()
def all_patterns() -> List[GraphPattern]:
    return basic_patterns() + conv2d_patterns() + matmul_patterns()


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
                if PassContext.current().verbose:
                    print('Applying transform: {}'.format(graph_pattern.name))
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
