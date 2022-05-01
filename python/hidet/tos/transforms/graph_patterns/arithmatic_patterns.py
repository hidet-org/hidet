from typing import List, Optional, Dict, Union

from hidet.tos.ir.graph import Operator, Tensor
from .base import GraphPattern, TensorPattern, OperatorPattern, MatchDict


# class GraphConstructor:
#     """
#     Construct the new subgraph according the matched subgraph and target graph.
#     """
#     def __init__(self, matched):
#         self.memo = {}
#         self.matched = matched
#         self.new_operators = []
#
#     def visit(self, obj: Union[TensorPattern, OperatorPattern]):
#         if obj in self.memo:
#             return self.memo[obj]
#         if isinstance(obj, OperatorPattern):
#             ret = self.visit_OperatorPattern(obj)
#         elif isinstance(obj, TensorPattern):
#             ret = self.visit_TensorPattern(obj)
#         else:
#             raise ValueError()
#         self.memo[obj] = ret
#         return ret
#
#     def visit_TensorPattern(self, t: TensorPattern) -> Tensor:
#         if t.trace is None:
#             # input in pattern
#             return self.matched[t]
#         else:
#             op, idx = t.trace
#             return self.visit(op).get_output(idx)
#
#     def visit_OperatorPattern(self, t: OperatorPattern) -> Operator:
#         inputs = [self.visit(x) for x in t.inputs]
#         op = t.op_cls(*inputs)
#         self.new_operators.append(op)
#         return op


class ArithmaticGraphPattern(GraphPattern):
    def __init__(self, name, fsrc, fdst):
        super().__init__(name)
        x, y = TensorPattern.tensors(2, is_symbolic=True)  # can not be const
        a, b = TensorPattern.tensors(2, is_const=True)  # can not be symbolic
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.src = fsrc(x, y, a, b)
        self.fdst = fdst

    def source(self) -> List[TensorPattern]:
        return [self.src]

    def target(self, matched: MatchDict) -> Optional[List[TensorPattern]]:
        x, y, a, b = [matched[v] if v in matched else None for v in [self.x, self.y, self.a, self.b]]
        return [self.fdst(x, y, a, b)]
        # constructor = GraphConstructor(matched)
        # return [constructor.visit(self.tgt)]


def arithmatic_patterns() -> List[GraphPattern]:
    # # tensors can be used as pattern inputs
    # x, y, z = TensorPattern.tensors(3, is_symbolic=True)  # can not be const
    # a, b, c = TensorPattern.tensors(3, is_const=True)  # can not be symbolic
    #
    # (source, target) pattern pairs
    pairs = [
        ['a + x => x + a', lambda x, y, a, b: a + x, lambda x, y, a, b: x + a],
        ['x - a => x + (-a)', lambda x, y, a, b: x - a, lambda x, y, a, b: x + (-a)],
        ['(x + a) + b => x + (a + b)', lambda x, y, a, b: (x + a) + b, lambda x, y, a, b: x + (a + b)],
        ['(x + a) * b => x * b + a * b', lambda x, y, a, b: (x + a) * b, lambda x, y, a, b: x * b + a * b],
        ['(x + a) + (y + b) => (x + y) + (a + b)', lambda x, y, a, b: (x + a) + (y + b), lambda x, y, a, b: (x + y) + (a + b)],
    ]
    return [ArithmaticGraphPattern(name, src, tgt) for name, src, tgt in pairs]
