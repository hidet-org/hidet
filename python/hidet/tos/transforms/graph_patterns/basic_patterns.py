from typing import List, Optional, Dict, Union

from hidet.tos import ops
from hidet.tos.ir.graph import Operator, Tensor
from hidet.tos.ops.definitions.transform import SqueezeOp
from .base import GraphPattern, TensorPattern, OperatorPattern, op_pattern


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
    def __init__(self, name, source, target):
        super().__init__(name)
        self.src = source
        self.tgt = target

    @staticmethod
    def all() -> List[GraphPattern]:
        # tensors can be used as pattern inputs
        x, y, z = TensorPattern.tensors(3, is_symbolic=True)  # can not be const
        a, b, c = TensorPattern.tensors(3, is_const=True)  # can not be symbolic

        # (source, target) pattern pairs
        pairs = [
            ['a + x => x + a', a + x, x + a],
            ['x - a => x + (-a)', x - a, x + (-a)],
            ['(x + a) + b => x + (a + b)', (x + a) + b, x + (a + b)],
            ['(x + a) * b => x * b + a * b', (x + a) * b, x * b + a * b],
            ['(x + a) + (y + b) => (x + y) + (a + b)', (x + a) + (y + b), (x + y) + (a + b)],
        ]
        return [SimpleGraphPattern(name, src, tgt) for name, src, tgt in pairs]

    def source(self) -> TensorPattern:
        return self.src

    def target(self, matched: Dict) -> Optional[TensorPattern]:
        constructor = GraphConstructor(matched)
        return constructor.visit(self.tgt)


class SqueezeMultiplyPattern(GraphPattern):
    def __init__(self):
        super().__init__('squeeze(x) * c => squeeze(x * c)')
        self.x = TensorPattern.tensor()
        self.c = TensorPattern.tensor(is_const=True)
        self.s = op_pattern(SqueezeOp, [self.x])
        self.y = self.s * self.c

    def source(self) -> TensorPattern:
        return self.y

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        x, c, s, y = matched[self.x], matched[self.c], matched[self.s], matched[self.y]
        dims = s.op.attributes['dims']
        if len(c.shape) < len(y.shape):
            c = c.unsqueeze(list(range(len(y.shape) - len(c.shape))))
        c = c.unsqueeze(dims)   # now, c has the same shape as x
        return ops.squeeze(x * c, dims=dims)


def basic_patterns() -> List[GraphPattern]:
    return [
        *SimpleGraphPattern.all(),
        SqueezeMultiplyPattern()
    ]
