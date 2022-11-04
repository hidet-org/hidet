from typing import List, Optional
from .base import GraphPattern, TensorPattern, MatchDict


class arithmeticGraphPattern(GraphPattern):
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


def arithmetic_patterns() -> List[GraphPattern]:
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
        [
            '(x + a) + (y + b) => (x + y) + (a + b)',
            lambda x, y, a, b: (x + a) + (y + b),
            lambda x, y, a, b: (x + y) + (a + b),
        ],
    ]
    return [arithmeticGraphPattern(name, src, tgt) for name, src, tgt in pairs]
