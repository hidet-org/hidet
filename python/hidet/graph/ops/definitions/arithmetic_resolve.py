from typing import List, Optional, Union
from hidet.graph.ir import Tensor
from hidet.graph import ops
from hidet.graph.transforms import ResolveRule, register_resolve_rule

from .arithmetic import PowOp


@register_resolve_rule(PowOp)
class PowOpResolveRule(ResolveRule):
    def resolve(self, op: PowOp) -> Optional[List[Tensor]]:
        a, b = op.inputs
        assert isinstance(b, Tensor)
        if b.is_symbolic():
            return None
        if len(b.shape) != 0:
            return None
        b: Union[float, int] = b.scalar()
        if b in [2.0, 2]:
            return [ops.square(a)]
        elif b in [1.0, 1]:
            return [a]
        elif b in [-1.0, -1]:
            return [ops.reciprocal(a)]
        else:
            return None
