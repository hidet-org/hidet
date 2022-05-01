from typing import Type, List, Optional
from hidet.tos.ir import FlowGraph, GraphRewriter, Tensor, Operator


class ResolveRule:
    def op_cls(self) -> Type[Operator]:
        raise NotImplementedError()

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        """
        Parameters
        ----------
        op: Operator
            The operator to be resolved.

        Returns
        -------
        ret: Optional[List[Tensor]]
            None - indicates the operator has not been resolved, keep the original operator.
            List[Tensor] - the output of resolved operators.
        """
        raise NotImplementedError()

