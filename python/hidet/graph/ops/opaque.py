"""
Opaque operator is an operator that does not provide the computation definition, but use an unique name to
identify the computation. Opaque operator is used to represent the operators that are hard to represent its
computation definition, or it is too tedious to represent its computation definition.
"""
from typing import List, Dict, Any, Optional, Union, Sequence
from hidet.graph.tensor import symbol
from .utils import Tensor, Task, Operator, IRModule, Expr, input_like


class OpaqueTask(Task):
    def __init__(self, name: str, inputs, outputs, op, share_map: Optional[Dict[int, int]] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs, attributes={'is_opaque': True}, share_map=share_map)
        self.op: OpaqueOperator = op

    def allow_prologue(self) -> bool:
        return self.op.allow_prologue()

    def allow_epilogue(self) -> bool:
        return self.op.allow_prologue()

    def implement_cuda(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return self.op.implement_cuda(self.op.inputs, self.op.outputs)

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return self.op.implement_cpu(self.op.inputs, self.op.outputs)


class OpaqueOperator(Operator):
    def __init__(
        self,
        name: str,
        inputs: Dict[str, Tensor],
        attributes: Optional[Dict[str, Any]] = None,
        share_map: Optional[Dict[int, int]] = None,
    ):
        symbol_outputs: Dict[str, Tensor] = self.symbolic_forward(**inputs)
        super().__init__(
            inputs=list(inputs.values()),
            attributes=attributes if attributes is not None else {},
            task=OpaqueTask(
                name=name,
                inputs=[input_like(tensor, name) for name, tensor in inputs.items()],
                outputs=[input_like(tensor, name) for name, tensor in symbol_outputs.items()],
                op=self,
                share_map=share_map,
            ),
        )

    def symbol(self, shape: Sequence[Union[int, str, Expr]], dtype='float32', device='cpu'):
        return symbol(shape, dtype, device)

    def allow_prologue(self):
        """
        Whether to allow prologue for this operator for prologue_epilogue_fusion pass.

        Returns
        -------
        ret: bool
            True if allow prologue, False otherwise.
        """
        return False

    def allow_epilogue(self):
        """
        Whether to allow epilogue for this operator for prologue_epilogue_fusion pass.


        Returns
        -------
        ret: bool
            True if allow epilogue, False otherwise.
        """
        return False

    def symbolic_forward(self, **args: Tensor) -> Dict[str, Tensor]:
        """
        Infer the dtype and shape of the output tensors given the input tensors.

        Parameters
        ----------
        args: Dict[str, Tensor]
            The input tensors.

        Returns
        -------
        ret: Dict[str, Tensor]
            The output tensors.
        """
        raise NotImplementedError()

    def implement_cuda(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        """
        Implement this operator on CUDA.

        Parameters
        ----------
        inputs: List[Tensor]
            The input tensors.
        outputs: List[Tensor]
            The output tensors.

        Returns
        -------
        ret: Union[IRModule, List[IRModule]]
            The IRModule or a list of IRModules that implement this operator. When multiple IRModules are returned,
            they must have the same functionality and hidet will pick the most performant one to use.
        """
        raise NotImplementedError('Opaque operator {} does not have CUDA implementation'.format(self.name))

    def implement_cpu(self, inputs: List[Tensor], outputs: List[Tensor]) -> Union[IRModule, List[IRModule]]:
        """
        Implement this operator on CPU.

        Parameters
        ----------
        inputs: List[Tensor]
            The input tensors.

        outputs: List[Tensor]
            The output tensors.

        Returns
        -------
        ret: Union[IRModule, List[IRModule]]
            The IRModule or a list of IRModules that implement this operator. When multiple IRModules are returned,
            they must have the same functionality and hidet will pick the most performant one to use.
        """
        raise NotImplementedError('Opaque operator {} does not have CPU implementation'.format(self.name))
