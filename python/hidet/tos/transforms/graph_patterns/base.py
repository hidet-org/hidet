from typing import List, Optional, Dict, Any, Union, Tuple, Type, Set
from hidet.tos.ir.graph import FlowGraph, Operator, Tensor
from hidet.tos.transforms import GraphPass, PassContext
from hidet.tos import ops
from hidet import tos


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
        return OperatorPattern(ops.definitions.arithmatic.AddOp, inputs=[self, other]).outputs[0]

    def __sub__(self, other):
        return OperatorPattern(ops.definitions.arithmatic.SubOp, inputs=[self, other]).outputs[0]

    def __mul__(self, other):
        return OperatorPattern(ops.definitions.arithmatic.MultiplyOp, inputs=[self, other]).outputs[0]

    def __neg__(self):
        return OperatorPattern(ops.definitions.arithmatic.NegOp, inputs=[self]).outputs[0]

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
            ops.definitions.arithmatic.NegOp: '-'
        }
        binary_ops = {
            ops.definitions.arithmatic.AddOp: '+',
            ops.definitions.arithmatic.SubOp: '-',
            ops.definitions.arithmatic.MultiplyOp: '*'
        }
        if self.op_cls in unary_ops:
            return '({}{})'.format(unary_ops[self.op_cls], input_items[0])
        elif self.op_cls in binary_ops:
            return '({} {} {})'.format(input_items[0], binary_ops[self.op_cls], input_items[1])
        else:
            return '{}({})'.format(self.op_cls.__class__[:-2], ', '.join(input_items))


class GraphPattern:
    def __init__(self, name):
        self.name = name

    def source(self) -> TensorPattern:
        """
        The pattern to match in the computation graph.
        """
        raise NotImplementedError()

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        """
        The target sub-graph used to replace the matched pattern.
        Return None means failed to generate the target sub-graph, and we should not do the transformation.
        """
        raise NotImplementedError()


def op_pattern(op_cls: Type[Operator], input_patterns: List[TensorPattern], num_outputs=1) -> Union[TensorPattern, List[TensorPattern]]:
    op = OperatorPattern(op_cls, input_patterns, num_outputs)
    if num_outputs == 1:
        return op.outputs[0]
    else:
        return op.outputs

