from hidet.ir.layout import DataLayout
from hidet.ir.type import TensorType, tensor_type
from hidet.ir.task import Task, Grid
from hidet.tos.operator import Operator, Tensor
from hidet.ir.dialects.compute import TensorInput, tensor_input, compute, reduce

from hidet.ir.functors import inline_compute


def input_like(tensor: Tensor, name: str) -> TensorInput:
    return TensorInput(name, dtype=tensor.dtype, shape=tensor.shape)

