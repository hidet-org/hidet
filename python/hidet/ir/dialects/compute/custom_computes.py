from hidet.ir.type import TensorType, int32
from hidet.ir.dialects.compute.primitives import TensorNode
from .primitives import CustomCompute, TensorNode

"""
The custom compute primitives are used to implement all compute patterns that can not be represented as
compute and reduce primitives.
"""


class ArgReduceCompute(CustomCompute):
    def __init__(self, x: TensorNode, dim: int, reduce_type: str):
        self.x = x
        self.dim = dim
        self.reduce_type = reduce_type
        super().__init__(
            input_tensors=[x],
            input_scalars=[],
            data_type=TensorType(x.data_type.scope, int32, shape=x.data_type.shape)
        )


def arg_reduce(x: TensorNode, dim: int, reduce_type: str) -> TensorNode:
    if reduce_type not in ['max', 'min']:
        raise ValueError('Compute primitive arg_reduce only supports reduce type "min" and "max", but got "{}".'.format(reduce_type))
    tensor_compute = ArgReduceCompute(x, dim, reduce_type)
    return TensorNode(
        name='arg{}'.format(reduce_type),
        data_type=tensor_compute.data_type,
        tensor_compute=tensor_compute
    )
