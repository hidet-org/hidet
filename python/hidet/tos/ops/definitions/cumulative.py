from .utils import Task, Operator, Tensor, TensorNode, compute, reduce, input_like, normalize_dim


class CumulativeTask(Task):
    def __init__(self, x: TensorNode, dim: int, reduce_type: str, exclusive: bool, reverse: bool):
        x_shape = x.const_shape()
        y_shape = x_shape

        if not reverse:
            y = compute(
                name='cum_{}'.format(reduce_type),
                shape=y_shape,
                fcompute=lambda *indices: reduce(
                    shape=[indices[dim] + (0 if exclusive else 1)],
                    fcompute=lambda k: x[indices[:dim] + (k,) + indices[dim+1:]],
                    reduce_type=reduce_type,
                    accumulate_dtype=x.data_type.scalar_type.name
                )
            )
        else:
            y = compute(
                name='cum_{}'.format(reduce_type),
                shape=y_shape,
                fcompute=lambda *indices: reduce(
                    shape=[y_shape[dim] - indices[dim] - (1 if exclusive else 0)],
                    fcompute=lambda k: x[indices[:dim] + (indices[dim] + k + (1 if exclusive else 0),) + indices[dim+1:]],
                    reduce_type=reduce_type,
                    accumulate_dtype=x.data_type.scalar_type.name
                )
            )

        super().__init__(
            name='cum_{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={
                'dim': dim,
                'reduce_type': reduce_type
            }
        )


class CumulativeBaseOp(Operator):
    def __init__(self, x: Tensor, dim: int, exclusive: bool, reverse: bool, reduce_type: str):
        if reduce_type not in ['sum']:
            raise NotImplementedError('Current do not support cumulative operator for {} reduction.'.format(reduce_type))
        dim = normalize_dim(dim, rank=len(x.shape))
        super().__init__(
            inputs=[x],
            task=CumulativeTask(input_like(x, 'x'), dim, reduce_type, exclusive, reverse),
            attributes={
                'dim': dim,
                'exclusive': exclusive,
                'reverse': reverse
            }
        )


class CumulativeSumOp(CumulativeBaseOp):
    def __init__(self, x: Tensor, dim: int, exclusive: bool, reverse: bool):
        super().__init__(x, dim, exclusive, reverse, 'sum')


def cumsum(x: Tensor, dim: int, exclusive: bool = False, reverse: bool = False) -> Tensor:
    return CumulativeSumOp(x, dim, exclusive, reverse).get_output(0)
