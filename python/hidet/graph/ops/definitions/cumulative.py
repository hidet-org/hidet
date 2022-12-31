# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
                    fcompute=lambda k: x[indices[:dim] + (k,) + indices[dim + 1 :]],
                    reduce_type=reduce_type,
                    accumulate_dtype=x.ttype.dtype.name,
                ),
            )
        else:
            y = compute(
                name='cum_{}'.format(reduce_type),
                shape=y_shape,
                fcompute=lambda *indices: reduce(
                    shape=[y_shape[dim] - indices[dim] - (1 if exclusive else 0)],
                    fcompute=lambda k: x[
                        indices[:dim] + (indices[dim] + k + (1 if exclusive else 0),) + indices[dim + 1 :]
                    ],
                    reduce_type=reduce_type,
                    accumulate_dtype=x.ttype.dtype.name,
                ),
            )

        super().__init__(
            name='cum_{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={'dim': dim, 'reduce_type': reduce_type},
        )


class CumulativeBaseOp(Operator):
    def __init__(self, x: Tensor, dim: int, exclusive: bool, reverse: bool, reduce_type: str):
        if reduce_type not in ['sum']:
            raise NotImplementedError(f'Current do not support cumulative operator for {reduce_type} reduction.')
        dim = normalize_dim(dim, rank=len(x.shape))
        super().__init__(
            inputs=[x],
            task=CumulativeTask(input_like(x, 'x'), dim, reduce_type, exclusive, reverse),
            attributes={'dim': dim, 'exclusive': exclusive, 'reverse': reverse},
        )


class CumulativeSumOp(CumulativeBaseOp):
    def __init__(self, x: Tensor, dim: int, exclusive: bool, reverse: bool):
        super().__init__(x, dim, exclusive, reverse, 'sum')


def cumsum(x: Tensor, dim: int, exclusive: bool = False, reverse: bool = False) -> Tensor:
    return CumulativeSumOp(x, dim, exclusive, reverse).get_output(0)
