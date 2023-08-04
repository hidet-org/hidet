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
from typing import List, Union, Tuple

from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from hidet.ir.module import IRModule
from hidet.ir.task import Target
from hidet.ir.layout import RowMajorLayout
from hidet.utils import prod
from hidet.cuda.nccl import str_to_nccl_op
from .utils import Task, TensorNode, Operator, Tensor, compute, input_like


class AllReduceTask(Task):
    def __init__(self, x: TensorNode, op: str, comm_id: int = 0):
        if not isinstance(x.type.layout, RowMajorLayout):
            raise RuntimeError("Communication operations only support row major layout.")
        y = compute('out', x.shape, lambda *indices: x[indices])
        self.comm_id = comm_id
        self.op = op

        super().__init__('distributed.all_reduce', inputs=[x], outputs=[y], attributes={'comm_id': comm_id, 'op': op})

    def implement(self, target: Union[Target, str], working_dir: str) -> List[IRModule]:
        import hidet
        from hidet.ir.primitives.cuda.nccl import all_reduce as _all_reduce
        from hidet.lang import attrs

        dtype: DataType = self.inputs[0].type.dtype
        shape: Tuple[Expr, ...] = self.inputs[0].shape
        size = prod(shape)

        with hidet.script_module() as script_module:

            @hidet.script
            def launch(x: dtype[shape], y: dtype[shape]):
                attrs.func_kind = 'public'
                _all_reduce(x, y, size, dtype, str_to_nccl_op(self.op), self.comm_id)

        return [script_module.ir_module()]


class AllReduceOp(Operator):
    def __init__(self, x: Tensor, op: str, comm_id: int):
        super().__init__(
            inputs=[x], attributes={'op': op, 'comm_id': comm_id}, task=AllReduceTask(input_like(x, 'x'), op, comm_id)
        )


class AllGatherTask(Task):
    def __init__(self, x: TensorNode, nranks: int, comm_id: int = 0):
        if not isinstance(x.type.layout, RowMajorLayout):
            raise RuntimeError("Communication operations only support row major layout.")
        y = compute('out', (nranks,) + tuple(x.shape), lambda *indices: x[indices[1:]])

        self.nranks = nranks
        self.comm_id = comm_id

        super().__init__(
            'distributed.all_gather', inputs=[x], outputs=[y], attributes={'nranks': nranks, 'comm_id': comm_id}
        )

    def implement(self, target: Union[Target, str], working_dir: str) -> List[IRModule]:
        import hidet
        from hidet.ir.primitives.cuda.nccl import all_gather as _all_gather
        from hidet.lang import attrs

        dtype: DataType = self.inputs[0].type.dtype
        shape: Tuple[Expr, ...] = self.inputs[0].shape
        size = prod(shape)
        out_shape = (self.nranks,) + tuple(shape)

        with hidet.script_module() as script_module:

            @hidet.script
            def launch(x: dtype[shape], y: dtype[out_shape]):
                attrs.func_kind = 'public'
                _all_gather(x, y, size, dtype, self.comm_id)

        return [script_module.ir_module()]


class AllGatherOp(Operator):
    def __init__(self, x: Tensor, nranks: int, comm_id: int):
        super().__init__(
            inputs=[x],
            attributes={'nranks': nranks, 'comm_id': comm_id},
            task=AllGatherTask(input_like(x, 'x'), nranks, comm_id),
        )


class ReduceScatterTask(Task):
    def __init__(self, x: TensorNode, op: str, comm_id: int = 0):
        if not isinstance(x.type.layout, RowMajorLayout):
            raise RuntimeError("Communication operations only support row major layout.")
        if len(x.shape) < 1:
            raise ValueError(
                "The number of dimensions of the input tensor must be positive."
                "And the first dimension should be equal to the world size."
            )
        y = compute('out', x.shape[1:], lambda *indices: x[(0,) + indices])
        self.comm_id = comm_id
        self.op = op

        super().__init__(
            'distributed.reduce_scatter', inputs=[x], outputs=[y], attributes={'comm_id': comm_id, 'op': op}
        )

    def implement(self, target: Union[Target, str], working_dir: str) -> List[IRModule]:
        import hidet
        from hidet.ir.primitives.cuda.nccl import reduce_scatter as _reduce_scatter
        from hidet.lang import attrs

        dtype: DataType = self.inputs[0].type.dtype
        shape: Tuple[Expr, ...] = self.inputs[0].shape
        size = prod(shape[1:])

        with hidet.script_module() as script_module:

            @hidet.script
            def launch(x: dtype[shape], y: dtype[shape[1:]]):
                attrs.func_kind = 'public'
                _reduce_scatter(x, y, size, dtype, str_to_nccl_op(self.op), self.comm_id)

        return [script_module.ir_module()]


class ReduceScatterOp(Operator):
    def __init__(self, x: Tensor, op: str, comm_id: int):
        super().__init__(
            inputs=[x],
            attributes={'op': op, 'comm_id': comm_id},
            task=ReduceScatterTask(input_like(x, 'x'), op, comm_id),
        )


def all_reduce(x: Tensor, op: str, comm_id: int = 0) -> Tensor:
    # if x.device.kind != 'cuda':
    # raise RuntimeError("NCCL only supports CUDA tensors")
    return AllReduceOp(x, op, comm_id).outputs[0]


def all_gather(x: Tensor, nranks: int, comm_id: int = 0) -> Tensor:
    # if x.device.kind != 'cuda':
    # raise RuntimeError("NCCL only supports CUDA tensors")
    return AllGatherOp(x, nranks, comm_id).outputs[0]


def reduce_scatter(x: Tensor, op: str, comm_id: int = 0) -> Tensor:
    # # if x.device.kind != 'cuda':
    #     raise RuntimeError("NCCL only supports CUDA tensors")
    return ReduceScatterOp(x, op, comm_id).outputs[0]


# We haven't decided how to integrate asymmetric communication functions into computational graphs
# When a tensor is sent to other peers, it should be added to the output. Otherwise it won't be traced
# But which part should do this?
def send(x: Tensor, peer: int, comm_id: int = 0) -> None:
    raise NotImplementedError()


def recv(peer: int, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()


def broadcast(x: Tensor, root: int, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()


def reduce(x: Tensor, root: int, op: str, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()
