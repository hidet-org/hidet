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
from hidet.utils import prod
from hidet.cuda.nccl import str_to_nccl_op
from .utils import Task, TensorNode, Operator, Tensor, compute, input_like


class AllReduceTask(Task):
    def __init__(self, x: TensorNode, op: str, comm_id: int = 0):
        y = compute('out', x.shape, lambda *indices: x[indices])
        self.comm_id = comm_id
        self.op = op

        super().__init__('all_reduce', inputs=[x], outputs=[y], attributes={'comm_id': comm_id, 'op': op})

    def __str__(self):
        return f"all_reduce"

    def implement(self, target: Union[Target, str], working_dir: str) -> List[IRModule]:
        import hidet
        from hidet.ir.primitives.cuda.nccl import all_reduce as _all_reduce
        from hidet.lang import attrs

        dtype: DataType = self.inputs[0].type.dtype
        shape: Tuple[Expr, ...] = self.inputs[0].shape
        nbytes = dtype.nbytes * prod(shape)

        with hidet.script_module() as script_module:

            @hidet.script
            def launch(x: dtype[shape], y: dtype[shape]):
                attrs.func_kind = 'public'
                _all_reduce(x, y, nbytes, dtype, str_to_nccl_op(self.op), self.comm_id)

        return [script_module.ir_module()]


class AllReduceOp(Operator):
    def __init__(self, x: Tensor, op: str, comm_id: int):
        super().__init__(
            inputs=[x], attributes={'op': op, 'comm_id': comm_id}, task=AllReduceTask(input_like(x, 'x'), op, comm_id)
        )


def all_reduce(x: Tensor, op: str, comm_id: int = 0) -> Tensor:
    if x.device.kind != 'cuda':
        raise RuntimeError("NCCL only supports CUDA tensors")
    return AllReduceOp(x, op, comm_id).outputs[0]


def broadcast(x: Tensor, root: int, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()


def reduce(x: Tensor, root: int, op: str, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()


def all_gather(x: Tensor, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()


def reduce_scatter(x: Tensor, op: str, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()


def send(x: Tensor, peer: int, comm_id: int = 0) -> None:
    raise NotImplementedError()


# Recv is a little bit tricky since we need to pass the metadata of the recv buffer
def recv(peer: int, comm_id: int = 0) -> Tensor:
    raise NotImplementedError()
