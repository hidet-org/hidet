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
from hidet.runtime.device import Device, instantiate_device
from .utils import Task, TensorNode, Operator, Tensor, compute, input_like

from hidet.cuda.nccl import NcclRedOp

def all_reduce(comm_id: int, x: Tensor, op: NcclRedOp) -> Tensor:
    raise NotImplementedError()

def broadcast(comm_id: int, x: Tensor, root:int) -> Tensor:
    raise NotImplementedError()

def reduce(comm_id: int, x: Tensor, root:int) -> Tensor:
    raise NotImplementedError()

def all_gather(comm_id: int, x: Tensor) -> Tensor:
    raise NotImplementedError()

def reduce_scatter(comm_id: int, x: Tensor) -> Tensor:
    raise NotImplementedError()

def send(comm_id: int, x: Tensor, peer: int) -> None:
    raise NotImplementedError()

# Recv is a little bit tricky since we need to pass the metadata of the recv buffer
def recv(comm_id: int, peer: int) -> Tensor:
    raise NotImplementedError()