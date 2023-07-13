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

# pylint: disable=W0223

from typing import Optional, List

import hidet
from hidet.graph import Tensor
from hidet.cuda.nccl import create_unique_id, NcclUniqueId, create_comm, NcclCommunicator, comms_to_array
from .store import Store


class ProcessGroup:
    def backend(self) -> str:
        raise NotImplementedError()

    def rank(self) -> int:
        raise NotImplementedError()

    def size(self) -> int:
        raise NotImplementedError()

    def broadcast(self, tensor: Tensor, src: int):
        raise NotImplementedError()

    def all_reduce(self, tensor: Tensor, op: str):
        raise NotImplementedError()

    def reduce(self, tensor: Tensor, dst: int, op: str):
        raise NotImplementedError()

    def all_gather(self, tensor_list: List[Tensor], tensor: Tensor):
        raise NotImplementedError()

    def all_gather_into_tensor(self, output_tensor: Tensor, input_tensor: Tensor):
        raise NotImplementedError()

    def gather(self, tensor: Tensor, gather_list: Optional[List[Tensor]] = None, dst: int = 0):
        raise NotImplementedError()

    def scatter(self, tensor: Tensor, scattler_list: Optional[List[Tensor]] = None):
        raise NotImplementedError()

    def reduce_scatter(self, output: Tensor, input_list: List[Tensor], op: str):
        raise NotImplementedError()

    def reduce_scatter_tensor(self, output: Tensor, input: Tensor, op: str):
        raise NotImplementedError()

    def barrier(self):
        raise NotImplementedError()


NCCL_COMMS: List[NcclCommunicator] = []
_NCCL_ARRAY: 'Array' = None


class NCCLProcessGroup(ProcessGroup):
    def __init__(self, comm: NcclCommunicator, world_size: int, rank: int):
        self._comm: NcclCommunicator = comm
        self._world_size: int = world_size
        self._rank: int = rank
        NCCL_COMMS.append(comm)

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world_size

    def all_reduce(self, tensor: Tensor, op: str):
        assert not tensor.is_symbolic()
        assert tensor.device.is_cuda()
        addr = tensor.storage.addr
        self._comm.all_reduce(addr, addr, tensor.size, tensor.dtype, op)

    def broadcast(self, tensor: Tensor, src: int):
        assert not tensor.is_symbolic()
        assert tensor.device.is_cuda()
        addr = tensor.storage.addr
        self._comm.broadcast(addr, addr, tensor.size, tensor.dtype, src)

    def reduce(self, tensor: Tensor, dst: int, op: str):
        assert not tensor.is_symbolic()
        assert tensor.device.is_cuda()
        addr = tensor.storage.addr
        self._comm.reduce(addr, addr, tensor.size, tensor.dtype, op, dst)

    def all_gather_into_tensor(self, output_tensor: Tensor, input_tensor: Tensor):
        assert not output_tensor.is_symbolic()
        assert not input_tensor.is_symbolic()
        assert output_tensor.device.is_cuda()
        assert input_tensor.device.is_cuda()

        assert output_tensor.size == input_tensor.size * self._world_size

        output_addr = output_tensor.storage.addr
        input_addr = input_tensor.storage.addr
        self._comm.all_gather(input_addr, output_addr, input_tensor.size, input_tensor.dtype)

    def reduce_scatter_tensor(self, output: Tensor, input: Tensor, op: str):
        assert not output.is_symbolic()
        assert not input.is_symbolic()
        assert output.device.is_cuda()
        assert input.device.is_cuda()

        assert output.size * self._world_size == input.size
        output_addr = output.storage.addr
        input_addr = input.storage.addr
        self._comm.reduce_scatter(input_addr, output_addr, output.size, output.dtype, op)

    def barrier(self):
        dummy_tensor = hidet.empty([], device='cuda')
        self.all_reduce(dummy_tensor, 'sum')
        hidet.cuda.synchronize()


def create_nccl_group(store: Store, world_size: int, rank: int):
    if rank == 0:
        unique_id = create_unique_id()
        store.set('unique_id', unique_id.internal)
    else:
        _id = store.get('unique_id')
        unique_id = NcclUniqueId()
        unique_id.internal[:] = _id[:]
    comm = create_comm(world_size, unique_id, rank)
    return NCCLProcessGroup(comm, world_size, rank)


def set_nccl_comms():
    global _NCCL_ARRAY
    from hidet.ffi.runtime_api import runtime_api

    _NCCL_ARRAY = comms_to_array(NCCL_COMMS)
    runtime_api.set_nccl_comms(_NCCL_ARRAY)
