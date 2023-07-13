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
from hidet.cuda.nccl import (
    create_unique_id,
    NcclUniqueId,
    create_comm,
    NcclCommunicator,
    comms_to_array,
    group_start,
    group_end,
)
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

    def scatter(self, tensor: Tensor, scatter_list: Optional[List[Tensor]] = None, src: int = 0):
        raise NotImplementedError()

    def reduce_scatter(self, output: Tensor, input_list: List[Tensor], op: str):
        raise NotImplementedError()

    def reduce_scatter_tensor(self, output: Tensor, input: Tensor, op: str):
        raise NotImplementedError()

    def barrier(self):
        raise NotImplementedError()

    def send(self, tensor: Tensor, dst: int):
        raise NotImplementedError()

    def recv(self, tensor: Tensor, src: int):
        raise NotImplementedError()


NCCL_COMMS: List[NcclCommunicator] = []
_NCCL_ARRAY: 'Array' = None


class NCCLProcessGroup(ProcessGroup):
    def __init__(self, comm: NcclCommunicator, world_size: int, rank: int):
        self._comm: NcclCommunicator = comm
        self._world_size: int = world_size
        self._rank: int = rank
        NCCL_COMMS.append(comm)

    @staticmethod
    def _check_cuda_tensor(tensor: Tensor):
        assert not tensor.is_symbolic() and tensor.device.is_cuda()

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world_size

    def all_reduce(self, tensor: Tensor, op: str):
        self._check_cuda_tensor(tensor)
        addr = tensor.storage.addr
        self._comm.all_reduce(addr, addr, tensor.size, tensor.dtype, op)

    def broadcast(self, tensor: Tensor, src: int):
        self._check_cuda_tensor(tensor)
        addr = tensor.storage.addr
        self._comm.broadcast(addr, addr, tensor.size, tensor.dtype, src)

    def reduce(self, tensor: Tensor, dst: int, op: str):
        self._check_cuda_tensor(tensor)
        addr = tensor.storage.addr
        self._comm.reduce(addr, addr, tensor.size, tensor.dtype, op, dst)

    def all_gather(self, tensor_list: List[Tensor], tensor: Tensor):
        assert len(tensor_list) == self._world_size
        map(self._check_cuda_tensor, tensor_list)
        assert self._check_cuda_tensor(tensor)

        group_start()
        for i, recv_tensor in enumerate(tensor_list):
            self.send(tensor, i)
            self.recv(recv_tensor, i)
        group_end()

    def all_gather_into_tensor(self, output_tensor: Tensor, input_tensor: Tensor):
        self._check_cuda_tensor(input_tensor)
        self._check_cuda_tensor(output_tensor)

        assert output_tensor.size == input_tensor.size * self._world_size

        output_addr = output_tensor.storage.addr
        input_addr = input_tensor.storage.addr
        self._comm.all_gather(input_addr, output_addr, input_tensor.size, input_tensor.dtype)

    def gather(self, tensor: Tensor, gather_list: Optional[List[Tensor]] = None, dst: int = 0):
        if dst == self._rank:
            assert gather_list is not None
            assert len(gather_list) == self._world_size
            map(self._check_cuda_tensor, gather_list)
            group_start()
            for i, recv_tensor in enumerate(gather_list):
                if i != self._rank:
                    self.recv(recv_tensor, i)
            group_end()
            gather_list[self._rank].copy_(tensor)
        else:
            self._check_cuda_tensor(tensor)
            self.send(tensor, dst)

    def scatter(self, tensor: Tensor, scatter_list: Optional[List[Tensor]] = None, src: int = 0):
        if src == self._rank:
            assert scatter_list is not None
            assert len(scatter_list) == self._world_size
            map(self._check_cuda_tensor, scatter_list)
            group_start()
            for i, send_tensor in enumerate(scatter_list):
                if i != self._rank:
                    self.send(send_tensor, i)
            group_end()
            tensor.copy_(scatter_list[self._rank])
        else:
            self._check_cuda_tensor(tensor)
            self.recv(tensor, src)

    def reduce_scatter(self, output: Tensor, input_list: List[Tensor], op: str):
        self._check_cuda_tensor(output)
        map(self._check_cuda_tensor, input_list)

        assert len(input_list) == self._world_size
        group_start()
        for i, tensor in enumerate(input_list):
            out_addr = output.storage.addr if i == self._rank else 0
            in_addr = tensor.storage.addr
            self._comm.reduce(in_addr, out_addr, tensor.size, tensor.dtype, op, i)
        group_end()

    def reduce_scatter_tensor(self, output: Tensor, input: Tensor, op: str):
        self._check_cuda_tensor(input)
        self._check_cuda_tensor(output)

        assert output.size * self._world_size == input.size
        output_addr = output.storage.addr
        input_addr = input.storage.addr
        self._comm.reduce_scatter(input_addr, output_addr, output.size, output.dtype, op)

    def barrier(self):
        dummy_tensor = hidet.empty([], device='cuda')
        self.all_reduce(dummy_tensor, 'sum')
        hidet.cuda.synchronize()

    def send(self, tensor: Tensor, dst: int):
        self._check_cuda_tensor(tensor)

        addr = tensor.storage.addr
        self._comm.send(addr, tensor.size, tensor.dtype, dst)

    def recv(self, tensor: Tensor, src: int):
        self._check_cuda_tensor(tensor)

        addr = tensor.storage.addr
        self._comm.recv(addr, tensor.size, tensor.dtype, src)


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
