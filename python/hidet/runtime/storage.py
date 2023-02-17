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
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Union
from collections import defaultdict
import hidet.cuda
from hidet.cuda.stream import Stream
from hidet.utils import green, initialize, exiting
from hidet.runtime.device import Device, instantiate_device


def nbytes2str(nbytes: int) -> str:
    if nbytes > 1024 * 1024:
        size = nbytes // 1024 // 1024
        unit = 'MiB'
    elif nbytes > 1024:
        size = nbytes // 1024
        unit = 'KiB'
    else:
        size = nbytes
        unit = 'Bytes'
    return green('{} {}'.format(size, unit))


class MemoryAPI:
    def __init__(self, device: Device):
        self.device: Device = device
        self.addr2nbytes: Dict[int, int] = {}
        self.peak_allocated: int = 0
        self.allocated: int = 0

    def malloc(self, nbytes: int) -> int:
        raise NotImplementedError

    def free(self, addr: int):
        raise NotImplementedError

    def memory_info(self) -> (int, int):
        raise NotImplementedError


class CudaMemoryAPI(MemoryAPI):
    def malloc(self, nbytes: int) -> int:
        with hidet.cuda.device(self.device.id):
            addr = hidet.cuda.malloc_async(nbytes)
        if addr == 0 and nbytes != 0:
            # out of memory
            return 0
        else:
            self.allocated += nbytes
            self.peak_allocated = max(self.peak_allocated, self.allocated)
            self.addr2nbytes[addr] = nbytes
            return addr

    def free(self, addr: int):
        with hidet.cuda.device(self.device.id):
            hidet.cuda.free_async(addr)
        self.allocated -= self.addr2nbytes.pop(addr)

    def memory_info(self) -> (int, int):
        return hidet.cuda.memory_info()


class CpuMemoryAPI(MemoryAPI):
    def malloc(self, nbytes: int) -> int:
        addr = hidet.cuda.malloc_host(nbytes)
        if addr == 0 and nbytes != 0:
            return 0
        self.allocated += nbytes
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.addr2nbytes[addr] = nbytes
        return addr

    def free(self, addr: int):
        hidet.cuda.free_host(addr)
        self.allocated -= self.addr2nbytes.pop(addr)

    def memory_info(self) -> (int, int):
        raise NotImplementedError()


class Storage:
    def __init__(self, device: Device, addr: int, num_bytes: int, free_handler: Callable[[Storage], None]):
        self.device: Device = device
        self.addr: int = addr
        self.num_bytes: int = num_bytes
        self.free_handler: Callable[[Storage], None] = free_handler

    def __del__(self):
        if self.addr != 0:
            self.free_handler(self)

    def __getstate__(self):
        raise ValueError()

    def __setstate__(self, state):
        raise ValueError()

    @staticmethod
    def new(device: Union[Device, str], num_bytes: int) -> Storage:
        """
        Allocate a new storage on the given device.

        Parameters
        ----------
        device: Device or str
            The device to allocate the storage on.

        num_bytes
            The number of bytes to allocate.

        Returns
        -------
        ret: Storage
            The allocated storage.
        """
        if isinstance(device, str):
            device = hidet.runtime.device.device(device)
        else:
            if not isinstance(device, Device):
                raise TypeError('device must be Device or str, but got {}'.format(type(device)))
        if device.is_cuda() and device.id is None:
            device = hidet.runtime.device.instantiate_device(device)
        return current_memory_pool(device).malloc(num_bytes)

    @staticmethod
    def _convert(
        src: Storage, dst_device: Device, non_blocking: bool, stream: Optional[Stream] = None, copy: bool = False
    ) -> Storage:
        if src.device == dst_device and not copy:
            return src

        dst: Storage = Storage.new(dst_device, src.num_bytes)
        if src.device.is_cuda() and dst.device.is_cuda() and src.device.id != dst_device.id:
            # peer to peer copy among cuda devices
            if non_blocking:
                hidet.cuda.memcpy_peer_async(dst.addr, dst_device.id, src.addr, src.device.id, src.num_bytes, stream)
            else:
                hidet.cuda.memcpy_peer(dst.addr, dst_device.id, src.addr, src.device.id, src.num_bytes)
        else:
            device = src.device if src.device.is_cuda() else dst_device
            with device:
                if non_blocking:
                    hidet.cuda.memcpy_async(dst.addr, src.addr, src.num_bytes, stream)
                else:
                    hidet.cuda.memcpy(dst.addr, src.addr, src.num_bytes)
        return dst

    def cpu(self) -> Storage:
        """
        Copy the storage to CPU. If the storage is already on CPU, return itself.

        Returns
        -------
        ret: Storage
            The storage on CPU.
        """
        return Storage._convert(self, Device('cpu'), non_blocking=False)

    def cpu_async(self, stream: Optional[Stream] = None):
        """
        Copy the storage to CPU asynchronously. If the storage is already on CPU, return itself.

        Parameters
        ----------
        stream: Stream, optional
            The stream to copy the storage. If None, use the current stream of the storage's device.

        Returns
        -------
        ret: Storage
            The storage on CPU.
        """
        return Storage._convert(self, Device('cpu'), non_blocking=True, stream=stream)

    def cuda(self, dst_id: int) -> Storage:
        """
        Copy the storage to CUDA device. If the storage is already on the device, return itself.

        Parameters
        ----------
        dst_id: int
            The id of the destination CUDA device.

        Returns
        -------
        ret: Storage
            The storage on the destination CUDA device.
        """
        return Storage._convert(self, Device('cuda', dst_id), non_blocking=False)

    def cuda_async(self, dst_id: int, stream: Optional[Stream] = None):
        """
        Copy the storage to CUDA device asynchronously. If the storage is already on the device, return itself.

        Parameters
        ----------
        dst_id: int
            The id of the destination CUDA device.

        stream: Stream, optional
            The stream to copy the storage. If None, use the current stream of the storage's device.

        Returns
        -------
        ret: Storage
            The storage on the destination CUDA device.
        """
        return Storage._convert(self, Device('cuda', dst_id), non_blocking=True, stream=stream)

    def copy(self) -> Storage:
        """
        Copy the storage to the same device. If the storage is already on the device, return itself.

        Returns
        -------
        ret: Storage
            The storage on the same device.
        """
        return Storage._convert(self, self.device, non_blocking=False, copy=True)

    def copy_async(self, stream: Optional[Stream] = None) -> Storage:
        """
        Copy the storage to the same device asynchronously. If the storage is already on the device, return itself.

        Parameters
        ----------
        stream: Stream, optional
            The stream to copy the storage. If None, use the current stream of the storage's device.

        Returns
        -------
        ret: Storage
            The storage on the same device.
        """
        return Storage._convert(self, self.device, non_blocking=True, stream=stream, copy=True)


class MemoryPool:
    def __init__(self, memory_api: MemoryAPI, block_size: int, max_reserve_size: int):
        self.memory_api: MemoryAPI = memory_api
        self.block_size: int = block_size
        self.max_reserve_size: int = max_reserve_size

        self.reserved_size: int = 0
        self.active_blocks = 0
        self.memory_blocks: Dict[int, List[Storage]] = defaultdict(list)

    def malloc(self, nbytes: int) -> Storage:
        allocated = (nbytes + self.block_size - 1) // self.block_size * self.block_size
        block_list = self.memory_blocks[allocated]
        if len(block_list) > 0:
            storage = block_list.pop()
            addr = storage.addr
            self.reserved_size -= storage.num_bytes
        else:
            addr = self.memory_api.malloc(allocated)
            if addr == 0 and allocated != 0:
                # out of memory
                self.clear()
                addr = self.memory_api.malloc(allocated)
                if addr == 0:
                    free, total = self.memory_api.memory_info()
                    raise MemoryError(
                        f'Can not allocate memory from {self.memory_api.device} device, '
                        f'total {nbytes2str(total)}, '
                        f'hidet allocated {nbytes2str(self.memory_api.allocated)}, '
                        f'free {nbytes2str(free)}, '
                        f'requesting {nbytes2str(allocated)}.'
                    )
        return Storage(device=self.memory_api.device, addr=addr, num_bytes=allocated, free_handler=self.free)

    def free(self, storage: Storage):
        self.memory_blocks[storage.num_bytes].append(storage)
        self.reserved_size += storage.num_bytes
        if self.reserved_size > self.max_reserve_size:
            self.clear()

    def clear(self):
        hidet.cuda.synchronize()
        for block_list in self.memory_blocks.values():
            for storage in block_list:
                self.memory_api.free(storage.addr)
                storage.addr = 0
        self.memory_blocks.clear()
        self.reserved_size = 0

    def status(self) -> str:
        allocated = self.memory_api.allocated
        peak_allocated = self.memory_api.peak_allocated
        items = [
            ['Allocated', allocated],
            ['Peak', peak_allocated],
            ['Reserved', self.reserved_size],
            ['Active', allocated - self.reserved_size],
        ]
        lines = [
            'Status of {} memory pool'.format(self.memory_api.device),
            *['{:>12}: {}'.format(name, nbytes2str(nbytes)) for name, nbytes in items],
        ]
        return '\n'.join(lines)

    def __str__(self):
        return self.status()

    def __del__(self, is_shutting_down=exiting.is_exiting):
        if is_shutting_down():
            return
        self.clear()


class MemoryPoolContext:
    def __init__(self, pool: MemoryPool):
        self.device: Device = pool.memory_api.device
        self.memory_pool: MemoryPool = pool
        self.prev_memory_pool: Optional[MemoryPool] = None

    def __enter__(self):
        self.prev_memory_pool = _device2pool[self.device]
        _device2pool[self.device] = self.memory_pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        _device2pool[self.device] = self.prev_memory_pool


_device2pool: Dict[Device, MemoryPool] = {}


@initialize()
def initialize_memory_pools():
    global _device2pool
    _device2pool = {
        Device('cpu'): MemoryPool(CpuMemoryAPI(Device('cpu')), block_size=4096, max_reserve_size=512 * 1024**2)
    }
    for device_id in range(hidet.cuda.device_count()):
        device = Device('cuda', device_id)
        _device2pool[device] = MemoryPool(CudaMemoryAPI(device), block_size=4096, max_reserve_size=4 * 1024**3)


def current_memory_pool(device: Union[Device, str]) -> MemoryPool:
    """
    Get current memory pool for the given device.

    All memory allocations on given device will be performed from the returned memory pool. You can change the current
    memory pool by using :func:`memory_pool` context manager.

    Parameters
    ----------
    device: Device or str
        Device for which to get the current memory pool.

    Returns
    -------
    ret: MemoryPool
        Current memory pool for the given device.
    """
    device = instantiate_device(device)
    if device not in _device2pool:
        raise ValueError('No memory pool for device {}'.format(device))
    return _device2pool[device]


def memory_pool(pool: MemoryPool):
    return MemoryPoolContext(pool)
