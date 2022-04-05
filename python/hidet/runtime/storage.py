from typing import Callable, Dict, List, Type
from collections import defaultdict
import ctypes
import numpy as np
from hidet.ffi import cuda_api


class StorageDevice:
    def name(self):
        raise NotImplementedError()

    def allocate(self, nbytes) -> int:
        raise NotImplementedError()

    def free(self, addr):
        raise NotImplementedError()


class Storage:

    def __init__(self, device, addr, num_bytes, free_handler):
        self.device: str = device
        self.addr: int = addr
        self.num_bytes: int = num_bytes
        self.free_handler: Callable[[Storage], None] = free_handler

    def __del__(self):
        if self.addr != 0:
            self.free_handler(self)

    def cpu(self):
        if self.device == 'cpu':
            return self
        elif self.device == 'cuda':
            host_storage = self.new('cpu', self.num_bytes)
            cuda_api.memcpy_async(src_addr=self.addr, dst_addr=host_storage.addr, num_bytes=self.num_bytes, kind=cuda_api.DeviceToHost)
            return host_storage
        else:
            raise NotImplementedError()

    def cuda(self):
        if self.device == 'cuda':
            return self
        elif self.device == 'cpu':
            cuda_storage = self.new('cuda', self.num_bytes)
            cuda_api.memcpy_async(src_addr=self.addr, dst_addr=cuda_storage.addr, num_bytes=self.num_bytes, kind=cuda_api.HostToDevice)
            return cuda_storage
        else:
            raise NotImplementedError()

    @staticmethod
    def new(device: str, num_bytes: int) -> 'Storage':
        if device == 'cpu':
            return cpu_pool.allocate(nbytes=num_bytes)
        elif device == 'cuda':
            return cuda_pool.allocate(nbytes=num_bytes)
        else:
            raise ValueError("Unrecognized device '{}', candidates: {}".format(device, ['cpu', 'cuda']))

    def as_array(self, dtype: str = 'float32') -> np.ndarray:
        """
        Convert to one-dimension numpy array, sharing the underlying storage.

        Parameters
        ----------
        dtype: str, default 'float32'
            The type of data in this storage.

        Returns
        -------
        ret: numpy.ndarray
            A numpy ndarray with one dimension that share the same data as the storage.
        """
        if self.device != 'cpu':
            raise ValueError('The storage must be cpu storage. Please use .cpu() to convert first.')
        if dtype == 'float32':
            assert self.num_bytes % 4 == 0
            num_elements = self.num_bytes // 4
            buf = (ctypes.c_float * num_elements).from_address(self.addr)
            buf._hidet_storage = self  # so this storage will not be freed as long as the buffer not been freed.
            return np.ctypeslib.as_array(buf)
        else:
            raise NotImplementedError()


class MemoryPool:
    global_pool = None

    def __init__(self, storage_device: StorageDevice, block_size: int, max_reserve_size: int):
        self.storage_device = storage_device
        self.block_size: int = block_size
        self.max_reserve_size: int = max_reserve_size

        self.reserved_size: int = 0
        self.active_blocks = 0
        self.memory_blocks: Dict[int, List[Storage]] = defaultdict(list)

    def allocate(self, nbytes: int) -> Storage:
        allocated = (nbytes + self.block_size - 1) // self.block_size * self.block_size
        block_list = self.memory_blocks[allocated]
        if len(block_list) > 0:
            storage = block_list.pop()
            addr = storage.addr
            self.reserved_size -= storage.num_bytes
        else:
            addr = self.storage_device.allocate(allocated)
        return Storage(
            device=self.storage_device.name(),
            addr=addr,
            num_bytes=allocated,
            free_handler=self.free
        )

    def free(self, storage: Storage):
        self.memory_blocks[storage.num_bytes].append(storage)
        self.reserved_size += storage.num_bytes
        if self.reserved_size > self.max_reserve_size:
            print('clearing ----------------------------- reserved {}'.format(nbytes2str(self.reserved_size)))
            self.clear()

    def clear(self):
        for block_list in self.memory_blocks.values():
            for storage in block_list:
                self.storage_device.free(storage.addr)
                storage.addr = 0
        self.memory_blocks.clear()
        self.reserved_size = 0

    def __del__(self):
        self.clear()


class CudaStorageDevice(StorageDevice):
    def __init__(self):
        self.addr2nbytes = {}
        self.peak_active_memory = 0
        self.active_memory = 0

    def name(self):
        return 'cuda'

    def allocate(self, nbytes):
        addr = cuda_api.malloc_async(nbytes)
        self.active_memory += nbytes
        self.peak_active_memory = max(self.peak_active_memory, self.active_memory)
        # print('allocated {}, active {}, peak {}'.format(
        #     nbytes2str(nbytes),
        #     nbytes2str(self.active_memory),
        #     nbytes2str(self.peak_active_memory)
        # ))
        self.addr2nbytes[addr] = nbytes
        return addr

    def free(self, addr):
        assert addr in self.addr2nbytes
        self.active_memory -= self.addr2nbytes[addr]
        # print('free {}, active {}, peak {}'.format(
        #     nbytes2str(self.addr2nbytes[addr]),
        #     nbytes2str(self.active_memory),
        #     nbytes2str(self.peak_active_memory)
        # ))
        del self.addr2nbytes[addr]
        cuda_api.free_async(addr)


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
    return '{} {}'.format(size, unit)


class CpuStorageDevice(StorageDevice):
    def name(self):
        return 'cpu'

    def allocate(self, nbytes):
        return cuda_api.malloc_host(nbytes)

    def free(self, addr):
        cuda_api.free_host(addr)


cpu_pool = MemoryPool(
    storage_device=CpuStorageDevice(),
    block_size=1,
    max_reserve_size=0  # do not reserve
)

cuda_pool = MemoryPool(
    storage_device=CudaStorageDevice(),
    block_size=4 * 1024,  # 4 KiB
    max_reserve_size=3 * 1024 * 1024 * 1024  # 3 GiB
)
