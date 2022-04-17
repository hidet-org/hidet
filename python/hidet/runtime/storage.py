from typing import Callable, Dict, List, Type
import warnings
from collections import defaultdict
import ctypes
import numpy as np
from hidet.ffi import cuda_api


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


class StorageDevice:
    def name(self):
        raise NotImplementedError()

    def allocate(self, nbytes) -> int:
        raise NotImplementedError()

    def free(self, addr):
        raise NotImplementedError()

    def allocated_memory(self) -> int:
        raise NotImplementedError()

    def peak_allocated_memory(self) -> int:
        raise NotImplementedError()

    def free_memory(self) -> int:
        raise NotImplementedError()

    def total_memory(self) -> int:
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

    def as_array(self, num_elements: int, dtype: str = 'float32') -> np.ndarray:
        """
        Convert to one-dimension numpy array, sharing the underlying storage.

        Parameters
        ----------
        num_elements: int
            The number of elements in the array. Because the storage may have a larger allocated memory, we can not
            infer the desired number of elements.

        dtype: str, default 'float32'
            The type of data in this storage.

        Returns
        -------
        ret: numpy.ndarray
            A numpy ndarray with one dimension that share the same data as the storage.
        """
        dtype2ctype = {
            'float32': ctypes.c_float,
            'int64': ctypes.c_int64
        }

        if self.device != 'cpu':
            raise ValueError('The storage must be cpu storage. Please use .cpu() to convert first.')
        buf = (dtype2ctype[dtype] * num_elements).from_address(self.addr)
        buf._hidet_storage = self  # so this storage will not be freed as long as the buffer not been freed.
        assert ctypes.sizeof(buf) <= self.num_bytes, 'Trying to view a storage as a larger array'
        with warnings.catch_warnings():
            # temporarily ignore a warning due to python bug.
            # See: https://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array
            warnings.simplefilter('ignore')
            return np.ctypeslib.as_array(buf)


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
            if addr == 0 and allocated != 0:
                # out of memory
                self.clear()
                addr = self.storage_device.allocate(allocated)
                if addr == 0:
                    raise MemoryError('Can not allocate memory from {} device, allocated {}, free {}, requesting {}.'.format(
                        self.storage_device.name(),
                        nbytes2str(self.storage_device.allocated_memory()),
                        nbytes2str(self.storage_device.free_memory()),
                        nbytes2str(allocated)
                    ))
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
            self.clear()

    def clear(self):
        for block_list in self.memory_blocks.values():
            for storage in block_list:
                self.storage_device.free(storage.addr)
                storage.addr = 0
        # print('Cleared memory pool, returned {} memory back to {} device'.format(
        #     nbytes2str(self.reserved_size), self.storage_device.name()
        # ))
        self.memory_blocks.clear()
        self.reserved_size = 0

    def status(self) -> str:
        allocated = self.storage_device.allocated_memory()
        peak_allocated = self.storage_device.peak_allocated_memory()
        items = [
            ['Allocated', allocated],
            ['Peak', peak_allocated],
            ['Reserved', self.reserved_size],
            ['Active', allocated - self.reserved_size]
        ]
        lines = [
            'Status of {} memory pool'.format(self.storage_device.name()),
            *['{:>12}: {}'.format(name, nbytes2str(nbytes)) for name, nbytes in items]
        ]
        return '\n'.join(lines)

    def __str__(self):
        return self.status()

    def __del__(self):
        self.clear()


class CudaStorageDevice(StorageDevice):
    def __init__(self):
        self.addr2nbytes = {}
        self._peak_allocated_memory = 0
        self._allocated_memory = 0

    def name(self):
        return 'cuda'

    def allocate(self, nbytes):
        addr = cuda_api.malloc_async(nbytes)
        if addr == 0 and nbytes != 0:
            # out of memory
            return 0
        self._allocated_memory += nbytes
        self._peak_allocated_memory = max(self._peak_allocated_memory, self._allocated_memory)
        self.addr2nbytes[addr] = nbytes
        return addr

    def free(self, addr):
        cuda_api.free_async(addr)
        self._allocated_memory -= self.addr2nbytes.pop(addr)

    def allocated_memory(self) -> int:
        return self._allocated_memory

    def peak_allocated_memory(self) -> int:
        return self._peak_allocated_memory

    def free_memory(self) -> int:
        return cuda_api.mem_info()[0]

    def total_memory(self) -> int:
        return cuda_api.mem_info()[1]


class CpuStorageDevice(StorageDevice):
    def __init__(self):
        self.addr2nbytes = {}
        self._allocated_memory = 0
        self._peak_allocated_memory = 0

    def name(self):
        return 'cpu'

    def allocate(self, nbytes):
        addr = cuda_api.malloc_host(nbytes)
        if addr == 0 and nbytes != 0:
            return 0
        self._allocated_memory += nbytes
        self._peak_allocated_memory = max(self._peak_allocated_memory, self._allocated_memory)
        self.addr2nbytes[addr] = nbytes
        return addr

    def free(self, addr):
        cuda_api.free_host(addr)
        self._allocated_memory -= self.addr2nbytes.pop(addr)

    def allocated_memory(self) -> int:
        return self._allocated_memory

    def peak_allocated_memory(self) -> int:
        return self._peak_allocated_memory

    def free_memory(self) -> int:
        return 0

    def total_memory(self) -> int:
        return 0


cpu_pool = MemoryPool(
    storage_device=CpuStorageDevice(),
    block_size=4 * 1024,  # 4 KiB
    max_reserve_size=128 * 1024 * 1024  # 128 MiB
)

cuda_pool = MemoryPool(
    storage_device=CudaStorageDevice(),
    block_size=4 * 1024,  # 4 KiB
    max_reserve_size=3 * 1024 * 1024 * 1024  # 5 GiB
)
