from __future__ import annotations
from typing import Callable, Dict, List, Optional, Union
import warnings
from collections import defaultdict
import ctypes
import numpy as np
import hidet.cuda
from hidet.cuda import Stream, current_stream
from hidet.ir import dtypes
from hidet.ir.type import DataType, data_type
from hidet.utils import green, prod


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


class StorageDevice:
    def __init__(self):
        self.froze = False

    def name(self):
        raise NotImplementedError()

    def freeze(self, flag: bool):  # when freeze, no allocate or free should happen. used in CudaGraph
        self.froze = flag

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


class CudaStorageDevice(StorageDevice):
    def __init__(self):
        super().__init__()
        self.addr2nbytes = {}
        self._peak_allocated_memory = 0
        self._allocated_memory = 0

    def name(self):
        return 'cuda'

    def allocate(self, nbytes):
        if self.froze:
            raise MemoryError('Should not allocate when the device is frozen.')

        addr = hidet.cuda.malloc_async(nbytes)
        if addr == 0 and nbytes != 0:
            # out of memory
            return 0
        self._allocated_memory += nbytes
        self._peak_allocated_memory = max(self._peak_allocated_memory, self._allocated_memory)
        self.addr2nbytes[addr] = nbytes
        return addr

    def free(self, addr):
        if self.froze:
            raise MemoryError('Should not free when the device is frozen.')

        hidet.cuda.free_async(addr)
        self._allocated_memory -= self.addr2nbytes.pop(addr)

    def allocated_memory(self) -> int:
        return self._allocated_memory

    def peak_allocated_memory(self) -> int:
        return self._peak_allocated_memory

    def free_memory(self) -> int:
        return hidet.cuda.memory_info()[0]

    def total_memory(self) -> int:
        return hidet.cuda.memory_info()[1]


class CpuStorageDevice(StorageDevice):
    def __init__(self):
        super().__init__()
        self.addr2nbytes = {}
        self._allocated_memory = 0
        self._peak_allocated_memory = 0

    def name(self):
        return 'cpu'

    def allocate(self, nbytes):
        if self.froze:
            raise MemoryError('Should not allocate when the device is frozen.')

        addr = hidet.cuda.malloc_host(nbytes)
        if addr == 0 and nbytes != 0:
            return 0
        self._allocated_memory += nbytes
        self._peak_allocated_memory = max(self._peak_allocated_memory, self._allocated_memory)
        self.addr2nbytes[addr] = nbytes
        return addr

    def free(self, addr):
        if self.froze:
            raise MemoryError('Should not free when the device is frozen.')

        hidet.cuda.free_host(addr)
        self._allocated_memory -= self.addr2nbytes.pop(addr)

    def allocated_memory(self) -> int:
        return self._allocated_memory

    def peak_allocated_memory(self) -> int:
        return self._peak_allocated_memory

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

    def __getstate__(self):
        raise ValueError()

    def __setstate__(self, state):
        raise ValueError()

    def cpu(self) -> Storage:
        if self.device == 'cpu':
            return self
        elif self.device == 'cuda':
            host_storage = self.new('cpu', self.num_bytes)
            hidet.cuda.memcpy(dst=host_storage.addr, src=self.addr, num_bytes=self.num_bytes)
            return host_storage
        else:
            raise NotImplementedError()

    def cpu_async(self, stream: Optional[Stream] = None):
        if self.device == 'cpu':
            return self
        elif self.device == 'cuda':
            host_storage = self.new('cpu', self.num_bytes)
            hidet.cuda.memcpy_async(dst=host_storage.addr, src=self.addr, num_bytes=self.num_bytes, stream=stream)
            return host_storage
        else:
            raise NotImplementedError()

    def cuda(self) -> Storage:
        if self.device == 'cuda':
            return self
        elif self.device == 'cpu':
            cuda_storage = self.new('cuda', self.num_bytes)
            hidet.cuda.memcpy(src=self.addr, dst=cuda_storage.addr, num_bytes=self.num_bytes)
            return cuda_storage
        else:
            raise NotImplementedError()

    def cuda_async(self, stream: Optional[Stream] = None):
        if self.device == 'cuda':
            return self
        elif self.device == 'cpu':
            cuda_storage = self.new('cuda', self.num_bytes)
            hidet.cuda.memcpy_async(dst=cuda_storage.addr, src=self.addr, num_bytes=self.num_bytes, stream=stream)
            return cuda_storage
        else:
            raise NotImplementedError()

    def copy(self) -> Storage:
        stream = current_stream()
        ret = self.copy_async(stream)
        stream.synchronize()
        return ret

    def copy_async(self, stream: Optional[Stream] = None) -> Storage:
        storage = Storage.new(self.device, self.num_bytes)
        hidet.cuda.memcpy_async(dst=storage.addr, src=self.addr, num_bytes=self.num_bytes, stream=stream)
        return storage

    @staticmethod
    def new(device: str, num_bytes: int) -> Storage:
        if device == 'cpu':
            return CpuMemoryPool.current().allocate(nbytes=num_bytes)
        elif device == 'cuda':
            return CudaMemoryPool.current().allocate(nbytes=num_bytes)
        else:
            raise ValueError("Unrecognized device '{}', candidates: {}".format(device, ['cpu', 'cuda']))

    def as_array(self, num_elements: int, dtype: Union[str, DataType] = 'float32', share_mem=True) -> np.ndarray:
        """
        Convert the storage to one-dimension numpy array.

        Parameters
        ----------
        num_elements: int
            The number of elements in the array. Because the storage may have a larger allocated memory, we can not
            infer the desired number of elements.

        dtype: str, default 'float32'
            The type of data in this storage.

        share_mem: bool, default True
            Whether to share the memory with the storage. If True, the returned array will share the memory with the
            storage. If False, the returned array will have a copy of the data in the storage.

        Returns
        -------
        ret: numpy.ndarray
            A numpy ndarray with one dimension that share the same data as the storage.
        """
        dtype = data_type(dtype)
        dtype2ctype = {
            dtypes.float64: ctypes.c_double,
            dtypes.float32: ctypes.c_float,
            dtypes.float16: ctypes.c_uint16,
            dtypes.int64: ctypes.c_int64,
            dtypes.int32: ctypes.c_int32,
            dtypes.int16: ctypes.c_int16,
            dtypes.int8: ctypes.c_int8,
            dtypes.uint64: ctypes.c_uint64,
            dtypes.uint32: ctypes.c_uint32,
            dtypes.uint16: ctypes.c_uint16,
            dtypes.uint8: ctypes.c_uint8,
            dtypes.boolean: ctypes.c_bool,
        }
        dtype2nptype = {dtypes.float16: np.float16}

        if self.device != 'cpu':
            raise ValueError('The storage must be cpu storage. Please use .cpu() to convert first.')
        buf = (dtype2ctype[dtype] * num_elements).from_address(self.addr)
        assert ctypes.sizeof(buf) <= self.num_bytes, 'Trying to view a storage as a larger array'
        with warnings.catch_warnings():
            # temporarily ignore a warning due to python bug.
            # See: https://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array
            warnings.simplefilter('ignore')
            array = np.ctypeslib.as_array(buf)
        if dtype in dtype2nptype:
            # reinterpret the array when needed
            array = array.view(dtype2nptype[dtype])
        if share_mem:
            # pylint: disable=protected-access
            buf._hidet_storage = self  # so this storage will not be freed as long as the buffer not been freed.
            return array
        else:
            return array.copy()


class TorchStorage(Storage):
    def __init__(self, torch_tensor):
        import torch

        if not isinstance(torch_tensor, torch.Tensor):
            raise ValueError('Expect a torch tensor, got {}'.format(type(torch_tensor).__name__))
        if not torch_tensor.is_contiguous():
            raise ValueError("Only contiguous torch tensor can be viewed as a Hidet storage.")
        self.torch_tensor = torch_tensor  # keep a reference to the tensor to prevent it being freed.
        super().__init__(
            device='cuda',
            addr=torch_tensor.data_ptr(),
            num_bytes=torch_tensor.element_size() * prod(torch_tensor.size()),
            free_handler=lambda storage: None,
        )


class MemoryPool:
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
                    raise MemoryError(
                        f'Can not allocate memory from {self.storage_device.name()} device, '
                        f'total {nbytes2str(self.storage_device.total_memory())}, '
                        f'hidet allocated {nbytes2str(self.storage_device.allocated_memory())}, '
                        f'free {nbytes2str(self.storage_device.free_memory())}, '
                        f'requesting {nbytes2str(allocated)}.'
                    )
        return Storage(device=self.storage_device.name(), addr=addr, num_bytes=allocated, free_handler=self.free)

    def free(self, storage: Storage):
        self.memory_blocks[storage.num_bytes].append(storage)
        self.reserved_size += storage.num_bytes
        if self.reserved_size > self.max_reserve_size:
            self.clear()

    def clear(self):
        hidet.cuda.synchronize()
        for block_list in self.memory_blocks.values():
            for storage in block_list:
                self.storage_device.free(storage.addr)
                storage.addr = 0
        self.memory_blocks.clear()
        self.reserved_size = 0

    def status(self) -> str:
        allocated = self.storage_device.allocated_memory()
        peak_allocated = self.storage_device.peak_allocated_memory()
        items = [
            ['Allocated', allocated],
            ['Peak', peak_allocated],
            ['Reserved', self.reserved_size],
            ['Active', allocated - self.reserved_size],
        ]
        lines = [
            'Status of {} memory pool'.format(self.storage_device.name()),
            *['{:>12}: {}'.format(name, nbytes2str(nbytes)) for name, nbytes in items],
        ]
        return '\n'.join(lines)

    def __str__(self):
        return self.status()

    def __del__(self):
        self.clear()


class CudaMemoryPool(MemoryPool):
    stack = []

    def __init__(self, block_size: int = 4096, max_reserve_size: int = 4 * 1024**3):
        super().__init__(CudaStorageDevice(), block_size, max_reserve_size)

    def __enter__(self):
        CudaMemoryPool.stack.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        CudaMemoryPool.stack.pop()

    @staticmethod
    def current() -> CudaMemoryPool:
        return CudaMemoryPool.stack[-1]


CudaMemoryPool.stack.append(CudaMemoryPool())


class CpuMemoryPool(MemoryPool):
    stack = []

    def __init__(self, block_size: int = 4096, max_reserve_size: int = 512 * 1024**2):
        super().__init__(CpuStorageDevice(), block_size, max_reserve_size)

    def __enter__(self):
        CpuMemoryPool.stack.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        CpuMemoryPool.stack.pop()

    @staticmethod
    def current() -> CpuMemoryPool:
        return CpuMemoryPool.stack[-1]


CpuMemoryPool.stack.append(CpuMemoryPool())
