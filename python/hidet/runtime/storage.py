import ctypes
import numpy as np
from hidet.ffi.cuda_api import CudaAPI


class Storage:
    def __init__(self, addr, num_bytes):
        self.addr: int = addr
        self.num_bytes: int = num_bytes

    def __del__(self):
        raise NotImplementedError()

    def cpu(self):
        if isinstance(self, HostStorage):
            return self
        elif isinstance(self, CudaStorage):
            host_storage = HostStorage(self.num_bytes)
            CudaAPI.memcpy_async(src_addr=self.addr, dst_addr=host_storage.addr, num_bytes=self.num_bytes, kind=CudaAPI.DeviceToHost)
            return host_storage
        else:
            raise NotImplementedError()

    def cuda(self):
        if isinstance(self, CudaStorage):
            return self
        elif isinstance(self, HostStorage):
            cuda_storage = CudaStorage(self.num_bytes)
            CudaAPI.memcpy_async(src_addr=self.addr, dst_addr=cuda_storage.addr, num_bytes=self.num_bytes, kind=CudaAPI.HostToDevice)
            return cuda_storage
        else:
            raise NotImplementedError()

    @staticmethod
    def new(device: str, num_bytes: int):
        device2storage = {
            'cuda': CudaStorage,
            'host': HostStorage,
            'cpu': HostStorage,
        }
        if device in device2storage:
            return device2storage[device](num_bytes)
        else:
            raise ValueError("Unrecognized device '{}', candidates: {}".format(device, list(device2storage)))

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
        if not isinstance(self, HostStorage):
            raise ValueError('The storage must be host storage. Please use .cpu() to convert first.')
        if dtype == 'float32':
            assert self.num_bytes % 4 == 0
            num_elements = self.num_bytes // 4
            buf = (ctypes.c_float * num_elements).from_address(self.addr)
            buf._hidet_storage = self  # so this HostStorage will not be freed as long as the buffer not been freed.
            return np.ctypeslib.as_array(buf)
        else:
            raise NotImplementedError()


class CudaStorage(Storage):
    def __init__(self, num_bytes: int):
        super().__init__(CudaAPI.malloc_async(num_bytes), num_bytes)

    def __del__(self):
        CudaAPI.free_async(self.addr)


class HostStorage(Storage):
    def __init__(self, num_bytes: int):
        super().__init__(CudaAPI.malloc_host(num_bytes), num_bytes)

    def __del__(self):
        CudaAPI.free_host(self.addr)
