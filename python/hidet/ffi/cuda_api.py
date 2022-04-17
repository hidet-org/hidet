from typing import Tuple
from ctypes import c_uint64, c_uint32, c_float, c_uint8, byref, POINTER
from hidet.ffi.ffi import get_func


class CudaAPI:
    # memory related apis
    _mem_info = get_func('hidet_cuda_mem_info', [POINTER(c_uint64), POINTER(c_uint64)], None)
    _malloc_async = get_func('hidet_cuda_malloc_async', [c_uint64], c_uint64)
    _malloc_host = get_func('hidet_cuda_malloc_host', [c_uint64], c_uint64)
    _free_async = get_func('hidet_cuda_free_async', [c_uint64], None)
    _free_host = get_func('hidet_cuda_free_host', [c_uint64], None)
    _memset_async = get_func('hidet_cuda_memset_async', [c_uint64, c_uint64, c_uint8], None)
    _memcpy_async = get_func('hidet_cuda_memcpy_async', [c_uint64, c_uint64, c_uint64, c_uint32], None)
    _mem_pool_trim_to = get_func('hidet_cuda_mem_pool_trim_to', [c_uint64], None)
    # device control
    _device_synchronization = get_func('hidet_cuda_device_synchronization', [], None)
    # random number generation
    _generate_uniform = get_func('hidet_curand_generate_uniform', [c_uint64, c_uint64], None)
    _generate_normal = get_func('hidet_curand_generate_normal', [c_uint64, c_uint64, c_float, c_float], None)

    @classmethod
    def mem_info(cls) -> Tuple[int, int]:
        free_bytes = c_uint64(0)
        total_bytes = c_uint64(0)
        cls._mem_info(byref(free_bytes), byref(total_bytes))
        return free_bytes.value, total_bytes.value

    @classmethod
    def malloc_async(cls, num_bytes: int) -> int:
        return cls._malloc_async(num_bytes)

    @classmethod
    def malloc_host(cls, num_bytes: int) -> int:
        return cls._malloc_host(num_bytes)

    @classmethod
    def free_async(cls, addr: int) -> None:
        return cls._free_async(addr)

    @classmethod
    def free_host(cls, addr: int) -> None:
        return cls._free_host(addr)

    @classmethod
    def memset_async(cls, addr: int, num_bytes: int, value: int) -> None:
        return cls._memset_async(addr, num_bytes, value)

    HostToHost = 0
    HostToDevice = 1
    DeviceToHost = 2
    DeviceToDevice = 3

    @classmethod
    def memcpy_async(cls, src_addr: int, dst_addr: int, num_bytes: int, kind: int) -> None:
        assert 0 <= kind <= 3
        cls._memcpy_async(src_addr, dst_addr, num_bytes, kind)
        if kind != cls.DeviceToDevice:
            cls.device_synchronization()

    @classmethod
    def mem_pool_trim_to(cls, min_bytes_to_keep: int) -> None:
        cls._mem_pool_trim_to(min_bytes_to_keep)

    @classmethod
    def device_synchronization(cls) -> None:
        return cls._device_synchronization()

    @classmethod
    def generate_uniform(cls, addr: int, num_elements: int) -> None:
        return cls._generate_uniform(addr, num_elements)

    @classmethod
    def generate_normal(cls, addr: int, num_elements: int, mean: float, stddev: float) -> None:
        return cls._generate_normal(addr, num_elements, mean, stddev)


cuda_api = CudaAPI()
