from ctypes import c_uint64, c_uint32, c_float, c_uint8
from hidet.ffi.ffi import get_func


class CudaAPI:
    # memory related apis
    _malloc_async = get_func('hidet_cuda_malloc_async', [c_uint64], c_uint64)
    _malloc_host = get_func('hidet_cuda_malloc_host', [c_uint64], c_uint64)
    _free_async = get_func('hidet_cuda_free_async', [c_uint64], None)
    _free_host = get_func('hidet_cuda_free_host', [c_uint64], None)
    _memset_async = get_func('hidet_cuda_memset_async', [c_uint64, c_uint64, c_uint8], None)
    _memcpy_async = get_func('hidet_cuda_memcpy_async', [c_uint64, c_uint64, c_uint64, c_uint32], None)
    # device control
    _device_synchronization = get_func('hidet_cuda_device_synchronization', [], None)
    # random number generation
    _generate_uniform = get_func('hidet_curand_generate_uniform', [c_uint64, c_uint64], None)
    _generate_normal = get_func('hidet_curand_generate_normal', [c_uint64, c_uint64, c_float, c_float], None)
    # kernels
    _fill_value = get_func('hidet_cuda_fill_value', [c_uint64, c_uint64, c_float], None)

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
        return cls._memcpy_async(src_addr, dst_addr, num_bytes, kind)

    @classmethod
    def device_synchronization(cls) -> None:
        return cls._device_synchronization()

    @classmethod
    def generate_uniform(cls, addr: int, num_elements: int) -> None:
        return cls._generate_uniform(addr, num_elements)

    @classmethod
    def generate_normal(cls, addr: int, num_elements: int, mean: float, stddev: float) -> None:
        return cls._generate_normal(addr, num_elements, mean, stddev)

    @classmethod
    def fill_value(cls, addr: int, num_elements: int, value: float) -> None:
        return cls._fill_value(addr, num_elements, value)
