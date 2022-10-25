from ctypes import c_uint64, c_float, c_int32, c_int64
from hidet.ffi.ffi import get_func


class CudaKernels:
    # kernels
    _fill_value_int32 = get_func('hidet_cuda_fill_value_int32', [c_uint64, c_uint64, c_int32], None)
    _fill_value_int64 = get_func('hidet_cuda_fill_value_int64', [c_uint64, c_uint64, c_int64], None)
    _fill_value_float32 = get_func('hidet_cuda_fill_value_float32', [c_uint64, c_uint64, c_float], None)

    @classmethod
    def fill_value(cls, addr: int, num_elements: int, value, dtype: str) -> None:
        if dtype == 'float32':
            return cls._fill_value_float32(addr, num_elements, value)
        elif dtype == 'int64':
            return cls._fill_value_int64(addr, num_elements, value)
        elif dtype == 'int32':
            return cls._fill_value_int32(addr, num_elements, value)
        else:
            raise NotImplementedError('Currently do not support fill value with dtype "{}"'.format(dtype))


cuda_kernels = CudaKernels()
