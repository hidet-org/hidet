from typing import Tuple
from ctypes import c_uint64, c_uint32, c_float, c_uint8, byref, POINTER, c_char_p
from hidet.ffi.ffi import get_func


class CudaAPI:
    # memory related apis
    _mem_info = get_func('hidet_cuda_mem_info', [POINTER(c_uint64), POINTER(c_uint64)], None)
    _malloc_async = get_func('hidet_cuda_malloc_async', [c_uint64], c_uint64)
    _malloc_host = get_func('hidet_cuda_malloc_host', [c_uint64], c_uint64)
    _free_async = get_func('hidet_cuda_free_async', [c_uint64], None)
    _free_host = get_func('hidet_cuda_free_host', [c_uint64], None)
    _memset_async = get_func('hidet_cuda_memset_async', [c_uint64, c_uint64, c_uint8], None)
    _memcpy_async = get_func('hidet_cuda_memcpy_async', [c_uint64, c_uint64, c_uint64, c_uint32, c_uint64], None)
    _mem_pool_trim_to = get_func('hidet_cuda_mem_pool_trim_to', [c_uint64], None)
    # device control
    _device_synchronize = get_func('hidet_cuda_device_synchronize', [], None)
    # stream and event
    _stream_create = get_func('hidet_cuda_stream_create', [], c_uint64)
    _stream_destroy = get_func('hidet_cuda_stream_destroy', [c_uint64], None)
    _stream_synchronize = get_func('hidet_cuda_stream_synchronize', [c_uint64], None)
    _event_create = get_func('hidet_cuda_event_create', [], c_uint64)
    _event_destroy = get_func('hidet_cuda_event_destroy', [c_uint64], None)
    _event_elapsed_time = get_func('hidet_cuda_event_elapsed_time', [c_uint64, c_uint64], c_float)
    _event_record = get_func('hidet_cuda_event_record', [c_uint64, c_uint64], None)
    # cuda graph
    _graph_create = get_func('hidet_cuda_graph_create', [], c_uint64)
    _graph_destroy = get_func('hidet_cuda_graph_destroy', [c_uint64], None)
    _stream_begin_capture = get_func('hidet_cuda_stream_begin_capture', [c_uint64], None)
    _stream_end_capture = get_func('hidet_cuda_stream_end_capture', [c_uint64], c_uint64)
    _graph_instantiate = get_func('hidet_cuda_graph_instantiate', [c_uint64], c_uint64)
    _graph_exec_launch = get_func('hidet_cuda_graph_exec_launch', [c_uint64, c_uint64], None)
    _graph_exec_destroy = get_func('hidet_cuda_graph_exec_destroy', [c_uint64], None)
    # profiler control
    _profiler_start = get_func('hidet_cuda_profiler_start', [], None)
    _profiler_stop = get_func('hidet_cuda_profiler_stop', [], None)
    # random number generation
    _generate_uniform = get_func('hidet_curand_generate_uniform', [c_uint64, c_uint64], None)
    _generate_normal = get_func('hidet_curand_generate_normal', [c_uint64, c_uint64, c_float, c_float], None)
    # get device property
    _device_property = get_func('hidet_cuda_get_device_property', [c_uint64, c_char_p], c_uint64)

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
    def memcpy_async(cls, src_addr: int, dst_addr: int, num_bytes: int, kind: int, stream: int = 0) -> None:
        assert 0 <= kind <= 3
        cls._memcpy_async(src_addr, dst_addr, num_bytes, kind, stream)

    @classmethod
    def memcpy(cls, src_addr: int, dst_addr: int, num_bytes: int, kind: int) -> None:
        cls.memcpy_async(src_addr, dst_addr, num_bytes, kind, stream=0)
        if kind != cls.DeviceToDevice:
            cls.device_synchronize()

    @classmethod
    def mem_pool_trim_to(cls, min_bytes_to_keep: int) -> None:
        cls._mem_pool_trim_to(min_bytes_to_keep)

    @classmethod
    def device_synchronize(cls) -> None:
        return cls._device_synchronize()

    @classmethod
    def generate_uniform(cls, addr: int, num_elements: int) -> None:
        return cls._generate_uniform(addr, num_elements)

    @classmethod
    def generate_normal(cls, addr: int, num_elements: int, mean: float, stddev: float) -> None:
        return cls._generate_normal(addr, num_elements, mean, stddev)

    @classmethod
    def create_stream(cls) -> int:
        return cls._stream_create()

    @classmethod
    def destroy_stream(cls, stream_handle: int) -> int:
        return cls._stream_destroy(stream_handle)

    @classmethod
    def stream_synchronize(cls, stream_handle: int) -> None:
        return cls._stream_synchronize(stream_handle)

    @classmethod
    def create_event(cls) -> int:
        return cls._event_create()

    @classmethod
    def destroy_event(cls, event_handle: int) -> None:
        return cls._event_destroy(event_handle)

    @classmethod
    def event_elapsed_time(cls, start_event_handle: int, end_event_handle: int) -> float:
        return cls._event_elapsed_time(start_event_handle, end_event_handle)

    @classmethod
    def event_record(cls, event_handle: int, stream_handle: int) -> None:
        return cls._event_record(event_handle, stream_handle)

    @staticmethod
    def create_graph() -> int:
        return CudaAPI._graph_create()

    @staticmethod
    def destroy_graph(graph_handle: int) -> None:
        return CudaAPI._graph_destroy(graph_handle)

    @staticmethod
    def stream_begin_capture(stream_handle: int) -> None:
        return CudaAPI._stream_begin_capture(stream_handle)

    @staticmethod
    def stream_end_capture(stream_handle: int) -> int:
        # return the cuda graph handle captured in this stream
        return CudaAPI._stream_end_capture(stream_handle)

    @staticmethod
    def instantiate_graph(graph_handle: int) -> int:
        return CudaAPI._graph_instantiate(graph_handle)

    @staticmethod
    def launch_graph_exec(graph_exec_handle: int, stream_handle: int) -> None:
        return CudaAPI._graph_exec_launch(graph_exec_handle, stream_handle)

    @staticmethod
    def destroy_graph_exec(graph_exec_handle: int) -> None:
        return CudaAPI._graph_exec_destroy(graph_exec_handle)

    @staticmethod
    def start_profiler() -> None:
        CudaAPI._profiler_start()
        CudaAPI.device_synchronize()

    @staticmethod
    def stop_profiler() -> None:
        CudaAPI._profiler_stop()
        CudaAPI.device_synchronize()

    PropertyMultiProcessorCount = 'multiProcessorCount'
    PropertyMajor = 'major'
    PropertyMinor = 'minor'

    @staticmethod
    def device_property(name: str, device_id: int = 0) -> int:
        return CudaAPI._device_property(device_id, name.encode('utf-8'))

    @staticmethod
    def compute_capability() -> Tuple[int, int]:
        return (CudaAPI.device_property(CudaAPI.PropertyMajor), CudaAPI.device_property(CudaAPI.PropertyMinor))


cuda = CudaAPI()

if __name__ == '__main__':
    print(cuda.device_property(cuda.PropertyMultiProcessorCount))
