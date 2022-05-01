from ctypes import c_void_p
from .ffi import get_func


class RuntimeAPI:
    _set_current_stream = get_func('set_cuda_stream', [c_void_p], None)
    _get_current_stream = get_func('get_cuda_stream', [], c_void_p)

    @staticmethod
    def set_current_stream(stream_handle: int) -> None:
        RuntimeAPI._set_current_stream(c_void_p(stream_handle))

    @staticmethod
    def get_current_stream() -> int:
        p = RuntimeAPI._get_current_stream()
        return p.value


runtime_api = RuntimeAPI()

