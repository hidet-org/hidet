# pylint: disable=no-name-in-module, c-extension-no-member
from __future__ import annotations
from typing import Union, Optional
from cuda import cudart
from cuda.cudart import cudaStream_t


class Stream:
    def __init__(self):
        self._handle: cudaStream_t
        err, self._handle = cudart.cudaStreamCreate()
        assert err == 0

    def __del__(self):
        if self._handle != 0:
            err = cudart.cudaStreamDestroy(self._handle)
            assert err == 0

    def __int__(self):
        return int(self._handle)

    def handle(self) -> cudaStream_t:
        return self._handle

    def synchronize(self) -> None:
        err = cudart.cudaStreamSynchronize(self._handle)
        if err != 0:
            raise RuntimeError("cudaStreamSynchronize failed with error: {}".format(err.name))

    @staticmethod
    def from_handle(handle: Union[cudaStream_t, int]) -> Stream:
        new_stream = Stream.__new__(Stream)
        new_stream._handle = cudaStream_t(handle)   # pylint: disable=protected-access
        return new_stream


_current_stream: Stream = Stream()
_default_stream: Stream = Stream.from_handle(cudart.cudaStreamDefault)


class StreamContext:
    def __init__(self, stream: Stream):  # pylint: disable=redefined-outer-name
        self._stream: Stream = stream
        self._prev: Optional[Stream] = None

    def __enter__(self):
        from hidet.ffi import runtime_api

        global _current_stream
        self._prev = _current_stream
        _current_stream = self._stream
        runtime_api.set_current_stream(stream_handle=_current_stream.handle())

    def __exit__(self, exc_type, exc_val, exc_tb):
        from hidet.ffi import runtime_api

        global _current_stream
        _current_stream = self._prev
        runtime_api.set_current_stream(stream_handle=_current_stream.handle())


def current_stream() -> Stream:
    return _current_stream


def default_stream() -> Stream:
    return _default_stream


def stream(s: Stream) -> StreamContext:
    return StreamContext(s)
