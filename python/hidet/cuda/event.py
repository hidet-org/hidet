# pylint: disable=no-name-in-module, c-extension-no-member
from __future__ import annotations
from cuda import cudart
from cuda.cudart import cudaEvent_t


class Event:
    def __init__(self, enable_timing: bool = False, blocking: bool = False):
        self._enable_timing: bool = enable_timing
        self._blocking: bool = blocking

        self._handle: cudaEvent_t
        if not enable_timing:
            flags = cudart.cudaEventDisableTiming
        else:
            flags = cudart.cudaEventDefault
        err, self._handle = cudart.cudaEventCreateWithFlags(flags)
        assert err == 0

    def __del__(self):
        err = cudart.cudaEventDestroy(self._handle)
        assert err == 0

    def handle(self) -> cudaEvent_t:
        return self._handle

    def elapsed_time(self, start_event: Event) -> float:
        # pylint: disable=protected-access
        if not self._enable_timing or not start_event._enable_timing:
            raise RuntimeError("Event does not have timing enabled")
        err, elapsed_time = cudart.cudaEventElapsedTime(start_event._handle, self._handle)
        assert err == 0
        return elapsed_time

    def record(self, stream):
        from hidet.cuda.stream import Stream

        if not isinstance(stream, Stream):
            raise TypeError("stream must be a Stream")
        err = cudart.cudaEventRecord(self._handle, stream.handle())
        assert err

    def synchronize(self):
        if not self._blocking:
            raise RuntimeError("Event does not have blocking enabled")
        err = cudart.cudaEventSynchronize(self._handle)
        if err != 0:
            raise RuntimeError("cudaEventSynchronize failed with error: {}".format(err.name))
