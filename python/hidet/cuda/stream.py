# pylint: disable=no-name-in-module, c-extension-no-member
from __future__ import annotations
from typing import Union, Optional
from cuda import cudart
from cuda.cudart import cudaStream_t
from .event import Event


class Stream:
    """
    A CUDA stream.

    Parameters
    ----------
    blocking: bool
        Whether to enable the implicit synchronization between this stream and the default stream.
        When enabled, any operation enqueued in the stream will wait for all previous operations in the default stream
        to complete before beginning execution.

    priority: int
        The priority of the stream. The priority is a hint to the CUDA driver that it can use to reorder
        operations in the stream relative to other streams. The priority can be 0 (default priority) and
        -1 (high priority). By default, all streams are created with priority 0.
    """

    def __init__(self, blocking: bool = False, priority: int = 0, **kwargs):
        self._handle: cudaStream_t
        self._external: bool
        if 'handle' in kwargs:
            self._handle: cudaStream_t = kwargs['handle']
            self._external = True
        else:
            flags = cudart.cudaStreamNonBlocking if not blocking else cudart.cudaStreamDefault
            err, self._handle = cudart.cudaStreamCreateWithPriority(flags, priority)
            assert err == 0, err
            self._external = False

    def __del__(self):
        if cudart is None or cudart.cudaStreamDestroy is None:  # cudart has been unloaded
            return
        if not self._external:
            (err,) = cudart.cudaStreamDestroy(self._handle)
            assert err == 0, err

    def __int__(self):
        return int(self._handle)

    def __hash__(self):
        return hash(self._handle)

    def __eq__(self, other: Stream):
        return isinstance(other, Stream) and self._handle == other._handle

    def handle(self) -> cudaStream_t:
        """
        Get the handle of the stream.

        Returns
        -------
        handle: cudaStream_t
            The handle of the stream.
        """
        return self._handle

    def synchronize(self) -> None:
        """
        Block the current host thread until the stream completes all operations.
        """
        (err,) = cudart.cudaStreamSynchronize(self._handle)
        if err != 0:
            raise RuntimeError("cudaStreamSynchronize failed with error: {}".format(err.name))

    def wait_event(self, event: Event) -> None:
        """
        Let the subsequent operations in the stream wait for the event to complete. The event might be recorded in
        another stream. The host thread will not be blocked.

        Parameters
        ----------
        event: Event
            The event to wait for.
        """
        (err,) = cudart.cudaStreamWaitEvent(self._handle, event.handle(), 0)
        assert err == 0, err


class ExternalStream(Stream):
    """
    An external CUDA stream created from a handle.

    Parameters
    ----------
    handle: int or cudaStream_t
        The handle of the stream.
    """

    def __init__(self, handle: Union[cudaStream_t, int]):
        super().__init__(handle=handle)


_default_stream: Stream = ExternalStream(cudart.cudaStreamDefault)
_non_blocking_stream: Stream = ExternalStream(cudart.cudaStreamNonBlocking)
_current_stream: Stream = _default_stream


class StreamContext:
    def __init__(self, stream: Stream):  # pylint: disable=redefined-outer-name
        self._stream: Stream = stream
        self._prev: Optional[Stream] = None

    def __enter__(self):
        from hidet.ffi import runtime_api

        global _current_stream
        self._prev = _current_stream
        _current_stream = self._stream
        runtime_api.set_current_stream(_current_stream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        from hidet.ffi import runtime_api

        global _current_stream
        _current_stream = self._prev
        runtime_api.set_current_stream(_current_stream)


def current_stream() -> Stream:
    """
    Get the current stream.

    Returns
    -------
    stream: Stream
        The current stream.
    """
    return _current_stream


def default_stream() -> Stream:
    """
    Get the default stream.

    Returns
    -------
    stream: Stream
        The default stream.

    """
    return _default_stream


def stream(s: Stream) -> StreamContext:
    """
    Set the current stream.


    Parameters
    ----------
    s: Stream
        The stream to set.

    Examples
    --------
    >>> import hidet
    >>> stream = hidet.cuda.Stream()
    >>> with hidet.cuda.stream(stream):
    >>>    ... # all hidet cuda kernels will be executed in the stream
    """
    return StreamContext(s)
