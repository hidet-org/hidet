# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=no-name-in-module, c-extension-no-member
from __future__ import annotations
from typing import Optional, Dict
from hip import hip
from hidet.utils import exiting
from .event import Event
from .device import HipDeviceContext, current_device


def _get_device_id(device) -> int:
    """
    Get the device ID from a device.

    Parameters
    ----------
    device: Device or int, optional
        The device.

    Returns
    -------
    device_id: int
        The device ID.
    """
    if device is None:
        return current_device()
    elif isinstance(device, int):
        return device
    else:
        from hidet.runtime.device import Device

        if isinstance(device, Device):
            assert device.is_hip(), "device must be a HIP device"
            if device.id is not None:
                return device.id
            else:
                return current_device()
        else:
            raise TypeError(f"device must be a hidet.Device or int, not {type(device)}")


class Stream:
    """
    A HIP stream.

    Parameters
    ----------
    device: int or hidet.Device, optional
        The device on which to create the stream. If None, the current device will be used.

    blocking: bool
        Whether to enable the implicit synchronization between this stream and the default stream.
        When enabled, any operation enqueued in the stream will wait for all previous operations in the default stream
        to complete before beginning execution.

    priority: int
        The priority of the stream. The priority is a hint to the HIP driver that it can use to reorder
        operations in the stream relative to other streams. The priority can be 0 (default priority) and
        -1 (high priority). By default, all streams are created with priority 0.
    """

    def __init__(self, device=None, blocking: bool = False, priority: int = 0, **kwargs):
        import hidet

        self._device_id: int
        self._handle: int  # hipStream_t represented as a void* in C
        self._external: bool
        if 'handle' in kwargs:
            self._device_id = _get_device_id(device)
            self._handle = kwargs['handle']
            self._external = True
        else:
            self._device_id = _get_device_id(device)
            with hidet.hip.device(self._device_id):
                flags = hip.hipStreamNonBlocking if not blocking else hip.hipStreamDefault
                err, handle = hip.hipStreamCreateWithPriority(flags, priority)
                assert err == 0, str(err)
                self._handle = handle
            self._external = False

    def __int__(self):
        return int(self._handle)

    def __hash__(self):
        return hash(self._handle)

    def __eq__(self, other: Stream):
        if not isinstance(other, Stream):
            raise TypeError(f"cannot compare Stream with {type(other)}")
        return self._device_id == other._device_id and self._handle == other._handle

    def __del__(self, is_exiting=exiting.is_exiting):
        if is_exiting():
            return
        if not self._external:
            (err,) = hip.hipStreamDestroy(self._handle)
            assert err == 0, str(err)

    def device_id(self) -> int:
        """
        Get the device ID of the stream.

        Returns
        -------
        device_id: int
            The device ID of the stream.
        """
        return self._device_id

    def handle(self) -> int:
        """
        Get the handle of the stream.

        Returns
        -------
        handle: int (which represents a hipStream_t, of type void* in C)
            The handle of the stream.
        """
        return self._handle

    def synchronize(self) -> None:
        """
        Block the current host thread until the stream completes all operations.
        """
        (err,) = hip.hipStreamSynchronize(self._handle)
        assert err == 0, str(err)

    def wait_event(self, event: Event) -> None:
        """
        Let the subsequent operations in the stream wait for the event to complete. The event might be recorded in
        another stream. The host thread will not be blocked.

        Parameters
        ----------
        event: Event
            The event to wait for.
        """
        (err,) = hip.hipStreamWaitEvent(self._handle, event.handle(), 0)
        assert err == 0, str(err)


class ExternalStream(Stream):
    """
    An external HIP stream created from a handle.

    Parameters
    ----------
    handle: int - The handle of the stream.

    device_id: int, optional
        The device ID of the stream. If None, the current device will be used.
    """

    def __init__(self, handle: int, device_id: Optional[int] = None):
        super().__init__(handle=handle, device=device_id)


_current_streams: Dict[int, Stream] = {}


class StreamContext:
    def __init__(self, stream: Stream):  # pylint: disable=redefined-outer-name
        self.device_context = HipDeviceContext(stream.device_id())
        self.device: int = stream.device_id()
        self.stream: Stream = stream
        self.prev_stream: Optional[Stream] = None

    def __enter__(self):
        from hidet.ffi import runtime_api

        current_streams = _current_streams
        self.prev_stream = current_streams[self.device]
        current_streams[self.device] = self.stream
        self.device_context.__enter__()
        runtime_api.set_current_hip_stream(self.stream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        from hidet.ffi import runtime_api

        current_streams = _current_streams
        current_streams[self.device] = self.prev_stream
        runtime_api.set_current_hip_stream(self.prev_stream)
        self.device_context.__exit__(exc_type, exc_val, exc_tb)


def current_stream(device=None) -> Stream:
    """
    Get the current stream.

    Parameters
    ----------
    device: int or hidet.Device, optional
        The device on which to get the current stream. If None, the current device will be used.

    Returns
    -------
    stream: Stream
        The current stream.
    """
    device_id = _get_device_id(device)
    if device_id not in _current_streams:
        _current_streams[device_id] = ExternalStream(handle=0, device_id=device_id)
    return _current_streams[_get_device_id(device)]


def default_stream(device=None) -> Stream:
    """
    Get the default stream.

    Parameters
    ----------
    device: int or hidet.Device, optional
        The device on which to get the default stream. If None, the current device will be used.

    Returns
    -------
    stream: Stream
        The default stream.
    """
    return ExternalStream(handle=0, device_id=_get_device_id(device))


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
    >>> stream = hidet.hip.Stream()
    >>> with hidet.hip.stream(stream):
    >>>    ... # all hidet hip kernels will be executed in the stream
    """
    return StreamContext(s)
