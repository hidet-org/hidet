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

from hip import hip
from hidet.utils import exiting


class Event:
    """
    A HIP event.

    Parameters
    ----------
    enable_timing: bool
        When enabled, the event is able to record the time between itself and another event.

    blocking: bool
        When enabled, we can use the :meth:`synchronize` method to block the current host thread until the event
        completes.
    """

    def __init__(self, enable_timing: bool = False, blocking: bool = False):
        self._enable_timing: bool = enable_timing
        self._blocking: bool = blocking

        self._handle: int  # hipEvent_t represented as a void* in C
        if not enable_timing:
            flags = hip.hipEventDisableTiming
        else:
            flags = hip.hipEventDefault
        err, self._handle = hip.hipEventCreateWithFlags(flags)
        assert err == 0, str(err)

    def __del__(self, is_exiting=exiting.is_exiting):
        if is_exiting():
            return
        (err,) = hip.hipEventDestroy(self._handle)
        assert err == 0, str(err)

    def handle(self) -> int:
        """
        Get the handle of the event.

        Returns
        -------
        handle: int (which represents a hipEvent_t, of type void* in C)
            The handle of the event.
        """
        return self._handle

    def elapsed_time(self, start_event: int) -> float:
        """
        Get the elapsed time between the start event and this event in milliseconds.

        Parameters
        ----------
        start_event: int (which represents a hipEvent_t, of type void* in C)
            The start event.

        Returns
        -------
        elapsed_time: float
            The elapsed time in milliseconds.
        """
        # pylint: disable=protected-access
        if not self._enable_timing or not start_event._enable_timing:
            raise RuntimeError("Event does not have timing enabled")
        err, elapsed_time = hip.hipEventElapsedTime(start_event._handle, self._handle)
        assert err == 0, str(err)
        return elapsed_time

    def record(self, stream=None):
        """
        Record the event in the given stream.

        After the event is recorded:

        - We can synchronize the event to block the current host thread until all the tasks before the event are
          completed via :meth:`Event.synchronize`.
        - We can also get the elapsed time between the event and another event via :meth:`Event.elapsed_time` (when
          `enable_timing` is `True`).
        - We can also let another stream to wait for the event via :meth:`Stream.wait_event`.

        Parameters
        ----------
        stream: Stream, optional
            The stream where the event is recorded.
        """
        from hidet.hip.stream import Stream, current_stream

        if stream is None:
            stream = current_stream()
        if not isinstance(stream, Stream):
            raise TypeError("stream must be a Stream")
        (err,) = hip.hipEventRecord(self._handle, stream.handle())
        assert err == 0, str(err)

    def synchronize(self):
        """
        Block the current host thread until the tasks before the event are completed. This method is only available
        when `blocking` is `True` when creating the event, and the event is recorded in a stream.
        """
        if not self._blocking:
            raise RuntimeError("Event does not have blocking enabled")
        (err,) = hip.hipEventSynchronize(self._handle)
        assert err == 0, str(err)
