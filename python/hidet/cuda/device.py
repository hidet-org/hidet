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
from typing import Tuple, Optional
from functools import lru_cache
from cuda import cudart
from cuda.cudart import cudaDeviceProp


class CudaDeviceContext:
    def __init__(self, device_id: int):
        self.device_id: int = device_id
        self.prev_device_id: Optional[int] = None

    def __enter__(self):
        self.prev_device_id = current_device()
        set_device(self.device_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_device(self.prev_device_id)


@lru_cache(maxsize=None)
def available() -> bool:
    """
    Returns True if CUDA is available, False otherwise.

    Returns
    -------
    ret: bool
        Whether CUDA is available.
    """
    return device_count() > 0


@lru_cache(maxsize=None)
def device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns
    -------
    count: int
        The number of available CUDA devices.
    """
    err, count = cudart.cudaGetDeviceCount()
    assert err == 0, err
    return count


@lru_cache(maxsize=None)
def properties(device_id: int = 0) -> cudaDeviceProp:
    """
    Get the properties of a CUDA device.

    Parameters
    ----------
    device_id: int
        The ID of the device.

    Returns
    -------
    prop: cudaDeviceProp
        The properties of the device.
    """
    err, prop = cudart.cudaGetDeviceProperties(device_id)
    assert err == 0, err
    return prop


def set_device(device_id: int):
    """
    Set the current cuda device.

    Parameters
    ----------
    device_id: int
        The ID of the cuda device.
    """
    (err,) = cudart.cudaSetDevice(device_id)
    assert err == 0, err


def current_device() -> int:
    """
    Get the current cuda device.

    Returns
    -------
    device_id: int
        The ID of the cuda device.
    """
    err, device_id = cudart.cudaGetDevice()
    assert err == 0, err
    return device_id


def device(device_id: int):
    """
    Context manager to set the current cuda device.

    Parameters
    ----------
    device_id: int
        The ID of the cuda device.

    Examples
    --------
    >>> import hidet
    >>> with hidet.cuda.device(0):
    >>>     # do something on device 0
    """
    return CudaDeviceContext(device_id)


@lru_cache(maxsize=None)
def compute_capability(device_id: int = 0) -> Tuple[int, int]:
    """
    Get the compute capability of a CUDA device.

    Parameters
    ----------
    device_id: int
        The ID of the device to query.

    Returns
    -------
    (major, minor): Tuple[int, int]
        The compute capability of the device.
    """
    prop = properties(device_id)
    return prop.major, prop.minor


def synchronize():
    """
    Synchronize the host thread with the device.

    This function blocks until the device has completed all preceding requested tasks.
    """
    (err,) = cudart.cudaDeviceSynchronize()
    if err != 0:
        raise RuntimeError("cudaDeviceSynchronize failed with error: {}".format(err.name))


def profiler_start():
    """
    Mark the start of a profiling range.
    """
    (err,) = cudart.cudaProfilerStart()
    assert err == 0, err


def profiler_stop():
    """
    Mark the end of a profiling range.
    """
    (err,) = cudart.cudaProfilerStop()
    assert err == 0, err


# We intentionally put the properties of all devices to the cache here.
#
# Reasons:
#   Hidet relies on the multiprocessing to parallelize the compilation. During the process, the forked process will
#   query the properties of the device. If we do not cache the properties, the forked process will query the device
#   via the cuda runtime API. However, the cuda runtime API does not work when the multiprocessing package is working
#   in the fork mode. With the properties of all the GPUs cached, the forked process will not run any cuda runtime API
#   and will not cause any problem.
if available():
    for i in range(device_count()):
        properties(i)
        compute_capability(i)
