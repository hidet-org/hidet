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
import ctypes.util

from hip import hip
from hip.hip import hipDeviceProp_t


class HipDeviceContext:
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
    Returns True if HIP is available, False otherwise.

    Use ctypes to check if libamdhip64.so is available.

    Returns
    -------
    ret: bool
        Whether HIP is available.
    """
    if ctypes.util.find_library('amdhip64'):
        return True
    return False


@lru_cache(maxsize=None)
def device_count() -> int:
    """
    Get the number of available HIP devices.

    Returns
    -------
    count: int
        The number of available HIP devices.
    """
    err, ret = hip.hipGetDeviceCount()
    assert err == 0, str(err)
    return ret


@lru_cache(maxsize=None)
def properties(device_id: int = 0) -> hipDeviceProp_t:
    """
    Get the properties of a HIP device.

    Parameters
    ----------
    device_id: int
        The ID of the device.

    Returns
    -------
    prop: hipDeviceProp_t
        The properties of the device.
    """
    prop = hipDeviceProp_t()
    (err,) = hip.hipGetDeviceProperties(prop, device_id)
    assert err == 0, str(err)
    return prop


def set_device(device_id: int):
    """
    Set the current HIP device.

    Parameters
    ----------
    device_id: int
        The ID of the HIP device.
    """
    (err,) = hip.hipSetDevice(device_id)
    assert err == 0, str(err)


def current_device() -> int:
    """
    Get the current HIP device.

    Returns
    -------
    device_id: int
        The ID of the HIP device.
    """
    err, device_id = hip.hipGetDevice()
    assert err == 0, str(err)
    return device_id


def device(device_id: int):
    """
    Context manager to set the current HIP device.

    Parameters
    ----------
    device_id: int
        The ID of the HIP device.

    Examples
    --------
    >>> import hidet
    >>> with hidet.hip.device(0):
    >>>     # do something on device 0
    """
    return HipDeviceContext(device_id)


@lru_cache(maxsize=None)
def compute_capability(device_id: int = 0) -> Tuple[int, int]:
    """
    Get the compute capability of a HIP device.

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
    (err,) = hip.hipDeviceSynchronize()
    assert err == 0, str(err)
