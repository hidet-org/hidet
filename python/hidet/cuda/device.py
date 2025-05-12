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
import subprocess
import warnings
import logging
from typing import Tuple, Optional, List
from functools import lru_cache
from cuda.bindings import runtime as cudart
from cuda.bindings.runtime import cudaDeviceProp


logger = logging.getLogger(__name__)


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

    Use ctypes to check if libcuda.so is available instead of calling cudart directly.

    Returns
    -------
    ret: bool
        Whether CUDA is available.
    """
    import ctypes.util

    if ctypes.util.find_library('cuda'):
        return True
    return False


def is_available() -> bool:
    """
    mimic torch.cuda.is_available()
    """
    return available()


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


def get_application_freqs(device_id: int) -> Tuple[int, int]:
    """
    Get the application clock frequencies of a CUDA device.

    Parameters
    ----------
    device_id: int
        The ID of the device to query.

    Returns
    -------
    (sm_clock, mem_clock): Tuple[int, int]
        The application clock frequencies of the device.
    """
    try:
        # Query application clocks for the specified device
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.applications.gr,clocks.applications.mem",
                "--format=csv,noheader,nounits",
                f"--id={device_id}",
            ],
            text=True,
        )
        output = output.split(",")
        output = [x.strip() for x in output]
        assert len(output) == 2
        if output[0].isdigit() and output[1].isdigit():
            sm_clock, mem_clock = [int(x) for x in output]
        else:
            sm_clock, mem_clock = None, None
        return sm_clock, mem_clock
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to retrieve application clock frequencies: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Failed to parse clock frequencies: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"nvidia-smi not found: {e}") from e


def set_application_freqs(device_id: int, sm_clock: int, mem_clock: int) -> None:
    """
    Set the application clock frequencies of a CUDA device.

    Parameters
    ----------
    device_id: int
        The ID of the device to configure.
    sm_clock: int
        The desired SM (Streaming Multiprocessor) clock frequency in MHz.
    mem_clock: int
        The desired memory clock frequency in MHz.

    Raises
    ------
    RuntimeError
        If setting the application clock frequencies fails.
    """
    warnings.simplefilter("once")
    try:
        # Set application clocks for the specified device
        logger.info(
            f"Setting application clocks for device {device_id}: "
            f"SM clock = {sm_clock} MHz, Memory clock = {mem_clock} MHz"
        )
        subprocess.check_call(
            ["sudo", "nvidia-smi", f"--applications-clocks={mem_clock},{sm_clock}", f"--id={device_id}"],
            stdout=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Failed to set application clock frequencies: {e}")
    except FileNotFoundError as e:
        warnings.warn(f"Failed to set application clock frequencies: {e}")


def get_supported_graphics_clocks(device_id: int) -> List[int]:
    """
    Get the supported graphics clock frequencies of a CUDA device.

    Parameters
    ----------
    device_id: int
        The ID of the device to query.

    Returns
    -------
    List[int]
        A list of supported graphics clock frequencies for the device.
    """
    try:
        # Query supported graphics clock frequencies for the specified device
        output = subprocess.check_output(
            ["nvidia-smi", "--query-supported-clocks=gr", "--format=csv,noheader,nounits", f"--id={device_id}"],
            text=True,
        )
        # Parse the output into a list of integers
        clocks = [int(clock) for clock in output.splitlines()]
        return clocks
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to retrieve supported graphics clocks: {e}") from e
    except ValueError as e:
        raise RuntimeError(f"Failed to parse supported clock frequencies: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"nvidia-smi not found: {e}") from e
