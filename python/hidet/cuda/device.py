# pylint: disable=no-name-in-module, c-extension-no-member
from typing import Tuple
from functools import lru_cache
from cuda import cudart
from cuda.cudart import cudaDeviceProp


@lru_cache(maxsize=1)
def available() -> bool:
    return device_count() > 0


@lru_cache(maxsize=1)
def device_count() -> int:
    err, count = cudart.cudaGetDeviceCount()
    assert err == 0, err
    return count


@lru_cache(maxsize=None)
def properties(device_id: int = 0) -> cudaDeviceProp:
    err, prop = cudart.cudaGetDeviceProperties(device_id)
    assert err == 0, err
    return prop


def compute_capability(device_id: int = 0) -> Tuple[int, int]:
    prop = properties(device_id)
    return prop.major, prop.minor


def synchronize():
    (err,) = cudart.cudaDeviceSynchronize()
    if err != 0:
        raise RuntimeError("cudaDeviceSynchronize failed with error: {}".format(err.name))


def profiler_start():
    (err,) = cudart.cudaProfilerStart()
    assert err == 0, err


def profiler_stop():
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
