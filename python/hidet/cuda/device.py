# pylint: disable=no-name-in-module, c-extension-no-member
from typing import Tuple
from functools import lru_cache
from cuda import cudart
from cuda.cudart import cudaDeviceProp


def device_count() -> int:
    err, count = cudart.cudaGetDeviceCount()
    assert err == 0
    return count


def synchronize():
    err = cudart.cudaDeviceSynchronize()
    if err != 0:
        raise RuntimeError("cudaDeviceSynchronize failed with error: {}".format(err.name))


@lru_cache()
def device_properties(device_id: int = 0) -> cudaDeviceProp:
    err, prop = cudart.cudaGetDeviceProperties(device_id)
    assert err == 0
    return prop


def compute_capability(device_id: int = 0) -> Tuple[int, int]:
    prop = device_properties(device_id)
    return prop.major, prop.minor


def profiler_start():
    err = cudart.cudaProfilerStart()
    assert err == 0


def profiler_stop():
    err = cudart.cudaProfilerStop()
    assert err == 0
