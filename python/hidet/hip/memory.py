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
from typing import Tuple, Union, Optional
from .ffi import (
    error_msg,
    hip_memory_info,
    hip_malloc,
    hip_malloc_async,
    hip_free,
    hip_free_async,
    hip_malloc_host,
    hip_free_host,
    hip_memset,
    hip_memset_async,
    hip_memcpy_device_to_host,
    hip_memcpy_host_to_device,
    hip_memcpy,
    hip_memcpy_async,
)
from .stream import Stream, current_stream


def memory_info() -> Tuple[int, int]:
    """
    Get the free and total memory on the current device in bytes.

    Returns
    -------
    (free, total): Tuple[int, int]
        The free and total memory on the current device in bytes.
    """
    error, _free, total = hip_memory_info()
    assert error == 0, error_msg("memory_info", error)
    return _free, total


def malloc(num_bytes: int) -> int:
    """
    Allocate memory on the current device.

    Parameters
    ----------
    num_bytes: int
        The number of bytes to allocate.

    Returns
    -------
    addr: int
        The address of the allocated memory.
    """
    error, ret = hip_malloc(num_bytes)
    assert error == 0, error_msg("malloc", error)
    return ret


def free(addr: int) -> None:
    """
    Free memory on the current hip device.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc` or
        :func:`malloc_async`.
    """
    error = hip_free(addr)
    assert error == 0, error_msg("free", error)


def malloc_async(num_bytes: int, stream: Optional[Union[Stream, int]] = None) -> int:
    """
    Allocate memory on the current hip device asynchronously.

    Parameters
    ----------
    num_bytes: int
        The number of bytes to allocate.

    stream: Optional[Union[Stream, int]]
        The stream to use for the allocation. If None, the current stream is used.

    Returns
    -------
    addr: int
        The address of the allocated memory.
    """
    if stream is None:
        stream = current_stream()
    error, ret = hip_malloc_async(num_bytes, int(stream))
    assert error == 0, error_msg("malloc_async", error)
    return ret


def free_async(addr: int, stream: Optional[Union[Stream, int]] = None) -> None:
    """
    Free memory on the current hip device asynchronously.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc` or
        :func:`malloc_async`.

    stream: Union[Stream, int], optional
        The stream to use for the free. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    error = hip_free_async(addr, int(stream))
    assert error == 0, error_msg("free_async", error)


def malloc_host(num_bytes: int) -> int:
    """
    Allocate pinned host memory.

    Parameters
    ----------
    num_bytes: int
        The number of bytes to allocate.

    Returns
    -------
    addr: int
        The address of the allocated memory.
    """
    error, ret = hip_malloc_host(num_bytes)
    assert error == 0, error_msg("malloc_host", error)
    return ret


def free_host(addr: int) -> None:
    """
    Free pinned host memory.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc_host`.
    """
    error = hip_free_host(addr)
    assert error == 0, error_msg("free_host", error)


def memset(addr: int, value: int, num_bytes: int) -> None:
    """
    Set the gpu memory to a given value.

    Parameters
    ----------
    addr: int
        The start address of the memory region to set.

    value: int
        The byte value to set the memory region to.

    num_bytes: int
        The number of bytes to set.
    """
    error = hip_memset(addr, value, num_bytes)
    assert error == 0, error_msg("memset", error)


def memset_async(addr: int, value: int, num_bytes: int, stream: Optional[Union[Stream, int]] = None) -> None:
    """
    Set the gpu memory to given value asynchronously.

    Parameters
    ----------
    addr: int
        The start address of the memory region to set.

    value: int
        The byte value to set the memory region to.

    num_bytes: int
        The number of bytes to set.

    stream: Union[Stream, int], optional
        The stream to use for the memset. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    error = hip_memset_async(addr, value, num_bytes, int(stream))
    assert error == 0, error_msg("memset_async", error)


def memcpy_host_to_device(dst: int, src: int, num_bytes: int) -> None:
    """
    Copy data from host memory into device memory.

    Parameters
    ----------
    dst: int
        The destination (device) address.

    src: int
        The source (host) address.

    num_bytes: int
        The number of bytes to copy.
    """
    error = hip_memcpy_host_to_device(dst, src, num_bytes)
    assert error == 0, error_msg("memcpy_host_to_device", error)


def memcpy_device_to_host(dst: int, src: int, num_bytes: int) -> None:
    """
    Copy data from host memory into device memory.

    Parameters
    ----------
    dst: int
        The destination (host) address.

    src: int
        The source (device) address.

    num_bytes: int
        The number of bytes to copy.
    """
    error = hip_memcpy_device_to_host(dst, src, num_bytes)
    assert error == 0, error_msg("memcpy_device_to_host", error)


def memcpy(dst: int, src: int, num_bytes: int) -> None:
    """
    Copy data from src to dst. Uses hipMemcpyDefault to automatically determine the
        type of copy (DtoH, HtoD, DtoD, or HtoH).

    Parameters
    ----------
    dst: int
        The destination (host) address.

    src: int
        The source (device) address.

    num_bytes: int
        The number of bytes to copy.
    """

    error = hip_memcpy(dst, src, num_bytes)
    assert error == 0, error_msg("memcpy", error)


def memcpy_async(dst: int, src: int, num_bytes: int, stream: Optional[Union[Stream, int]] = None) -> None:
    """
    Copy gpu memory from one location to another asynchronously.

    Parameters
    ----------
    dst: int
        The destination address.

    src: int
        The source address.

    num_bytes: int
        The number of bytes to copy.

    stream: Union[Stream, int], optional
        The stream to use for the memcpy. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    error = hip_memcpy_async(dst, src, num_bytes, int(stream))
    assert error == 0, error_msg("memcpy_async", error)
