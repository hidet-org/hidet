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
from hip import hip
from hip.hip import hipStream_t
from .stream import Stream, current_stream


def memory_info() -> Tuple[int, int]:
    """
    Get the free and total memory on the current device in bytes.

    Returns
    -------
    (free, total): Tuple[int, int]
        The free and total memory on the current device in bytes.
    """
    err, _free, total = hip.hipMemGetInfo()
    assert err == 0, str(err)
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
    err, ret = hip.hipMalloc(num_bytes)
    assert err == 0, str(err)
    return int(ret)


def free(addr: int) -> None:
    """
    Free memory on the current hip device.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc` or
        :func:`malloc_async`.
    """
    (err,) = hip.hipFree(addr)
    assert err == 0, str(err)


def malloc_async(num_bytes: int, stream: Optional[Union[Stream, hipStream_t, int]] = None) -> int:
    """
    Allocate memory on the current hip device asynchronously.

    Parameters
    ----------
    num_bytes: int
        The number of bytes to allocate.

    stream: Optional[Union[Stream, hipStream_t, int]]
        The stream to use for the allocation. If None, the current stream is used.

    Returns
    -------
    addr: int
        The address of the allocated memory.
    """
    if stream is None:
        stream = current_stream()
    err, ret = hip.hipMallocAsync(num_bytes, int(stream))
    assert err == 0, str(err)
    return int(ret)


def free_async(addr: int, stream: Optional[Union[Stream, hipStream_t, int]] = None) -> None:
    """
    Free memory on the current hip device asynchronously.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc` or
        :func:`malloc_async`.

    stream: Union[Stream, hipStream_t, int], optional
        The stream to use for the free. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    (err,) = hip.hipFreeAsync(addr, int(stream))
    assert err == 0, str(err)


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
    err, ret = hip.hipMallocHost(num_bytes)
    assert err == 0, str(err)
    return int(ret)


def free_host(addr: int) -> None:
    """
    Free pinned host memory.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc_host`.
    """
    (err,) = hip.hipFreeHost(addr)
    assert err == 0, str(err)


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
    (err,) = hip.hipmMemset(addr, value, num_bytes)
    assert err == 0, str(err)


def memset_async(
    addr: int, value: int, num_bytes: int, stream: Optional[Union[Stream, hipStream_t, int]] = None
) -> None:
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

    stream: Union[Stream, hipStream_t, int], optional
        The stream to use for the memset. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    (err,) = hip.hipMemsetAsync(addr, value, num_bytes, int(stream))
    assert err == 0, str(err)


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

    (err,) = hip.hipMemcpy(dst, src, num_bytes, hip.hipMemcpyKind.hipMemcpyDefault)
    assert err == 0, str(err)


def memcpy_async(dst: int, src: int, num_bytes: int, stream: Optional[Union[Stream, hipStream_t, int]] = None) -> None:
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

    stream: Union[Stream, hipStream_t, int], optional
        The stream to use for the memcpy. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    (err,) = hip.hipMemcpyAsync(dst, src, num_bytes, hip.hipMemcpyKind.hipMemcpyDefault, int(stream))
    assert err == 0, str(err)
