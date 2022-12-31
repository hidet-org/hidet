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
from cuda import cudart
from cuda.cudart import cudaStream_t, cudaMemcpyKind
from .stream import Stream, current_stream


def memory_info() -> Tuple[int, int]:
    """
    Get the free and total memory on the current device in bytes.

    Returns
    -------
    (free, total): Tuple[int, int]
        The free and total memory on the current device in bytes.
    """
    err, free_bytes, total_bytes = cudart.cudaMemGetInfo()
    assert err == 0, err
    return free_bytes, total_bytes


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
    err, addr = cudart.cudaMalloc(num_bytes)
    assert err == 0, err
    return addr


def free(addr: int) -> None:
    """
    Free memory on the current cuda device.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc` or
        :func:`malloc_async`.
    """
    (err,) = cudart.cudaFree(addr)
    assert err == 0, err


def malloc_async(num_bytes: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None) -> int:
    """
    Allocate memory on the current device asynchronously.

    Parameters
    ----------
    num_bytes: int
        The number of bytes to allocate.

    stream: Optional[Union[Stream, cudaStream_t, int]]
        The stream to use for the allocation. If None, the current stream is used.

    Returns
    -------
    addr: int
        The address of the allocated memory.
    """
    if stream is None:
        stream = current_stream()
    err, addr = cudart.cudaMallocAsync(num_bytes, int(stream))
    assert err == 0, err
    return addr


def free_async(addr: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None) -> None:
    """
    Free memory on the current cuda device asynchronously.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc` or
        :func:`malloc_async`.

    stream: Union[Stream, cudaStream_t, int], optional
        The stream to use for the free. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    (err,) = cudart.cudaFreeAsync(addr, int(stream))
    assert err == 0, err


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
    err, addr = cudart.cudaMallocHost(num_bytes)
    assert err == 0, err
    return addr


def free_host(addr: int) -> None:
    """
    Free pinned host memory.

    Parameters
    ----------
    addr: int
        The address of the memory to free. This must be the address of memory allocated with :func:`malloc_host`.
    """
    (err,) = cudart.cudaFreeHost(addr)
    assert err == 0, err


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
    (err,) = cudart.cudaMemset(addr, value, num_bytes)
    assert err == 0, err


def memset_async(
    addr: int, value: int, num_bytes: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None
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

    stream: Union[Stream, cudaStream_t, int], optional
        The stream to use for the memset. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    (err,) = cudart.cudaMemsetAsync(addr, value, num_bytes, int(stream))
    assert err == 0, err


def memcpy(dst: int, src: int, num_bytes: int) -> None:
    """
    Copy gpu memory from one location to another.

    Parameters
    ----------
    dst: int
        The destination address.

    src: int
        The source address.

    num_bytes: int
        The number of bytes to copy.
    """
    (err,) = cudart.cudaMemcpy(dst, src, num_bytes, cudaMemcpyKind.cudaMemcpyDefault)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy failed with error code {err.name}")


def memcpy_async(dst: int, src: int, num_bytes: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None) -> None:
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

    stream: Union[Stream, cudaStream_t, int], optional
        The stream to use for the memcpy. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    (err,) = cudart.cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyKind.cudaMemcpyDefault, int(stream))
    assert err == 0, err


def memcpy_peer(dst: int, dst_id: int, src: int, src_id: int, num_bytes: int) -> None:
    """
    Copy gpu memory from one device to another.

    Parameters
    ----------
    dst: int
        The destination address.

    dst_id: int
        The id of the destination device.

    src: int
        The source address.

    src_id: int
        The id of the source device.

    num_bytes: int
        The number of bytes to copy.
    """
    (err,) = cudart.cudaMemcpyPeer(dst, dst_id, src, src_id, num_bytes)
    assert err == 0, err


def memcpy_peer_async(
    dst: int,
    dst_id: int,
    src: int,
    src_id: int,
    num_bytes: int,
    stream: Optional[Union[Stream, cudaStream_t, int]] = None,
) -> None:
    """
    Copy gpu memory from one device to another.

    Parameters
    ----------
    dst: int
        The destination address.

    dst_id: int
        The id of the destination device.

    src: int
        The source address.

    src_id: int
        The id of the source device.

    num_bytes: int
        The number of bytes to copy.

    stream: Union[Stream, cudaStream_t, int], optional
        The stream to use for the memcpy. If None, the current stream is used.
    """
    if stream is None:
        stream = current_stream()
    (err,) = cudart.cudaMemcpyPeerAsync(dst, dst_id, src, src_id, num_bytes, int(stream))
    assert err == 0, err
