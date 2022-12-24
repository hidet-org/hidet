# pylint: disable=no-name-in-module, c-extension-no-member
from typing import Tuple, Union, Optional
from cuda import cudart
from cuda.cudart import cudaStream_t, cudaMemcpyKind
from .stream import Stream, current_stream


def memory_info() -> Tuple[int, int]:
    err, free_bytes, total_bytes = cudart.cudaMemGetInfo()
    assert err == 0, err
    return free_bytes, total_bytes


def malloc(num_bytes: int) -> int:
    err, addr = cudart.cudaMalloc(num_bytes)
    assert err == 0, err
    return addr


def free(addr: int) -> None:
    (err,) = cudart.cudaFree(addr)
    assert err == 0, err


def malloc_async(num_bytes: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None) -> int:
    if stream is None:
        stream = current_stream()
    err, addr = cudart.cudaMallocAsync(num_bytes, int(stream))
    assert err == 0, err
    return addr


def free_async(addr: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None) -> None:
    if stream is None:
        stream = current_stream()
    (err,) = cudart.cudaFreeAsync(addr, int(stream))
    assert err == 0, err


def malloc_host(num_bytes: int) -> int:
    err, addr = cudart.cudaMallocHost(num_bytes)
    assert err == 0, err
    return addr


def free_host(addr: int) -> None:
    (err,) = cudart.cudaFreeHost(addr)
    assert err == 0, err


def memset(addr: int, value: int, num_bytes: int) -> None:
    (err,) = cudart.cudaMemset(addr, value, num_bytes)
    assert err == 0, err


def memset_async(
    addr: int, value: int, num_bytes: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None
) -> None:
    if stream is None:
        stream = current_stream()
    (err,) = cudart.cudaMemsetAsync(addr, value, num_bytes, int(stream))
    assert err == 0, err


def memcpy(dst: int, src: int, num_bytes: int) -> None:
    (err,) = cudart.cudaMemcpy(dst, src, num_bytes, cudaMemcpyKind.cudaMemcpyDefault)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy failed with error code {err.name}")


def memcpy_async(dst: int, src: int, num_bytes: int, stream: Optional[Union[Stream, cudaStream_t, int]] = None) -> None:
    if stream is None:
        stream = current_stream()
    (err,) = cudart.cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyKind.cudaMemcpyDefault, int(stream))
    assert err == 0, err
