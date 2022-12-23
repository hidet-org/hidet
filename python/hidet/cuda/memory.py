# pylint: disable=no-name-in-module, c-extension-no-member
from typing import Tuple, Union
from cuda import cudart
from cuda.cudart import cudaStream_t, cudaMemcpyKind
from .stream import Stream


def memory_info() -> Tuple[int, int]:
    err, free_bytes, total_bytes = cudart.cudaMemGetInfo()
    assert err == 0
    return free_bytes, total_bytes


def malloc(num_bytes: int) -> int:
    err, addr = cudart.cudaMalloc(num_bytes)
    assert err == 0
    return addr


def free(addr: int) -> None:
    err = cudart.cudaFree(addr)
    assert err == 0


def malloc_async(num_bytes: int, stream: Union[Stream, cudaStream_t, int]) -> int:
    err, addr = cudart.cudaMallocAsync(num_bytes, int(stream))
    assert err == 0
    return addr


def free_async(addr: int, stream: Union[Stream, cudaStream_t, int]) -> None:
    err = cudart.cudaFreeAsync(addr, int(stream))
    assert err == 0


def malloc_host(num_bytes: int) -> int:
    err, addr = cudart.cudaMallocHost(num_bytes)
    assert err == 0
    return addr


def free_host(addr: int) -> None:
    err = cudart.cudaFreeHost(addr)
    assert err == 0


def memset(addr: int, value: int, num_bytes: int) -> None:
    err = cudart.cudaMemset(addr, value, num_bytes)
    assert err == 0


def memset_async(addr: int, value: int, num_bytes: int, stream: Union[Stream, cudaStream_t, int]) -> None:
    err = cudart.cudaMemsetAsync(addr, value, num_bytes, int(stream))
    assert err == 0


def memcpy(dst: int, src: int, num_bytes: int, kind: cudaMemcpyKind) -> None:
    err = cudart.cudaMemcpy(dst, src, num_bytes, kind)
    assert err == 0


def memcpy_async(
    dst: int, src: int, num_bytes: int, kind: cudaMemcpyKind, stream: Union[Stream, cudaStream_t, int]
) -> None:
    err = cudart.cudaMemcpyAsync(dst, src, num_bytes, kind, int(stream))
    assert err == 0
