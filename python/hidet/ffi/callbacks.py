from typing import Dict
from ctypes import CFUNCTYPE, c_uint64, c_int32
import hidet.cuda
from hidet.ffi.runtime_api import runtime_api


def register_runtime_callback(restype, argtypes):
    decorator = CFUNCTYPE(restype, *argtypes)

    def wrapper(func):
        cfunc = decorator(func)
        runtime_api.register_callback(func.__name__, cfunc)
        return cfunc

    return wrapper


runtime_allocated_storages: Dict[int, 'hidet.runtime.storage.Storage'] = {}


@register_runtime_callback(restype=c_uint64, argtypes=[c_uint64])
def allocate_cuda_storage(nbytes: int) -> int:
    # pylint: disable=import-outside-toplevel
    from hidet.runtime.storage import Storage

    storage = Storage.new('cuda', nbytes)
    runtime_allocated_storages[storage.addr] = storage
    return storage.addr


@register_runtime_callback(restype=None, argtypes=[c_uint64])
def free_cuda_storage(addr: int) -> None:
    if addr == 0:
        return
    if addr not in runtime_allocated_storages:
        raise ValueError('Runtime trying to free a storage that has not been allocated.')
    del runtime_allocated_storages[addr]


@register_runtime_callback(restype=c_uint64, argtypes=[c_uint64])
def allocate_cpu_storage(nbytes: int) -> int:
    # pylint: disable=import-outside-toplevel
    from hidet.runtime.storage import Storage

    storage = Storage.new('cpu', nbytes)
    runtime_allocated_storages[storage.addr] = storage
    return storage.addr


@register_runtime_callback(restype=None, argtypes=[c_uint64])
def free_cpu_storage(addr: int) -> None:
    if addr == 0:
        return
    if addr not in runtime_allocated_storages:
        raise ValueError('Runtime trying to free a storage that has not been allocated.')
    del runtime_allocated_storages[addr]


@register_runtime_callback(restype=None, argtypes=[c_uint64, c_int32, c_uint64])
def cuda_memset(addr: int, value: int, nbytes: int) -> None:
    return hidet.cuda.memset(addr, value, nbytes)
