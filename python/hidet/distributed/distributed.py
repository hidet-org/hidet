from typing import Optional
from datetime import timedelta
from .store import Store, FileStore
from .group import create_nccl_group, ProcessGroup

import hidet
from hidet.graph import Tensor
from hidet.cuda.nccl import nccl_available


DEFAULT_TIMEOUT = timedelta(seconds=1800)

DEFAULT_GROUP = None

def init_process_group(
    backend: str = 'nccl',
    init_method: Optional[str] = None,
    store: Optional[Store] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    world_size: int = -1,
    rank: int = -1,
):
    """
    We ues the same api as PyTorch.
    Currently we only support FileStore. There are two ways to initialize via FileStore.
        1. Manually create a FileStore object and pass it as ``store``;
        2. Specify ``init_method`` with ``files://path-to-file```
    Now world_size and rank still need to be specified manually.
    """
    global DEFAULT_GROUP

    if world_size <= 0 or rank < 0:
        raise RuntimeError("'world_size' and 'rank' must be specified.")
    
    if rank >= world_size:
        raise RuntimeError("'rank' must be smaller than 'world_size'")

    if store is None:
        if init_method is None:
            raise RuntimeError("One of 'init_method' and 'store' must be specified.")
        else:
            if not init_method.startswith('file://'):
                raise RuntimeError("Currently only FileStore is supported. Please speficy the path to the filestore with 'file://path-to-file'")
            path_to_file = init_method[len('file://'):]
            store = FileStore(path_to_file)
    else:
        if init_method is not None:
            raise RuntimeError("'init_method' and 'store' are mutually exclusive.")
        
    store.set_timeout(timeout)
    if backend == 'nccl':
        if not is_nccl_available():
            raise RuntimeError("NCCL is not found.")
        DEFAULT_GROUP = create_nccl_group(store, world_size, rank)

def is_initialized():
    return DEFAULT_GROUP is not None

def is_nccl_available():
    return nccl_available()

def broadcast():
    raise NotImplementedError()

def all_reduce(tensor: Tensor, op:str, group:Optional[ProcessGroup]=None):
    if group is None:
        group = DEFAULT_GROUP
    group.all_reduce(tensor, op)

def reduce():
    raise NotImplementedError()

def all_gather_into_tensor():
    raise NotImplementedError()

def scatter():
    raise NotImplementedError()

def reduce_scatter_tensor():
    raise NotImplementedError()

if __name__ == '__main__':
    init_process_group(init_method='file://tmp', world_size=1, rank=0)
    print(is_initialized())
    test = hidet.randn((2, 2), device='cuda')
    print(test)
    all_reduce(test, 'sum')
    print(test)