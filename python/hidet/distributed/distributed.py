from typing import Optional
from datetime import timedelta
from .store import Store, FileStore

DEFAULT_TIMEOUT = timedelta(seconds=1800)

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



def is_initialized():
    pass

def is_nccl_available():
    pass

def broadcast():
    pass

def all_reduce():
    pass

def reduce():
    pass

def all_gather_into_tensor():
    pass

def scatter():
    pass

def reduce_scatter_tensor():
    pass