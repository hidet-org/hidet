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

from typing import Optional, List
from datetime import timedelta

from hidet.graph import Tensor
from hidet.cuda.nccl import nccl_available
from .store import Store, FileStore
from .group import create_nccl_group, ProcessGroup


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
                raise RuntimeError(
                    "Currently only FileStore is supported. "
                    "Please speficy the path to the filestore with 'file://path-to-file'"
                )
            path_to_file = init_method[len('file://') :]
            store = FileStore(path_to_file)
    else:
        if init_method is not None:
            raise RuntimeError("'init_method' and 'store' are mutually exclusive.")

    store.set_timeout(timeout)
    if backend == 'nccl':
        if not is_nccl_available():
            raise RuntimeError("NCCL is not found.")
        DEFAULT_GROUP = create_nccl_group(store, world_size, rank)
    else:
        raise ValueError(f"Backend {backend} is not supported.")


def is_initialized():
    return DEFAULT_GROUP is not None


def is_nccl_available():
    return nccl_available()


# The runtime API of collective communaction operations
# Aligned with PyTorch, but different from Hidet ops


def broadcast(tensor: Tensor, src: int, group=None):
    """
    The caller should make sure the metadata (shape, dtype) of the tensor is aligned with
    the sender.
    """
    # TODO: support group
    if group is None:
        group = DEFAULT_GROUP
    group.broadcast(tensor, src)


def all_reduce(tensor: Tensor, op: str, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.all_reduce(tensor, op)


def reduce(tensor: Tensor, dst: int, op: str, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.reduce(tensor, dst, op)


def all_gather(tensor_list: List[Tensor], tensor: Tensor, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.all_gather(tensor_list, tensor)


def all_gather_into_tensor(output_tensor: Tensor, input_tensor: Tensor, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.all_gather_into_tensor(output_tensor, input_tensor)


def gather(
    tensor: Tensor, gather_list: Optional[List[Tensor]] = None, dst: int = 0, group: Optional[ProcessGroup] = None
):
    if group is None:
        group = DEFAULT_GROUP
    group.gather(tensor, gather_list, dst)


def scatter(
    tensor: Tensor, scatter_list: Optional[List[Tensor]] = None, src: int = 0, group: Optional[ProcessGroup] = None
):
    if group is None:
        group = DEFAULT_GROUP
    group.scatter(tensor, scatter_list, src)


def reduce_scatter(output: Tensor, input_list: List[Tensor], op: str, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.reduce_scatter(output, input_list, op)


def reduce_scatter_tensor(output: Tensor, input: Tensor, op: str, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.reduce_scatter_tensor(output, input, op)


def barrier(group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.barrier()


def send(tensor: Tensor, dst: int, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.send(tensor, dst)


def recv(tensor: Tensor, src: int, group: Optional[ProcessGroup] = None):
    if group is None:
        group = DEFAULT_GROUP
    group.recv(tensor, src)
