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
from .group import create_nccl_group, ProcessGroup, set_nccl_comms


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
        set_nccl_comms()
    else:
        raise ValueError(f"Backend {backend} is not supported.")


def is_initialized():
    return DEFAULT_GROUP is not None


def is_nccl_available():
    return nccl_available()


# The runtime API of collective communaction operations is aligned with PyTorch


def broadcast(tensor: Tensor, src: int, group=None):
    """Broadcast the tensor from device 'src' to all devices.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of the tensor is identical on all peers.

    Parameters
    ----------
    tensor: Tensor
        For the sender, the tensor to be broadcasted.
        For other devices, the tensor to store the broadcasted data. It will be updated in-place.

    src: int
        The rank of the device that sends the data.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.broadcast(tensor, src)


def all_reduce(tensor: Tensor, op: str, group: Optional[ProcessGroup] = None):
    """Reduce a tensor across all peers and each device will have the reduced result.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of the tensor is identical on all peers.

    Parameters
    ----------
    tensor: Tensor
        The tensor to be reduced. It will be updated in-place.

    op: str
        The reduction operation, which can be 'sum', 'prod', 'max', 'min', 'avg'.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.all_reduce(tensor, op)


def reduce(tensor: Tensor, dst: int, op: str, group: Optional[ProcessGroup] = None):
    """Reduce a tensor across all peers and store the result on the device 'dst'.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of the tensor is identical on all peers.

    Parameters
    ----------
    tensor: Tensor
        The input tensor to be reduced. The result will be stored in the tensor on the device with rank 'dst'.

    op: str
        The reduction operation, which can be 'sum', 'prod', 'max', 'min', 'avg'.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.reduce(tensor, dst, op)


def all_gather(tensor_list: List[Tensor], tensor: Tensor, group: Optional[ProcessGroup] = None):
    """All devices gather tensors sent from each device.

    The input 'tensor' from the i-th device will be stored in 'tensor_list[i]' on all devices after all_gather.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of 'tensor' from the i-th device
        is the same as tensor_list[i] on all peers.

    Parameters
    ----------
    tensor_list: List[Tensor]
        A list of tensors where the result will be stored.

    tensor: Tensor
        The input tensor to be broadcasted to all devices.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.all_gather(tensor_list, tensor)


def all_gather_into_tensor(output_tensor: Tensor, input_tensor: Tensor, group: Optional[ProcessGroup] = None):
    """All devices gather tensors sent from each device. The results will be stored in a single larger tensor.

    The 'input_tensor' from the i-th device will be stored in 'output_tensor[i]' on all devices after all_gather.
    'output_tensor' is a larger tensor with an extra first dimension of size 'world_size'.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of 'input_tensor' is the same on all devices.
        And 'output_tensor' has the correct shape and dtype on all devices.

    Parameters
    ----------
    output_tensor: Tensor[world_size, dim_1, dim_2, ..., dim_n]
        The output tensor to store the result.

    input_tensor: Tensor[dim_1, dim_2, ..., dim_n]
        The input tensor to be broadcasted to all devices.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.all_gather_into_tensor(output_tensor, input_tensor)


def gather(
    tensor: Tensor, gather_list: Optional[List[Tensor]] = None, dst: int = 0, group: Optional[ProcessGroup] = None
):
    """Gather the tensor sent from each device.

    The input 'tensor' from the i-th device will be stored in 'tensor_list[i]' on the device dst.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of 'tensor' from the i-th device
        is the same as tensor_list[i] on the device dst.

    Parameters
    ----------
    tensor: Tensor
        The input tensor to be broadcasted to all devices.

    gather_list: Optional[List[Tensor]]
        On the device dst, a list of tensors where the result will be stored.
        On other devices, it can be set as None.

    dst: int
        The rank of the device that gathers tensors

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.gather(tensor, gather_list, dst)


def scatter(
    tensor: Tensor, scatter_list: Optional[List[Tensor]] = None, src: int = 0, group: Optional[ProcessGroup] = None
):
    """Scatter a list of tensors from the device 'src' to all devices .

    scatter_list[i] from the device 'src' will be stored in 'tensor' on the i-th device.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of scatter_list[i] from
        the device 'src' is the same as tensor on the i-th device.

    Parameters
    ----------
    tensor: Tensor
        The tensor to store the received data.

    scatter_list: Optional[List[Tensor]]
        On the device src, a list of tensors to be scattered.
        On other devices, it can be set as None.

    src: int
        The rank of the device that sends data.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.scatter(tensor, scatter_list, src)


def reduce_scatter(output: Tensor, input_list: List[Tensor], op: str, group: Optional[ProcessGroup] = None):
    """Reduce each tensor in a list across all devices and store the result of each reduction on one device

    input_list[i] from all devices will be reduced and stored in the 'output' tensor on the i-th device.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of input_list[i] from all devices is the same,
        and also same as the 'output' tensor on the i-th device.

    Parameters
    ----------
    output: Tensor
        The tensor to store the reduction result.

    input_list: List[Tensor]
        The input tensors to be reduced.

    op: str
        The reduction operation, which can be 'sum', 'prod', 'max', 'min', 'avg'.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.reduce_scatter(output, input_list, op)


def reduce_scatter_tensor(output: Tensor, input: Tensor, op: str, group: Optional[ProcessGroup] = None):
    """Reduce the input tensor across all devices and store a fraction of the result on each device

    input[i] from all devices will be reduced and stored in the 'output' tensor on the i-th device.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of input_list from all devices is the same,
        and the 'output' tensor has the correct shape and dtype to store the reduction result.

    Parameters
    ----------
    output: Tensor[dim_1, dim_2, ..., dim_n]
        The tensor to store a fraction of the reduction result.

    input: Tensor[world_size, dim_1, dim_2, ..., dim_n]
        The input tensor to be reduced.

    op: str
        The reduction operation, which can be 'sum', 'prod', 'max', 'min', 'avg'.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.reduce_scatter_tensor(output, input, op)


def barrier(group: Optional[ProcessGroup] = None):
    """Synchonize all devices.

    Parameters
    ----------
    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.barrier()


def send(tensor: Tensor, dst: int, group: Optional[ProcessGroup] = None):
    """Send a tensor to the device dst.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of tensor is the same on the sender and the receiver.

    Parameters
    ----------
    tensor: Tensor
        The tensor to be sent.

    dst: int
        Rank of the device that data is sent to.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.send(tensor, dst)


def recv(tensor: Tensor, src: int, group: Optional[ProcessGroup] = None):
    """Receive a tensor from the device src.

    .. tip::

        The caller should make sure the metadata (shape, dtype) of tensor is the same on the sender and the receiver.

    Parameters
    ----------
    tensor: Tensor
        The tensor to store the received data.

    src: int
        Rank of the device that data is sent from.

    group: Optional[ProcessGroup]
        The process group to work on. If None, the default process group will be used.
    """
    if group is None:
        group = DEFAULT_GROUP
    group.recv(tensor, src)
