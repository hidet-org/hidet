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
from typing import Sequence, Optional, Union, List, Tuple

import hidet
from hidet.ir.compute import ReduceOperation
from hidet.graph import Tensor


class TensorShardSpec:
    """
    For disambiguation, we adopt a convention that 'dim' refers to the dimension or axis of the
    tensor, and 'mesh axis' refers to the axis of hardware hierarchy.

    The constructor can be called by specifying the dimension sharded by each mesh axis. The
    parameter `sharded_dims` is a list of integers, and the i-th element of it is the dimension
    sharded by the i-th mesh axis. If any mesh axis does not shard any dimension (tensor are
    duplicated along that mesh axis), its corresponding value is None.  We assume that each mesh
    axis can shard at most one dimension, and each dimension can be sharded by multiple mesh axes.

    It also accept a single-value parameter, which will be viewed as a single-layer mesh hierarchy.

    None or negative numbers will be treated as not to shard on this dimension
    """

    def __init__(self, ndim: int, sharded_dim: Union[Optional[int], Sequence[Optional[int]]] = None):
        self.ndim: int = ndim
        if not isinstance(sharded_dim, Sequence):
            sharded_dim = [sharded_dim]
        for i in range(len(sharded_dim)):
            if sharded_dim[i] is not None and sharded_dim[i] < 0:
                sharded_dim[i] = None
        self._sharded_dim: Sequence[Optional[int]] = sharded_dim
        self.num_mesh_axis: int = len(sharded_dim)

        # Sanity check and build mesh_axes to dim map
        self.mesh_axes_per_dim: List[List[int]] = [[] for _ in range(self.ndim)]
        for mesh_axis, dim in enumerate(sharded_dim):
            if dim is not None:
                if dim >= self.ndim:
                    raise ValueError("Sharded dim should be less than the number of dimensions")
                self.mesh_axes_per_dim[dim].append(mesh_axis)

    def is_single_layer_hierarchy(self) -> bool:
        return self.num_mesh_axis

    def sharded_dim(self) -> int:
        if not self.is_single_layer_hierarchy():
            raise ValueError("Only single layer hierarchy supports directly querying the sharded axis.")
        return self._sharded_dim[0]

    def __str__(self) -> str:
        str_per_dim = [
            'R' if len(axes) == 0 else ''.join([f'S{axis}' for axis in axes]) for axes in self.mesh_axes_per_dim
        ]
        return f"({', '.join(str_per_dim)})"

    def is_full(self) -> bool:
        return all((len(mesh_axes) == 0 for mesh_axes in self.mesh_axes_per_dim))

    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorShardSpec):
            return False
        if self.ndim != other.ndim:
            return False
        return self._sharded_dim == other._sharded_dim


def get_tile(tensor, nshards, shard_dim, rank=0):
    # Only works for 1-D partition
    slice_list = []
    for i in range(len(tensor.shape)):
        if i != shard_dim:
            slice_list.append(slice(tensor.shape[i]))
        else:
            shard_size = tensor.shape[i] // nshards
            slice_list.append(slice(shard_size * rank, shard_size * (rank + 1)))
    return tensor[slice_list]


class ReduceFunction:
    def __init__(self, op: ReduceOperation):
        self._op = op

    @property
    def op(self) -> str:
        return str(self._op)

    def __call__(self, tensor: Tensor, num_shards: int, rank: int) -> Tensor:
        raise NotImplementedError()

    def cost(self, tensor: Tensor) -> int:
        raise NotImplementedError()


class AllReduce(ReduceFunction):
    def __str__(self):
        return "all_reduce_" + str(self.op)

    def __call__(self, tensor: Tensor, num_shards: int) -> Tensor:
        return hidet.ops.distributed.all_reduce(tensor, self.op)

    def cost(self, tensor: Tensor) -> int:
        return 2 * tensor.nbytes


class ReduceScatter(ReduceFunction):
    def __str__(self):
        return "reduce_scatter_" + str(self.op)

    def __call__(self, tensor: Tensor, num_shards: int) -> Tensor:
        tensor = tensor.reshape((num_shards, tensor.shape[0] // num_shards) + tensor.shape[1:])
        return hidet.ops.distributed.reduce_scatter(tensor, self.op)

    def cost(self, tensor: Tensor) -> int:
        return tensor.nbytes


class OpShardSpec:
    def __init__(
        self,
        input_specs: Sequence[TensorShardSpec],
        output_specs: Sequence[TensorShardSpec],
        reduce_fn: Optional[List[ReduceFunction]] = None,
    ):
        self.input_specs = input_specs
        self.output_specs = output_specs
        if reduce_fn is None:
            reduce_fn = [None] * len(output_specs)
        assert len(reduce_fn) == len(output_specs)
        self.reduce_fn = reduce_fn

    def __str__(self) -> str:
        in_str = ', '.join(map(str, self.input_specs))
        o_list = []
        for o, r in zip(self.output_specs, self.reduce_fn):
            s = str(o)
            if r is not None:
                s += f'[{str(r)}]'
            o_list.append(s)
        out_str = ', '.join(o_list)
        return in_str + " -> " + out_str


def node_comm_cost(output_tensors, spec) -> int:
    assert len(output_tensors) == len(spec.output_specs)
    cost = 0
    for tensor, reduce_fn in zip(output_tensors, spec.reduce_fn):
        if reduce_fn is not None:
            cost += reduce_fn.cost(tensor)
    return cost


def _gather(tensor, num_shards, shard_dim) -> int:
    gather_input = hidet.ops.distributed.all_gather(tensor, num_shards)
    assert gather_input.size == tensor.size * num_shards
    updated_input = gather_input.rearrange(
        [[d] for d in range(1, shard_dim + 1)]
        + [[0, shard_dim + 1]]
        + [[d] for d in range(shard_dim + 2, len(gather_input.shape))]
    )
    return updated_input


class ReshardFunction:
    def __init__(self, produce_spec: TensorShardSpec, consume_spec: TensorShardSpec):
        self.produce_spec = produce_spec
        self.consume_spec = consume_spec

    def __call__(self, tensor: Tensor, num_shards: int, rank: int):
        raise NotImplementedError()


class ReshardSlice(ReshardFunction):
    def __str__(self):
        return 'reshard_slice'

    def __call__(self, tensor: Tensor, num_shards: int, rank: int):
        shard_dim = self.consume_spec.sharded_dim()
        assert self.produce_spec.is_full()
        return get_tile(tensor, num_shards, shard_dim, rank)


class ReshardGather(ReshardFunction):
    def __str__(self):
        return 'reshard_gather'

    def __call__(self, tensor: Tensor, num_shards: int, rank: int):
        shard_dim = self.produce_spec.sharded_dim()
        return _gather(tensor, num_shards, shard_dim)


class ReshardGatherSlice(ReshardFunction):
    def __str__(self):
        return 'reshard_gather_slice'

    def __call__(self, tensor: Tensor, num_shards: int, rank: int):
        shard_dim = self.produce_spec.sharded_dim()
        gathered_tensor = _gather(tensor, num_shards, shard_dim)
        return get_tile(gathered_tensor, num_shards, self.consume_spec.sharded_dim(), rank)


def connect(tensor: Tensor, produce_spec: TensorShardSpec, consume_spec: TensorShardSpec) -> Tuple[ReduceFunction, int]:
    # Only supports 1D partition now
    if produce_spec == consume_spec:
        return (None, 0)
    if produce_spec.is_full():
        return (ReshardSlice(produce_spec, consume_spec), 0)
    if consume_spec.is_full():
        return (ReshardGather(produce_spec, consume_spec), tensor.nbytes)
    else:
        return (ReshardGatherSlice(produce_spec, consume_spec), tensor.nbytes)
