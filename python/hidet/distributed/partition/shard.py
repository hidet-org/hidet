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
from typing import Sequence, Set, Optional, Union, List
from enum import Enum

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
    """
    def __init__(self, ndim: int, sharded_dim: Union[Optional[int], Sequence[Optional[int]]]=None):
        self.ndim: int = ndim
        if not isinstance(sharded_dim, Sequence):
            sharded_dim = [sharded_dim]
        self._sharded_dim: Sequence[Optional[int]] = sharded_dim
        self.num_mesh_axis: int = len(sharded_dim)
        
        # Sanity check and build mesh_axes to dim map
        self.mesh_axes_per_dim: List[List[int]] = [[] for _ in range(self.ndim)]
        for mesh_axis, dim in enumerate(sharded_dim):
            if dim is None:
                continue
            if dim >= self.ndim:
                raise ValueError("Sharded dim should be less than the number of dimensions")
            self.mesh_axes_per_dim[dim].append(mesh_axis) 

    def is_single_layer_hierarchy(self) -> bool:
        return self.num_mesh_axis

    def sharded_dim(self) -> int:
        if not self.is_single_layer_hierarchy():
            raise ValueError("Only single layer hierarchy supports directly querying the sharded axis.")
        return self.sharded_dim[0]
        
    def __str__(self) -> str:
        str_per_dim = ['R' if len(axes) == 0 else ''.join([f'S{axis}' for axis in axes])
                        for axes in self.mesh_axes_per_dim]
        return f"({', '.join(str_per_dim)})"

class ReduceFunction(Enum):
    REDUCE_SCATTER = 1,
    ALL_REDUCE = 2

class OpShardSpec:
    def __init__(self, input_specs: Sequence[TensorShardSpec], output_specs: Sequence[TensorShardSpec], reduce_fn: Optional[ReduceFunction]=None):
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.reduce_fn = reduce_fn

    def __str__(self):
        in_str = ', '.join(map(str, self.input_specs))
        out_str = ', '.join(map(str, self.input_specs))
        return in_str + " -> " + out_str
    
if __name__ == '__main__':
    s = TensorShardSpec(3, sharded_dim=0)
    print(s)
    s = TensorShardSpec(3, mesh_axes_per_dim=[[0], [1]])
    print(s)