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
from typing import Sequence, Set
from enum import Enum

class TensorShardSpec:
    """
    For disambiguation, we adopt a convention that 'dim' refers to the dimension or axis of 
    the tensor, and 'mesh axis' refers to the axis of hardware hierarchy.

    The constructor can be called either by specifying the mesh axes of each dimension, or 
    specifying the sharded dimension when there is only one mesh aixs.
    """
    def __init__(self, ndim: int, sharded_dim: int=None, mesh_axes_per_dim: Sequence[Set[int]]=None):
        self.ndim = ndim

        if mesh_axes_per_dim is not None:
            if sharded_dim is not None:
                raise ValueError("Only one of sharded_dim or mesh_axes_per_dim need to be specified.")
            assert isinstance(mesh_axes_per_dim, Sequence)
            self.mesh_axes_per_dim = mesh_axes_per_dim
        else:
            assert isinstance(sharded_dim, int)
            self.num_mesh_axis = 1
            self.mesh_axes_per_dim = [{0} if i == sharded_dim else set() for i in range(ndim)]
        # Sanity check

    def is_single_layer_hierarchy(self) -> bool:
        return self.num_mesh_axis

    def sharded_dim(self) -> int:
        if not self.is_single_layer_hierarchy():
            raise ValueError("Only single layer hierarchy supports directly querying the sharded axis.")
        
    def __str__(self) -> str:
        str_per_dim = ['R' if len(axes) == 0 else ''.join([f'S{axis}' for axis in axes])
                        for axes in self.mesh_axes_per_dim]
        return f"({', '.join(str_per_dim)})"

class ReduceFunction(Enum):
    REDUCE_SCATTER = 1,
    ALL_REDUCE = 2

class OpShardSpec:
    def __init__(self, input_specs: Sequence[TensorShardSpec], output_specs: Sequence[TensorShardSpec], reduce_fn: ReduceFunction):
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.reduce_fn = reduce_fn

    

if __name__ == '__main__':
    s = TensorShardSpec(3, sharded_dim=0)
    print(s)
    s = TensorShardSpec(3, mesh_axes_per_dim=[[0], [1]])
    print(s)