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
from typing import Union, Tuple
from hidet.ir.expr import Expr


Int = Union[Expr, int]
Dim3 = Union[Int, Tuple[Int, Int], Tuple[Int, Int, Int]]

# The grid dimension of a cuda kernel, specifying the number of thread blocks
grid_dim: Dim3 = 1

# The block dimension of a cuda kernel, specifying the number of threads per block
block_dim: Dim3 = 1

# A hint to nvcc compiler the minimal number of thread blocks should be executed on
# the same streaming processor (SM). This attribute will influence the register allocation
# strategy adopted by nvcc.
min_blocks: int = 1

# The size of dynamic shared memory allocated to the cuda kernel.
dynamic_smem_bytes: Int = 0
