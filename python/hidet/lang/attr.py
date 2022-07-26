from typing import Optional, Union, Tuple
from hidet.ir import Expr

Int = Union[Expr, int]
Dim3 = Union[Int, Tuple[Int, Int], Tuple[Int, Int, Int]]

"""
general attributes
"""
# The label of the scope
label: Optional[str] = None

"""
function attributes
"""
# The grid dimension of a cuda kernel, specifying the number of thread blocks
cuda_grid_dim: Dim3 = 1

# The block dimension of a cuda kernel, specifying the number of threads per block
cuda_block_dim: Dim3 = 1

# A hint to nvcc compiler the minimal number of thread blocks should be executed on
# the same streaming processor (SM). This attribute will influence the register allocation
# strategy adopted by nvcc.
cuda_min_blocks: int = 1

# The size of dynamic shared memory allocated to the cuda kernel.
cuda_dyn_smem_bytes: Int = 0
