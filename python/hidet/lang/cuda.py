from collections import namedtuple
from hidet.ir.primitives.cuda import thread_idx, block_idx, block_dim, grid_dim, syncthreads
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync

dim3 = namedtuple('dim3', field_names=['x', 'y', 'z'])

threadIdx = dim3(thread_idx('x'), thread_idx('y'), thread_idx('z'))
blockIdx = dim3(block_idx('x'), block_idx('y'), block_idx('z'))
blockDim = dim3(block_dim('x'), block_dim('y'), block_dim('z'))
gridDim = dim3(grid_dim('x'), grid_dim('y'), grid_dim('z'))
