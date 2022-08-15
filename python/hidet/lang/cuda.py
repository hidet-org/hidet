from collections import namedtuple
from hidet.ir.dialects.lowlevel import PointerType
from hidet.ir.primitives.cuda import thread_idx, block_idx, block_dim, grid_dim, syncthreads
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync, ldmatrix
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.cp_async import cp_async, cp_async_commit_group, cp_async_wait_group, cp_async_wait_all

dim3 = namedtuple('dim3', field_names=['x', 'y', 'z'])

threadIdx = dim3(thread_idx('x'), thread_idx('y'), thread_idx('z'))
blockIdx = dim3(block_idx('x'), block_idx('y'), block_idx('z'))
blockDim = dim3(block_dim('x'), block_dim('y'), block_dim('z'))
gridDim = dim3(grid_dim('x'), grid_dim('y'), grid_dim('z'))


dyn_smem_storage = PointerType(base_type='uint8', specifiers=['extern', '__shared__'], use_bracket=True)
