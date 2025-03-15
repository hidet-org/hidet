from hidet.ir.primitives.cuda.vars import threadIdx, blockIdx, blockDim, gridDim
from hidet.ir.primitives.cuda.cluster import this_cluster
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory, set_kernel_max_dynamic_smem_bytes
from hidet.ir.primitives.cuda.sync import syncthreads, syncthreads_and, syncthreads_count, syncthreads_or, syncwarp
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync, ldmatrix
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.cp_async import cp_async, cp_async_commit_group, cp_async_wait_group, cp_async_wait_all
from hidet.ir.primitives.cuda.ldst import load, store
from hidet.ir.primitives.cuda.time import nano_sleep
from hidet.ir.primitives.cuda.memcpy import memcpy_async, memcpy
from hidet.ir.primitives.cuda.atomic import atomic_add, atomic_sub, atomic_min, atomic_max, atomic_exchange, atomic_cas
from hidet.ir.primitives.cuda.fastintdiv import fast_intdiv, fast_intmod
from hidet.ir.primitives.cuda.shfl import shfl_sync, shfl_up_sync, shfl_xor_sync, shfl_down_sync
from hidet.ir.primitives.cuda.mutex import acquire_lock, release_lock, acquire_seq_semaphore, release_seq_semaphore
from hidet.ir.primitives.cuda.wgmma import (
    WgmmaConfig,
    wgmma_async,
    make_wgmma_desc,
    wgmma_wait_group,
    wgmma_commit_group,
    wgmma_fence,
)
from hidet.ir.library.cuda import cublas
from hidet.lang.constructs.declare import register_tensor, shared_tensor
from hidet.ir.primitives.cuda.setmaxnreg import setmaxnreg

from . import contexts
