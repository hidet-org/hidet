from collections import namedtuple
from hidet.ir.dialects.lowlevel import PointerType
dyn_smem_storage = PointerType(base_type='uint8', specifiers=['extern', '__shared__'], use_bracket=True)

from hidet.ir.primitives.cuda.vars import threadIdx, blockIdx, blockDim, gridDim
from hidet.ir.primitives.cuda.sync import syncthreads, syncthreads_and, syncthreads_count, syncthreads_or
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync, ldmatrix
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda.cp_async import cp_async, cp_async_commit_group, cp_async_wait_group, cp_async_wait_all
from hidet.ir.primitives.cuda.ldst import load, store
from hidet.ir.primitives.cuda.time import nano_sleep
from hidet.ir.primitives.cuda.atomic import atomic_add, atomic_sub, atomic_exchange, atomic_cas
from hidet.ir.primitives.cuda.shfl import shfl_sync, shfl_up_sync, shfl_xor_sync, shfl_down_sync
from hidet.ir.primitives.cuda.mutex import acquire_lock, release_lock
