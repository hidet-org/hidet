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
# pylint: disable=unused-import
from typing import Union, Optional, Sequence
from hidet.ir.type import DataType, tensor_type
from hidet.ir.expr import Expr
from hidet.ir.stmt import DeclareScope
from hidet.ir.layout import DataLayout
from hidet.ir.primitives.hip.vars import threadIdx, blockIdx, blockDim, gridDim

# from hidet.ir.primitives.hip.smem import dynamic_shared_memory, set_kernel_max_dynamic_smem_bytes
# from hidet.ir.primitives.hip.sync import syncthreads, syncthreads_and, syncthreads_count, syncthreads_or, syncwarp
# from hidet.ir.primitives.hip.mma import MmaConfig, mma_sync, ldmatrix
# from hidet.ir.primitives.hip.cvta import cvta_generic_to_shared
# from hidet.ir.primitives.hip.cp_async import cp_async, cp_async_commit_group, cp_async_wait_group, cp_async_wait_all
# from hidet.ir.primitives.hip.ldst import load, store
# from hidet.ir.primitives.hip.time import nano_sleep
# from hidet.ir.primitives.hip.atomic import atomic_add, atomic_sub, atomic_min, atomic_max, atomic_exchange, atomic_cas
# from hidet.ir.primitives.hip.shfl import shfl_sync, shfl_up_sync, shfl_xor_sync, shfl_down_sync
# from hidet.ir.primitives.hip.mutex import acquire_lock, release_lock, acquire_seq_semaphore, release_seq_semaphore
from hidet.lang.constructs.declare import register_tensor, shared_tensor
