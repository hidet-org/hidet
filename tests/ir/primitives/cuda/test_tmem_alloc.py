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

from hidet.ir.primitives.cuda.sync import syncthreads
import pytest

import hidet
from hidet.ir.expr import Constant
from hidet.ir.primitives import blockIdx
from hidet.ir.primitives.cuda import threadIdx
from hidet.ir.primitives.cuda.tmem import tcgen05_alloc, tcgen05_dealloc, tcgen05_relinquish_alloc_permit
from hidet.ir.primitives.debug import printf
from hidet.ir.dtypes import i32, u32
from hidet.lang import attrs, script
from hidet.lang.constructs.declare import shared_tensor


@pytest.mark.requires_cuda_blackwell
def test_tmem_basic():
    with hidet.script_module() as script_module:

        @script
        def tmem_basic():
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            # Allocate a shared memory location to store the TMEM address
            tmem_addr = shared_tensor("u32", [1])
            warp_idx = threadIdx.x // 32
            if warp_idx == 0:
                tcgen05_alloc(tmem_addr, 128)
            syncthreads()
            printf("Allocated TMEM address: %u\n", tmem_addr[0])
            if warp_idx == 0:
                tcgen05_dealloc(tmem_addr[0], 128)
            syncthreads()
            printf("Deallocated TMEM address: %u\n", tmem_addr[0])
            if warp_idx == 0:
                tcgen05_relinquish_alloc_permit()

        func = script_module.build()
        func()
        hidet.cuda.synchronize()


@pytest.mark.requires_cuda_blackwell
def test_multiple_tmem_alloc():
    with hidet.script_module() as script_module:

        @script
        def multiple_alloc():
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            # Allocate shared memory locations to store TMEM addresses
            tmem_addrs = shared_tensor("u32", [3])
            warp_idx = threadIdx.x // 32

            # One warp handles TMEM allocation/deallocation
            if warp_idx == 0:
                # Allocate three different sizes of TMEM
                tcgen05_alloc(~tmem_addrs[0], 32)
                tcgen05_alloc(~tmem_addrs[1], 64)
                tcgen05_alloc(~tmem_addrs[2], 128)

            syncthreads()
            printf("CTA %d: TMEM addresses: %u, %u, %u\n", blockIdx.x, tmem_addrs[0], tmem_addrs[1], tmem_addrs[2])

            if warp_idx == 0:
                printf("First TMEM address: %u\n", tmem_addrs[0])
                printf("Second TMEM address: %u\n", tmem_addrs[1])
                printf("Third TMEM address: %u\n", tmem_addrs[2])

                # Deallocate in reverse order
                tcgen05_dealloc(tmem_addrs[2], 128)
                tcgen05_dealloc(tmem_addrs[1], 64)
                tcgen05_dealloc(tmem_addrs[0], 32)

                # Relinquish the allocation permit
                tcgen05_relinquish_alloc_permit()

            syncthreads()

    func = script_module.build()
    func()
    hidet.cuda.synchronize()


@pytest.mark.requires_cuda_blackwell
def test_tmem_cta_pair():
    with hidet.script_module() as script_module:

        @script
        def tmem_cta_pair():
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 2
            attrs.cuda.block_dim = 128
            attrs.cuda.cluster_dim = 2

            tmem_addr = shared_tensor("u32", [1])

            # one warp in each CTA handles TMEM operations
            warp_idx = threadIdx.x // 32
            if warp_idx == 0:
                tcgen05_alloc(tmem_addr, 64, use_cta_pair=True)
                printf("CTA %d: TMEM address: %u\n", blockIdx.x, tmem_addr[0])

            syncthreads()

            if warp_idx == 0:
                tcgen05_dealloc(tmem_addr[0], 64, use_cta_pair=True)
                printf("CTA %d: Deallocated TMEM address: %u\n", blockIdx.x, tmem_addr[0])

    func = script_module.build()
    func()
    hidet.cuda.synchronize()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
