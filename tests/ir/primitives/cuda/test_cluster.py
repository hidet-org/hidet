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
import pytest

import hidet
from hidet.ir.expr import Constant
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.dtypes import i32
from hidet.lang import attrs, script
from hidet.lang.constructs.declare import as_tensor_pointer
from hidet.lang.cuda import blockDim, gridDim, blockIdx, threadIdx, this_cluster, atomic_add


@pytest.mark.hopper
def test_cluster_histogram():
    # +-------------------------------------------------------------------------------------------------------------------+
    # |                                                 Grid (4096 threads)                                               |
    # | +-----------------------------------------------------+ +-------------------------------------------------------+ |
    # | |                      Cluster (2048)                 | |                       Cluster (2048)                  | |
    # | | +------------------------+ +----------------------+ | | +--------------------------+ +------------------------+ |
    # | | |    Thread Block (1024) | |  Thread Block (1024)   | |     Thread Block (1024)    | | Thread Block (1024)    | |
    # +-------------------------------------------------------------------------------------------------------------------+
    #
    # On H100, max shared mem per block/SM is 287 kilobytes = 73472 ints. So a histogram with >73472 bins would not fit into
    # a single thread block's shared memory. But with cluster size 2, we can use shared memory of 2 thread blocks, increasing
    # max bin count to 73472.
    #
    # Adapted from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory
    #

    num_bins = 2**17  # 131072
    num_elements = 4096
    threads_per_block = 1024

    num_blocks = num_elements // threads_per_block
    bins_per_block = num_bins // num_blocks

    with hidet.script_module() as script_module:

        @script
        def histogram(bins: ~i32, arr: ~i32, num_bins: i32, num_elements: i32, bins_per_block: i32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = num_blocks
            attrs.cuda.block_dim = threads_per_block
            attrs.cuda.cluster_dim = 2

            # sum of all smem = total needed for all bins
            # smem is defined per block
            attrs.cuda.dynamic_smem_bytes = bins_per_block * 4  # sizeof(int)

            bins = as_tensor_pointer(bins, i32, [num_elements])
            smem_bins = dynamic_shared_memory(byte_offset=0, dtype=i32)

            tid = blockDim.x * blockIdx.x + threadIdx.x

            i = threadIdx.x
            while i < bins_per_block:
                smem_bins[i] = 0
                i += 1

            # ensure all blocks in cluster have started and init 0
            this_cluster.sync()

            i = tid
            while i < num_elements:
                data = arr[i]
                bin_id = min(max(data, 0), num_bins - 1)

                dst_block_rank = bin_id // bins_per_block
                dst_offset = bin_id % bins_per_block

                # find mem address in target block space
                dst_smem = this_cluster.map_shared_rank(smem_bins, dst_block_rank, i32)

                # atomic add in smem instead of global mem
                atomic_add(dst_smem + dst_offset, Constant(1, i32))

                # next batch, used if # of elements > # of threads
                i += blockDim.x * gridDim.x

            # ensure all blocks in cluster finished computation
            this_cluster.sync()

            # move results to global mem
            global_cluster_offset = this_cluster.block_rank * bins_per_block
            i = threadIdx.x
            while i < bins_per_block:
                # atomic because multiple clusters
                atomic_add(bins + global_cluster_offset + i, smem_bins[i])
                i += blockDim.x

    module = script_module.build()

    arr = hidet.arange(0, num_elements, dtype=i32, device='cuda')
    bins = hidet.zeros([num_bins], dtype=i32, device='cuda')

    module(bins, arr, num_bins, num_elements, bins_per_block)
    hidet.cuda.synchronize()

    import torch

    assert torch.all(bins[:4096].torch())
    assert not torch.any(bins[4096:].torch())


@pytest.mark.hopper
def test_cluster_divides_grid():
    # grid_dim % cluster_dim != 0
    with hidet.script_module() as script_module:

        @script
        def a(x: ~i32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 2, 4
            attrs.cuda.block_dim = 16
            attrs.cuda.cluster_dim = 2, 3

            x += 1

    module = script_module.build()
    assert "assert(false)" in module.source()

    # prod(cluster_dim) > 8
    with hidet.script_module() as script_module:

        @script
        def a(x: ~i32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 4, 4
            attrs.cuda.block_dim = 4, 4
            attrs.cuda.cluster_dim = 4, 4

            x += 1

    module = script_module.build()
    assert "assert(false)" in module.source()


if __name__ == '__main__':
    pytest.main(__file__)
