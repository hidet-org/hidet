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
from hidet.ir.primitives.cuda import mbarrier_init, mbarrier_invalidate
import pytest

import hidet
from hidet.ir.builders import FunctionBuilder
from hidet.ir.expr import Constant, Var, tensor_var
from hidet.ir.primitives.cuda import blockDim, threadIdx, this_cluster
from hidet.ir.primitives.cuda.barrier import (
    fence_view_async_shared,
    mbarrier_arrive,
    mbarrier_expect_transaction,
    mbarrier_test_wait,
    mbarrier_try_wait,
    mbarrier_wait,
)
from hidet.ir.primitives.cuda.copy_tma import (
    copy_bulk_commit_group,
    copy_bulk_g2s,
    copy_bulk_s2g,
    copy_bulk_s2s,
    copy_bulk_wait_group,
    copy_tensor_g2s,
    copy_tensor_s2g,
)
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.primitives.cuda.tensor_map import create_tensor_map
from hidet.ir.primitives.debug import printf
from hidet.ir.type import OpaqueType, tensor_pointer_type, tensor_type
from hidet.ir.dtypes import i32, u32, u64, u16
from hidet.lang import attrs, script
from hidet.lang.constructs.declare import shared_tensor


@pytest.mark.hopper
def test_cp_async_bulk_g2s_multicast():
    """
    Test global to smem bulk async copy to multiple thread blocks at once
    """

    with hidet.script_module() as script_module:

        @script
        def cp_async_bulk_g2s_multicast(a: ~i32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 256
            attrs.cuda.grid_dim = 8
            attrs.cuda.cluster_dim = 8
            attrs.cuda.dynamic_smem_bytes = 256 * 4

            smem_a = dynamic_shared_memory(0, i32)
            smem_a[threadIdx.x] = 0

            mbar = shared_tensor("u64", [1])
            if threadIdx.x == 0:
                # mbarrier operations at thread block level
                mbarrier_init(mbar, Constant(256, i32))
                mbarrier_expect_transaction(mbar, 256 * 4)

            this_cluster.sync()  # wait for mbar to be created, smem init to 0

            for i in range(100):
                # cta0 issues copy command
                # each thread copies 16 bytes = 4 ints
                # so only 256 / 4 = 64 threads needed
                # each 16 bytes duplicated across 8 thread blocks
                if this_cluster.block_rank == 0 and threadIdx.x < 64:
                    target_index = threadIdx.x * 4
                    copy_bulk_g2s(16, smem_a + target_index, a + target_index, mbar, Constant(0b11111111, u16))

                mbarrier_arrive(mbar)
                if threadIdx.x == 0:
                    count = 0
                    wait_complete = 0
                    while wait_complete == 0:
                        wait_complete = mbarrier_test_wait(mbar, i & 1, wait_complete)
                        count += 1

                    printf("count: %d\\n", count)
                    mbarrier_expect_transaction(mbar, 256 * 4)

                this_cluster.sync()  # causes all threads to wait for mbarrier to complete
                assert smem_a[threadIdx.x] == 1
                smem_a[threadIdx.x] = 0
                assert smem_a[threadIdx.x] == 0
                this_cluster.sync()

            if threadIdx.x == 0:
                mbarrier_invalidate(mbar)

    module = script_module.build()

    a = hidet.ones([256], i32, device='cuda')
    module(a)

    hidet.cuda.synchronize()


@pytest.mark.hopper
def test_cp_async_bulk_g2s():
    """
    Test global to smem bulk async copy to single thread block
    """

    with hidet.script_module() as script_module:

        @script
        def cp_async_bulk_g2s(a: ~i32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 256
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 256 * 4

            smem_a = dynamic_shared_memory(0, i32)
            smem_a[threadIdx.x] = 0

            mbar = shared_tensor("u64", [1])
            if threadIdx.x == 0:
                # mbarrier operations at thread block level
                mbarrier_init(mbar, Constant(256, i32))
                mbarrier_expect_transaction(mbar, 256 * 4)

            syncthreads()

            for i in range(100):
                if threadIdx.x < 64:
                    target_index = threadIdx.x * 4
                    copy_bulk_g2s(16, smem_a + target_index, a + target_index, mbar)

                mbarrier_arrive(mbar)
                if threadIdx.x == 0:
                    count = 0
                    wait_complete = 0
                    while wait_complete == 0:
                        wait_complete = mbarrier_test_wait(mbar, i & 1, wait_complete)
                        count += 1

                    printf("count: %d\\n", count)
                    mbarrier_expect_transaction(mbar, 256 * 4)

                syncthreads()
                assert smem_a[threadIdx.x] == 1
                smem_a[threadIdx.x] = 0
                assert smem_a[threadIdx.x] == 0
                syncthreads()

            if threadIdx.x == 0:
                mbarrier_invalidate(mbar)

    module = script_module.build()

    a = hidet.ones([256], i32, device='cuda')
    module(a)

    hidet.cuda.synchronize()


@pytest.mark.hopper
def test_cp_async_bulk_s2g():
    """
    Test smem to global bulk async copy
    """

    with hidet.script_module() as script_module:

        @script
        def cp_async_bulk_s2g(a: ~i32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 256
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 256 * 4

            smem_a = dynamic_shared_memory(0, i32)
            smem_a[threadIdx.x] = 1

            syncthreads()

            for i in range(100):
                if threadIdx.x < 64:
                    target_index = threadIdx.x * 4
                    copy_bulk_s2g(16, a + target_index, smem_a + target_index)
                    copy_bulk_commit_group()

                    copy_bulk_wait_group(0)

                syncthreads()
                assert a[threadIdx.x] == 1
                a[threadIdx.x] = 0
                assert a[threadIdx.x] == 0
                syncthreads()

    module = script_module.build()

    a = hidet.zeros([256], i32, device='cuda')
    module(a)

    hidet.cuda.synchronize()


@pytest.mark.hopper
def test_cp_async_bulk_s2s():
    """
    Test smem thread block to smem cluster bulk async copy
    """
    with hidet.script_module() as script_module:

        @script
        def cp_async_bulk_s2s():
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 256
            attrs.cuda.cluster_dim = 2
            attrs.cuda.grid_dim = 2
            attrs.cuda.dynamic_smem_bytes = 256 * 4

            smem_a = dynamic_shared_memory(0, i32)
            smem_a[threadIdx.x] = this_cluster.block_rank  # one block has 1, the other 0

            mbar = shared_tensor("u64", [1])
            if threadIdx.x == 0:
                # mbarrier operations at thread block level
                mbarrier_init(mbar, Constant(256, i32))
                mbarrier_expect_transaction(mbar, 256 * 4)

            this_cluster.sync()

            last_value = this_cluster.block_rank
            for i in range(100):
                # swap smem
                if threadIdx.x < 64:
                    target_index = threadIdx.x * 4
                    target_addr = this_cluster.map_shared_rank(smem_a + target_index, 1 - this_cluster.block_rank, i32)
                    copy_bulk_s2s(16, target_addr, smem_a + target_index, mbar)
                mbarrier_arrive(mbar)
                if threadIdx.x == 0:
                    count = 0
                    wait_complete = 0
                    while wait_complete == 0:
                        wait_complete = mbarrier_test_wait(mbar, i & 1, wait_complete)
                        count += 1

                    printf("count: %d\\n", count)
                    mbarrier_expect_transaction(mbar, 256 * 4)

                this_cluster.sync()
                assert smem_a[threadIdx.x] == 1 - last_value
                last_value = 1 - last_value
                this_cluster.sync()

            if threadIdx.x == 0:
                mbarrier_invalidate(mbar)

    module = script_module.build()
    module()

    hidet.cuda.synchronize()


@pytest.mark.hopper
def test_cp_async_bulk_tensor_g2s():
    """
    Test global to smem bulk async tensor copy to single thread block
    """

    with hidet.script_module() as script_module:

        @script
        def cp_async_bulk_tensor_g2s(tensor_map: OpaqueType('CUtensorMap', 'const', '__grid_constant__')):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 16, 16
            attrs.cuda.grid_dim = 1, 1
            attrs.cuda.dynamic_smem_bytes = 256 * 4

            tid = threadIdx.x + threadIdx.y * blockDim.x
            smem_a = dynamic_shared_memory(0, i32)
            smem_a[tid] = 0

            mbar = shared_tensor("u64", [1])
            if tid == 0:
                # mbarrier operations at thread block level
                mbarrier_init(mbar, Constant(256, i32))
                mbarrier_expect_transaction(mbar, 256 * 4)

            syncthreads()

            for i in range(100):
                if tid == 0:
                    copy_tensor_g2s(2, smem_a, ~tensor_map, mbar, 0, 0)

                mbarrier_arrive(mbar)
                if tid == 0:
                    count = 0
                    wait_complete = 0
                    while wait_complete == 0:
                        wait_complete = mbarrier_test_wait(mbar, i & 1, wait_complete)
                        count += 1

                    printf("count: %d\\n", count)
                    mbarrier_expect_transaction(mbar, 256 * 4)

                syncthreads()
                assert smem_a[tid] == 1
                smem_a[tid] = 0
                assert smem_a[tid] == 0
                syncthreads()

            if tid == 0:
                mbarrier_invalidate(mbar)

        @script
        def launch(a: ~i32):
            attrs.func_kind = 'public'

            tensor_map: OpaqueType('CUtensorMap')

            size = tensor_type(u64, [2])
            size[0] = 16
            size[1] = 16

            stride = tensor_type(u64, [1])
            stride[0] = 16 * 4  # sizeof(int)

            box_size = tensor_type(u32, [2])
            box_size[0] = 16
            box_size[1] = 16

            elem_stride = tensor_type(u32, [2])
            elem_stride[0] = 1
            elem_stride[1] = 1

            create_tensor_map(~tensor_map, "int32", 2, a, size, stride, box_size, elem_stride)
            cp_async_bulk_tensor_g2s(tensor_map)

    module = script_module.build()
    a = hidet.ones([16, 16], i32, device='cuda')
    module(a)

    hidet.cuda.synchronize()


@pytest.mark.hopper
def test_cp_async_bulk_tensor_s2g():
    """
    Test smem to global bulk async tensor copy
    """

    with hidet.script_module() as script_module:

        @script
        def cp_async_bulk_tensor_s2g(tensor_map: OpaqueType('CUtensorMap', 'const', '__grid_constant__'), a: ~i32):
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 16, 16
            attrs.cuda.grid_dim = 1, 1
            attrs.cuda.dynamic_smem_bytes = 256 * 4  # sizeof(int)

            tid = threadIdx.x + threadIdx.y * blockDim.x
            smem_a = dynamic_shared_memory(0, i32)
            smem_a[tid] = 1
            assert smem_a[tid] == 1

            fence_view_async_shared()
            syncthreads()

            for i in range(100):
                if tid == 0:
                    copy_tensor_s2g(2, smem_a, ~tensor_map, 0, 0)

                    copy_bulk_commit_group()
                    copy_bulk_wait_group(0)

                fence_view_async_shared()
                syncthreads()

                # FIXME
                # The following assertion may fail on H100 because the fence
                # instruction doesn't seem to guarantee the memory consistency
                # as described in the document.
                # We filed an issue to NVIDIA. For details, please refer to
                # https://forums.developer.nvidia.com/t/tma-async-bulk-tensor-copy-memory-consistency/290971
                # We are waiting for the reply from NVIDIA so that we can fix
                # the issue.
                # The issue is tracked by the following ticket
                # https://github.com/CentML/hidet/issues/177
                assert a[tid] == 1
                a[tid] = 0
                assert a[tid] == 0

                fence_view_async_shared()
                syncthreads()

        @script
        def launch(a: tensor_type(i32, [16, 16])):
            attrs.func_kind = 'public'

            tensor_map: OpaqueType('CUtensorMap')

            size = tensor_type(u64, [2])
            size[0] = 16
            size[1] = 16

            stride = tensor_type(u64, [1])
            stride[0] = 16 * 4  # sizeof(int)

            box_size = tensor_type(u32, [2])
            box_size[0] = 16
            box_size[1] = 16

            elem_stride = tensor_type(u32, [2])
            elem_stride[0] = 1
            elem_stride[1] = 1

            create_tensor_map(~tensor_map, "int32", 2, a, size, stride, box_size, elem_stride)
            cp_async_bulk_tensor_s2g(tensor_map, a)

    module = script_module.build()
    a = hidet.zeros([16, 16], i32, device='cuda')
    module(a)

    hidet.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
