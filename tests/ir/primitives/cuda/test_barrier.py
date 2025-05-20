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
from hidet.ir.primitives.cuda import cp_async_barrier_arrive, mbarrier_init, mbarrier_invalidate
import pytest

import hidet
from hidet.ir.builders import FunctionBuilder
from hidet.ir.expr import Constant, Var, tensor_var
from hidet.ir.module import IRModule
from hidet.ir.primitives.cuda import cp_async, threadIdx, this_cluster
from hidet.ir.primitives.cuda.barrier import (
    mbarrier_arrive,
    mbarrier_arrive_and_expect_tx,
    mbarrier_complete_transaction,
    mbarrier_expect_transaction,
    mbarrier_test_wait,
    mbarrier_try_wait,
    mbarrier_wait,
    barrier_sync,
)
from hidet.ir.primitives.debug import printf
from hidet.ir.stmt import (
    AssertStmt,
    AssignStmt,
    BlackBoxStmt,
    DeclareStmt,
    DeclareScope,
    SeqStmt,
    WhileStmt,
    BufferStoreStmt,
)
from hidet.ir.type import tensor_pointer_type
from hidet.ir.dtypes import i32, u32, u64
from hidet.lang import attrs, script
from hidet.lang.constructs.declare import shared_tensor
from hidet.testing.capture_stdout import capture_stdout


@pytest.mark.requires_cuda
def test_mbarrier_basic():
    """
    Basic create/destroy mbarrier in shared memory.
    """
    with FunctionBuilder(name='mbarrier_basic', kind='cuda_kernel', grid_dim=1, block_dim=64) as fb:
        mbar = tensor_var('mbar', [1], 'u64')
        fb += DeclareStmt(mbar, scope=DeclareScope.Shared)
        fb += mbarrier_init(mbar, Constant(64, u32))
        fb += mbarrier_invalidate(mbar)

    func = fb.func
    ir_module = IRModule(functions={func.name: func})
    func = ir_module.build()
    func()
    hidet.cuda.synchronize()


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize('wait_type', ["wait", "test_wait", "try_wait"])
def test_mbarrier_cp_async_single_cta(wait_type: str):
    """
    Test blocking/non-blocking wait instructions on mbarrier.

    Check for eventual async completion and no race conditions.

    Uses local operations in cta state space.
    """
    with FunctionBuilder(name='mbarrier_cp_async', kind='cuda_kernel', grid_dim=1, block_dim=64) as fb:
        mbar = tensor_var('mbar', [1], 'u64')
        fb += DeclareStmt(mbar, scope=DeclareScope.Shared)

        a = Var('a', tensor_pointer_type("i32", [64]))
        fb.extend_params([a])

        smem_a = tensor_var('smem_a', [64], 'i32')
        fb += DeclareStmt(smem_a, scope=DeclareScope.Shared)

        fb += mbarrier_init(mbar, Constant(64, i32))

        with fb.for_loop("i", 1100) as i:
            fb += cp_async(~smem_a[threadIdx.x], ~a[threadIdx.x], 4)

            fb += cp_async_barrier_arrive(mbar)
            fb += mbarrier_arrive(mbar)

            if wait_type == "wait":
                fb += mbarrier_wait(mbar, i & 1)  # toggle barrier parity
            else:
                wait_complete = Var('wait_complete', u32)
                fb += DeclareStmt(wait_complete, Constant(0, u32))

                cnt = Var('cnt', u32)
                fb += DeclareStmt(cnt, Constant(0, u32))

                if wait_type == "test_wait":
                    fb += WhileStmt(
                        wait_complete == Constant(0, u32),
                        SeqStmt(
                            [
                                AssignStmt(wait_complete, mbarrier_test_wait(mbar, i & 1, wait_complete)),
                                AssignStmt(cnt, cnt + 1),
                            ]
                        ),
                    )

                elif wait_type == "try_wait":
                    fb += WhileStmt(
                        wait_complete == Constant(0, u32),
                        SeqStmt([AssignStmt(wait_complete, mbarrier_try_wait(mbar, i & 1)), AssignStmt(cnt, cnt + 1)]),
                    )

                fb += BlackBoxStmt('printf("c: %d\\n", {});', cnt)

            fb += AssertStmt(smem_a[threadIdx.x] == Constant(1, i32), msg="Async assignment eventually appears")

            fb += BufferStoreStmt(smem_a, [threadIdx.x], Constant(0, i32))
            fb += AssertStmt(smem_a[threadIdx.x] == 0, msg="Immediate assignment")

        fb += mbarrier_invalidate(mbar)

    func = fb.func
    ir_module = IRModule(functions={func.name: func})
    func = ir_module.build()

    a = hidet.ones([64], dtype='i32', device='cuda')
    func(a)

    hidet.cuda.synchronize()


@pytest.mark.requires_cuda_hopper
def test_mbarrier_cp_async_cluster():
    """
    Test blocking/non-blocking wait instructions on mbarrier.

    Check for eventual async completion and no race conditions.

    Uses cluster state space.
    """

    with hidet.script_module() as script_module:

        @script
        def cluster_arrive():
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 1024
            attrs.cuda.grid_dim = 8
            attrs.cuda.cluster_dim = 8

            mbar = shared_tensor("u64", [1])
            if this_cluster.thread_rank == 0:
                mbarrier_init(mbar, Constant(1024 * 8, i32))
                mbarrier_expect_transaction(mbar, 1024 * 8)

            this_cluster.sync()  # wait for mbar to be created

            for i in range(100):
                mbarrier_complete_transaction(mbar, 1, 0, True)  # cluster invocations, includes mapping to cta0 smem
                mbarrier_arrive(mbar, 0, True)

                if this_cluster.thread_rank == 0:
                    count = 0
                    wait_complete = 0
                    while wait_complete == 0:
                        wait_complete = mbarrier_test_wait(mbar, i & 1, wait_complete)
                        count += 1

                    printf("count: %d\\n", count)
                    mbarrier_expect_transaction(mbar, 1024 * 8)

                this_cluster.sync()  # causes all threads to wait for mbarrier to complete

            if this_cluster.thread_rank == 0:
                mbarrier_invalidate(mbar)

    module = script_module.build()
    module()

    hidet.cuda.synchronize()


@pytest.mark.requires_cuda_hopper
def test_mbarrier_tx_count_ops():
    """
    Test transaction count operations with arrive-on operation for phase completion.
    """

    with FunctionBuilder(name='mbarrier_tx', kind='cuda_kernel', grid_dim=1, block_dim=64) as fb:
        mbar = tensor_var('mbar', [1], 'u64')
        fb += DeclareStmt(mbar, scope=DeclareScope.Shared)
        fb += mbarrier_init(mbar, Constant(64, u32))

        # mbarrier will require 200 transactions + 64 threads to arrive on barrier
        fb += mbarrier_expect_transaction(mbar, Constant(100, u32))
        fb += mbarrier_arrive_and_expect_tx(mbar, Constant(100, u32))

        wait_complete = Var('wait_complete', u32)
        fb += DeclareStmt(wait_complete, Constant(0, u32))

        # some delay
        fb += BlackBoxStmt('printf("d");')
        fb += AssignStmt(wait_complete, mbarrier_test_wait(mbar, Constant(0, u32), wait_complete))

        # test wait should fail, need 200 transactions to complete
        fb += AssertStmt(wait_complete == Constant(0, u32), msg="Need all transactions to complete")

        # test wait should fail, need another 100 transactions to complete
        fb += mbarrier_complete_transaction(mbar, Constant(100, u32))

        # some delay
        fb += BlackBoxStmt('printf("d");')
        fb += AssignStmt(wait_complete, mbarrier_test_wait(mbar, Constant(0, u32), wait_complete))

        # test wait should fail, need 200 transactions to complete
        fb += AssertStmt(wait_complete == Constant(0, u32), msg="Need all transactions to complete")

        fb += mbarrier_complete_transaction(mbar, Constant(100, u32))

        # some delay
        fb += BlackBoxStmt('printf("d");')
        fb += AssignStmt(wait_complete, mbarrier_test_wait(mbar, Constant(0, u32), wait_complete))

        # test wait should pass, 0 transactions and pending arrivals
        fb += AssertStmt(wait_complete == Constant(1, u32), msg="Need all transactions to complete")

        fb += mbarrier_invalidate(mbar)

    func = fb.func
    ir_module = IRModule(functions={func.name: func})
    func = ir_module.build()
    func()
    hidet.cuda.synchronize()


@pytest.mark.requires_cuda
def test_barrier():
    from hidet.lang import attrs, printf, asm
    from hidet.lang.cuda import threadIdx, syncthreads
    from hidet.ir.primitives.cuda import barrier_sync

    with hidet.script_module() as script_module:

        num_groups = 2
        group_size = 32

        @hidet.script
        def with_barrier():
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = num_groups * group_size

            for i in range(num_groups):
                if threadIdx.x // group_size == i:
                    if threadIdx.x % group_size <= 1:
                        asm('nanosleep.u32 1024;')
                        printf('group %d, thread %d, before sync\n', i, threadIdx.x % group_size)
                    barrier_sync(1, group_size)
                    if group_size - 1 - threadIdx.x % group_size <= 1:
                        printf('group %d, thread %d, after sync\n', i, threadIdx.x % group_size)

                barrier_sync(0, aligned=True)
                syncthreads()

        @hidet.script
        def without_barrier():
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = num_groups * group_size

            for i in range(num_groups):
                if threadIdx.x // group_size == i:
                    if threadIdx.x % group_size <= 1:
                        asm('nanosleep.u32 1024;')
                        printf('group %d, thread %d, before sync\n', i, threadIdx.x % group_size)
                    if group_size - 1 - threadIdx.x % group_size <= 1:
                        printf('group %d, thread %d, after sync\n', i, threadIdx.x % group_size)

                barrier_sync(0, aligned=True)
                syncthreads()

        @hidet.script
        def launch():
            attrs.func_kind = 'public'
            printf('with barrier\n')
            with_barrier()
            BlackBoxStmt('cudaDeviceSynchronize();')
            printf('without barrier\n')
            without_barrier()
            BlackBoxStmt('cudaDeviceSynchronize();')

    func = script_module.build()
    with capture_stdout() as captured:
        func()

    assert (
        str(captured).strip()
        == """
with barrier
group 0, thread 0, before sync
group 0, thread 1, before sync
group 0, thread 30, after sync
group 0, thread 31, after sync
group 1, thread 0, before sync
group 1, thread 1, before sync
group 1, thread 30, after sync
group 1, thread 31, after sync
without barrier
group 0, thread 30, after sync
group 0, thread 31, after sync
group 0, thread 0, before sync
group 0, thread 1, before sync
group 1, thread 30, after sync
group 1, thread 31, after sync
group 1, thread 0, before sync
group 1, thread 1, before sync
    """.strip()
    )


if __name__ == "__main__":
    pytest.main([__file__])
