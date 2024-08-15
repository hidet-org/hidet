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
# pylint: disable=line-too-long
from typing import Optional, Union
from hidet.utils import initialize
from hidet.ir.expr import Constant, Expr
from hidet.ir.stmt import asm
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.lang import script, i32, u32, u64, int32, attrs


@initialize()
def register_mbarrier():
    from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

    func_name = 'cuda_mbarrier_init'
    template_string = 'mbarrier.init.shared::cta.b64 [%1], %0;'

    @script
    def cuda_mbarrier_init(mbar: ~u64, arrive_count: u32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[arrive_count, smem_addr])

    assert isinstance(cuda_mbarrier_init, Function)
    register_primitive_function(name=cuda_mbarrier_init.name, func_or_type=cuda_mbarrier_init)

    func_name = 'cuda_mbarrier_wait'
    template_string = (
        '{ .reg.pred P1; LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; @!P1 bra.uni LAB_WAIT; }'
    )

    @script
    def cuda_mbarrier_wait(mbar: ~u64, phase: u32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        ticks = u32(10_000_000)
        asm(template=template_string, inputs=[smem_addr, phase, ticks])

    assert isinstance(cuda_mbarrier_wait, Function)
    register_primitive_function(name=cuda_mbarrier_wait.name, func_or_type=cuda_mbarrier_wait)

    func_name = 'cuda_mbarrier_test_wait'
    template_string = '{ .reg .pred P1; .reg .pred P2; setp.eq.u32 P2, %3, 1; @P2 mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2; selp.u32 %0, 1, 0, P1; }'

    @script
    def cuda_mbarrier_test_wait(mbar: ~u64, phase: u32, wait_complete: u32, pred: u32) -> u32:
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, outputs=[wait_complete], inputs=[smem_addr, phase, pred])
        return wait_complete

    assert isinstance(cuda_mbarrier_test_wait, Function)
    register_primitive_function(name=cuda_mbarrier_test_wait.name, func_or_type=cuda_mbarrier_test_wait)

    func_name = 'cuda_mbarrier_try_wait'
    template_string = '{ .reg .pred P1; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2; selp.u32 %0, 1, 0, P1; }'

    @script
    def cuda_mbarrier_try_wait(mbar: ~u64, phase: u32, wait_complete: u32) -> u32:
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, outputs=[wait_complete], inputs=[smem_addr, phase])
        return wait_complete

    assert isinstance(cuda_mbarrier_try_wait, Function)
    register_primitive_function(name=cuda_mbarrier_try_wait.name, func_or_type=cuda_mbarrier_try_wait)

    func_name = 'cuda_mbarrier_arrive'
    template_string = '{ .reg.pred p; .reg.b32 remAddr32; setp.eq.u32 p, %2, 1; @p mapa.shared::cluster.u32 remAddr32, %0, %1; @p mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32]; }'

    @script
    def cuda_mbarrier_arrive(mbar: ~u64, cta_id: u32, pred: u32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr, cta_id, pred])

    assert isinstance(cuda_mbarrier_arrive, Function)
    register_primitive_function(name=cuda_mbarrier_arrive.name, func_or_type=cuda_mbarrier_arrive)

    func_name = 'cuda_mbarrier_arrive_on_local'
    template_string = 'mbarrier.arrive.shared::cta.b64 _, [%0];'

    @script
    def cuda_mbarrier_arrive_on_local(mbar: ~u64):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr])

    assert isinstance(cuda_mbarrier_arrive_on_local, Function)
    register_primitive_function(name=cuda_mbarrier_arrive_on_local.name, func_or_type=cuda_mbarrier_arrive_on_local)

    func_name = 'cuda_mbarrier_invalidate'
    template_string = 'mbarrier.inval.shared::cta.b64 [%0];'

    @script
    def cuda_mbarrier_invalidate(mbar: ~u64):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr])

    assert isinstance(cuda_mbarrier_invalidate, Function)
    register_primitive_function(name=cuda_mbarrier_invalidate.name, func_or_type=cuda_mbarrier_invalidate)

    func_name = 'cuda_mbarrier_arrive_and_expect_tx_on_local'
    template_string = 'mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;'

    @script
    def cuda_mbarrier_arrive_and_expect_tx_on_local(mbar: ~u64, transaction_bytes: u32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[transaction_bytes, smem_addr])

    assert isinstance(cuda_mbarrier_arrive_and_expect_tx_on_local, Function)
    register_primitive_function(
        name=cuda_mbarrier_arrive_and_expect_tx_on_local.name, func_or_type=cuda_mbarrier_arrive_and_expect_tx_on_local
    )

    func_name = 'cuda_mbarrier_arrive_and_expect_tx'
    template_string = '{ .reg .pred p; .reg .b32 remAddr32; setp.eq.u32 p, %2, 1; @p mapa.shared::cluster.u32 remAddr32, %0, %1; @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remAddr32], %3; }'

    @script
    def cuda_mbarrier_arrive_and_expect_tx(mbar: ~u64, transaction_bytes: u32, cta_id: u32, pred: i32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr, cta_id, pred, transaction_bytes])

    assert isinstance(cuda_mbarrier_arrive_and_expect_tx, Function)
    register_primitive_function(
        name=cuda_mbarrier_arrive_and_expect_tx.name, func_or_type=cuda_mbarrier_arrive_and_expect_tx
    )

    func_name = 'cuda_mbarrier_expect_transaction'
    template_string = 'mbarrier.expect_tx.shared::cta.b64 [%0], %1;'

    @script
    def cuda_mbarrier_expect_transaction(mbar: ~u64, transaction_bytes: u32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr, transaction_bytes])

    assert isinstance(cuda_mbarrier_expect_transaction, Function)
    register_primitive_function(
        name=cuda_mbarrier_expect_transaction.name, func_or_type=cuda_mbarrier_expect_transaction
    )

    func_name = 'cuda_mbarrier_complete_transaction'
    template_string = '{ .reg .pred p; .reg .b32 remAddr32; setp.eq.u32 p, %2, 1; @p mapa.shared::cluster.u32 remAddr32, %0, %1; @p mbarrier.complete_tx.relaxed.cluster.shared.b64 [remAddr32], %3; }'

    @script
    def cuda_mbarrier_complete_transaction(mbar: ~u64, transaction_bytes: u32, dst_cta_id: u32, pred: u32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr, dst_cta_id, pred, transaction_bytes])

    assert isinstance(cuda_mbarrier_complete_transaction, Function)
    register_primitive_function(
        name=cuda_mbarrier_complete_transaction.name, func_or_type=cuda_mbarrier_complete_transaction
    )

    func_name = 'cuda_mbarrier_complete_transaction_on_local'
    template_string = 'mbarrier.complete_tx.shared::cta.b64 [%0], %1;'

    @script
    def cuda_mbarrier_complete_transaction_on_local(mbar: ~u64, transaction_bytes: u32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr, transaction_bytes])

    assert isinstance(cuda_mbarrier_complete_transaction_on_local, Function)
    register_primitive_function(
        name=cuda_mbarrier_complete_transaction_on_local.name, func_or_type=cuda_mbarrier_complete_transaction_on_local
    )

    func_name = 'cuda_fence_barrier_init'
    template_string = 'fence.mbarrier_init.release.cluster;'

    @script
    def cuda_fence_barrier_init():
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string, inputs=[])

    assert isinstance(cuda_fence_barrier_init, Function)
    register_primitive_function(name=cuda_fence_barrier_init.name, func_or_type=cuda_fence_barrier_init)

    func_name = 'cuda_fence_view_async_shared'
    template_string = 'fence.proxy.async.shared::cta;'

    @script
    def cuda_fence_view_async_shared():
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string, inputs=[])

    assert isinstance(cuda_fence_view_async_shared, Function)
    register_primitive_function(name=cuda_fence_view_async_shared.name, func_or_type=cuda_fence_view_async_shared)

    func_name = 'cuda_cp_async_barrier_arrive'
    template_string = 'cp.async.mbarrier.arrive.shared::cta.b64 [%0];'

    @script
    def cuda_cp_async_barrier_arrive(mbar: ~u64):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        smem_addr = cvta_generic_to_shared(mbar)
        asm(template=template_string, inputs=[smem_addr])

    assert isinstance(cuda_cp_async_barrier_arrive, Function)
    register_primitive_function(name=cuda_cp_async_barrier_arrive.name, func_or_type=cuda_cp_async_barrier_arrive)


@initialize()
def register_barrier():
    for aligned in [False, True]:
        for mode in ['arrive', 'sync', 'sync_all']:
            func_name = 'barrier_{}{}'.format(mode, '_aligned' if aligned else '')

            if mode == 'sync_all':
                template = 'barrier.sync{} %0;'.format('.aligned' if aligned else '')

                @script
                def barrier_func(barrier: int32):
                    attrs.func_kind = 'cuda_internal'
                    attrs.func_name = func_name

                    asm(template, inputs=[barrier], is_volatile=True)

            else:
                template = 'barrier.sync{} %0, %1;'.format('.aligned' if aligned else '')

                @script
                def barrier_func(barrier: int32, count: int32):
                    attrs.func_kind = 'cuda_internal'
                    attrs.func_name = func_name

                    asm(template, inputs=[barrier, count], is_volatile=True)

            assert isinstance(barrier_func, Function)
            register_primitive_function(name=barrier_func.name, func_or_type=barrier_func)


def mbarrier_init(mbar: Expr, arrive_count: Expr):
    """
    Init a barrier
    """
    func_name = 'mbarrier_init'
    return call_cuda(func_name, [mbar, arrive_count])


def mbarrier_wait(mbar: Expr, phase: Expr):
    """
    Wait on a barrier
    """
    func_name = 'mbarrier_wait'
    return call_cuda(func_name, [mbar, phase])


def mbarrier_test_wait(mbar: Expr, phase: Expr, wait_complete: Expr, pred: Expr = Constant(1, u32)):
    """
    Test wait if pred is true (skip otherwise)
    """
    func_name = 'mbarrier_test_wait'
    return call_cuda(func_name, [mbar, phase, wait_complete, pred])


def mbarrier_try_wait(mbar: Expr, phase: Expr, wait_complete: Expr):
    """
    Try wait
    """
    func_name = 'mbarrier_try_wait'
    return call_cuda(func_name, [mbar, phase, wait_complete])


def mbarrier_arrive(mbar: Expr, cta_id: Optional[Expr] = None, pred: Optional[Expr] = None):
    """
    Arrive
    """
    if not ((cta_id is None and pred is None) or (cta_id is not None and pred is not None)):
        raise ValueError('cta_id and pred are not both given or None. (cta_id:{},pred:{})'.format(cta_id, pred))
    arrive_on_local = cta_id is None and pred is None
    func_name = 'mbarrier_arrive' if not arrive_on_local else 'mbarrier_arrive_on_local'
    if arrive_on_local:
        return call_cuda(func_name, [mbar])
    else:
        return call_cuda(func_name, [mbar, cta_id, pred])


def mbarrier_invalidate(mbar: Expr):
    """
    Invalidate
    """
    func_name = 'mbarrier_invalidate'
    return call_cuda(func_name, [mbar])


def mbarrier_arrive_and_expect_tx(
    mbar: Expr, transaction_bytes: Expr, cta_id: Optional[Expr] = None, pred: Optional[Expr] = None
):
    """
    Arrive and expect transaction count
    """
    if not ((cta_id is None and pred is None) or (cta_id is not None and pred is not None)):
        raise ValueError('cta_id and pred are not both given or None. (cta_id:{} ,pred:{})'.format(cta_id, pred))
    arrive_on_local = cta_id is None and pred is None
    func_name = 'mbarrier_arrive_and_expect_tx' if not arrive_on_local else 'mbarrier_arrive_and_expect_tx_on_local'
    if arrive_on_local:
        return call_cuda(func_name, [mbar, transaction_bytes])
    else:
        return call_cuda(func_name, [mbar, transaction_bytes, cta_id, pred])


def mbarrier_expect_transaction(mbar: Expr, transaction_bytes: Expr):
    """
    Expect transactions
    """
    func_name = 'mbarrier_expect_transaction'
    return call_cuda(func_name, [mbar, transaction_bytes])


def mbarrier_complete_transaction(
    mbar: Expr, transaction_bytes: Expr, cta_id: Optional[Expr] = None, pred: Optional[Expr] = None
):
    """
    Complete transaction
    """
    if not ((cta_id is None and pred is None) or (cta_id is not None and pred is not None)):
        raise ValueError('cta_id and pred are not both given or None. (cta_id:{} ,pred:{})'.format(cta_id, pred))
    arrive_on_local = cta_id is None and pred is None
    func_name = 'mbarrier_complete_transaction' if not arrive_on_local else 'mbarrier_complete_transaction_on_local'
    if arrive_on_local:
        return call_cuda(func_name, [mbar, transaction_bytes])
    else:
        return call_cuda(func_name, [mbar, transaction_bytes, cta_id, pred])


def fence_barrier_init():
    """
    Fence barrier init
    """
    return call_cuda('fence_barrier_init', [])


def fence_view_async_shared():
    """
    Fence view async shared
    """
    return call_cuda('fence_view_async_shared', [])


def cp_async_barrier_arrive(mbar: Expr):
    """
    cp async barrier arrive
    """
    return call_cuda('cp_async_barrier_arrive', [mbar])


def _barrier(barrier: Union[int, Expr], count: Optional[Union[int, Expr]], aligned: bool, mode: str):
    # resolve function name
    func_name = 'barrier_{}{}'.format(mode, '_aligned' if aligned else '')

    # call the function
    args = [barrier]
    if count is not None:
        args.append(count)
    return call_primitive_func(func_name, args=args)


def barrier_sync(barrier: Union[int, Expr], count: Optional[Union[int, Expr]] = None, aligned: bool = False):
    """
    Performs barrier synchronization and communication within a CTA.

    The threads will synchronize at the named barrier.

    See Also
    --------
    The PTX ISA documentation for the `barrier` instruction:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier

    Parameters
    ----------
    barrier:
        The named barrier to synchronize on. This must be an integer from 0 to 15.

    count: Optional[int]
        The number of threads to synchronize. If not provided, all threads in the CTA will synchronize.

    aligned:
        When specified, it indicates that all threads in CTA will execute the same barrier instruction.
    """
    mode = 'sync_all' if count is None else 'sync'
    return _barrier(barrier, count, aligned, mode=mode)


def barrier_arrive(barrier: Union[int, Expr], count: Union[int, Expr], aligned: bool = False):
    """
    Performs barrier synchronization and communication within a CTA.

    The threads will mark their arrival at the named barrier but will not be blocked.

    See Also
    --------
    The PTX ISA documentation for the `barrier` instruction:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier

    Parameters
    ----------
    barrier: Union[int, Expr]
        The named barrier to synchronize on. This must be an integer from 0 to 15.

    count: Union[int, Expr]
        The number of threads to synchronize.

    aligned: bool
        When specified, it indicates that all threads in CTA will execute the same barrier instruction.
    """
    return _barrier(barrier, count, aligned, mode='arrive')
