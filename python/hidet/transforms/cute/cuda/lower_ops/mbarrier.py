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
from typing import List, Union

from hidet.ir.expr import Expr, is_constant
from hidet.ir.dtypes import u64, u32

from hidet.ir.cute.ops.copy import MBarriers, MBarrierArrive, MBarrierTryWait, MBarrierWait

from hidet.ir.primitives.cuda.barrier import (
    mbarrier_expect_transaction,
    mbarrier_arrive,
    mbarrier_try_wait,
    mbarrier_wait,
    mbarrier_init,
    fence_view_async_shared,
)
from hidet.ir.primitives.cuda import cp_async_barrier_arrive
from hidet.ir.primitives.cuda import threadIdx
from hidet.ir.cute.contexts import tid_in_groups

from .registry import OpEmitter, Buffer, register_impl


@register_impl(MBarriers)
class MBarriersEmitter(OpEmitter):
    def request_smem_nbytes(self, op: MBarriers):
        return op.num_barriers * u64.nbytes

    def emit(self, op: MBarriers, args: List[Union[Buffer, Expr]], output: Buffer):
        output.buffer = self.auto_var(hint=op.name, e=self.get_smem_ptr(op, u64, 0))

        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        with self.if_then(tid == 0):
            annotations = op.annotations
            assert "num_threads" in annotations
            num_threads = annotations["num_threads"]
            with self.for_grid([op.num_barriers]) as i:
                self.append(mbarrier_init(output.buffer + i, num_threads))
            self.append(fence_view_async_shared())


@register_impl(MBarrierArrive)
class MBarrierArriveEmitter(OpEmitter):
    def emit(self, op: MBarrierArrive, args: List[Union[Buffer, Expr]], output: Buffer):
        mbarrier = args[0]
        mbarrier = mbarrier.buffer + mbarrier.offset
        if "group_ids" in op.annotations:
            group_ids = op.annotations["group_ids"]
            tid = tid_in_groups(group_ids)
        else:
            tid = threadIdx.x

        # should have some mechanism to infer multicast
        if not is_constant(op.count) or not op.count == 0:
            with self.if_then(tid == 0):
                self.append(mbarrier_expect_transaction(mbarrier, op.count))
        annotations = op.annotations
        if 'tma_fallback_copy' in annotations:
            self.append(cp_async_barrier_arrive(mbarrier))
        self.append(mbarrier_arrive(mbarrier))


@register_impl(MBarrierTryWait)
class MBarrierTryWaitEmitter(OpEmitter):
    def emit(self, op: MBarrierTryWait, args: List[Union[Buffer, Expr]], output: Buffer):
        mbarrier = args[0]
        mbarrier = mbarrier.buffer + mbarrier.offset
        phase = u32(op.phase)
        self.buffer_store(output.buffer, [0], mbarrier_try_wait(mbarrier, phase))


@register_impl(MBarrierWait)
class MBarrierWaitEmitter(OpEmitter):
    def emit(self, op: MBarrierWait, args: List[Union[Buffer, Expr]], output: Buffer):
        mbarrier = args[0]
        mbarrier = mbarrier.buffer + mbarrier.offset
        phase = u32(op.phase)
        self.append(mbarrier_wait(mbarrier, phase))
