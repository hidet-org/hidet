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

"""
This module provides functionality for annotating memory barriers (mbarriers) in CUDA code.

The module analyzes and annotates memory barrier operations with the number of threads that will join
the barrier. The thread count is determined based on the following rules:
1. When a barrier is used in a copy operation, the thread count equals the number of threads working
   together to perform the copy operation.
2. When a barrier is not used in any copy operation, the thread count equals the total thread count
   in the CUDA kernel.
3. When warp specialization context is specified, the thread count equals the number of threads in
   the current warp groups.
"""

from typing import List, Dict

from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.expr import Var
from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass

from hidet.ir.cute.ops import MBarriers, Copy, MBarrierArrive
from hidet.ir.cute import canonicalize_thread_value_layout
from hidet.transforms.cute.analysis import TensorInfo, TensorAliasAnalysis


class MBarrierCollector(IRVisitor):
    """
    A visitor class that collects memory barrier operations and their associated copy operations.

    This class traverses the IR and identifies where memory barriers are used in copy operations,
    building a mapping between memory barriers and their corresponding copy operations. This mapping
    is used to determine the number of threads that will join each barrier based on the copy operation's
    thread requirements.

    Attributes:
        var2tensor (Dict[Var, TensorInfo]): Mapping from variables to their tensor information.
        mbarrier2copies (Dict[MBarriers, List[Copy]]): Mapping from memory barriers to their associated copy operations.
    """

    def __init__(self, var2tensor: Dict[Var, TensorInfo]):
        super().__init__()
        self.var2tensor: Dict[Var, TensorInfo] = var2tensor
        self.mbarrier2copies: Dict[MBarriers, List[Copy]] = {}
        self.mbarrier2context_threads: Dict[MBarriers, int] = {}

    def visit_Copy(self, copy: Copy):
        """
        Visit a copy operation and collect associated memory barrier information.

        This method identifies when a memory barrier is used in a copy operation and records
        this relationship. The copy operation's thread requirements will be used to determine
        the number of threads that will join the barrier.

        Args:
            copy (Copy): The copy operation to analyze.
        """
        mbarrier = copy.mbarrier
        if mbarrier is None:
            return
        tensor_info = self.var2tensor.get(mbarrier, None)
        if tensor_info is not None:
            mbarrier = tensor_info.tensor
            assert isinstance(mbarrier, MBarriers)
            if mbarrier in self.mbarrier2copies:
                self.mbarrier2copies[mbarrier].append(copy)
            else:
                self.mbarrier2copies[mbarrier] = [copy]

    def visit_MBarrierArrive(self, op: MBarrierArrive):
        mbarrier = op.mbarrier
        tensor_info = self.var2tensor.get(mbarrier, None)
        assert tensor_info is not None
        mbarrier = tensor_info.tensor
        assert isinstance(mbarrier, MBarriers)
        if "group_ids" in op.annotations:
            group_threads = op.annotations["group_threads"]
            if mbarrier in self.mbarrier2context_threads:
                if self.mbarrier2context_threads[mbarrier] != group_threads:
                    raise AssertionError(
                        f"MBarriers {mbarrier} is used in different warp groups with different number of threads."
                        f" (got: {group_threads}, expected: {self.mbarrier2context_threads[mbarrier]})"
                    )
            else:
                self.mbarrier2context_threads[mbarrier] = group_threads

    def collect(self, func: Function):
        """
        Collect all memory barrier information from a function.

        This method traverses the entire function to identify all memory barriers and their
        relationships with copy operations. This information is used to determine the appropriate
        thread count for each barrier.

        Args:
            func (Function): The function to analyze.

        Returns:
            Tuple[Dict[MBarriers, List[Copy]], Dict[MBarriers, int]]:
            Mapping from memory barriers to their associated copy operations and context threads.
        """
        self.visit(func)
        return self.mbarrier2copies, self.mbarrier2context_threads


class AnnotateMBarrierRewriter(IRRewriter):
    """
    A rewriter class that annotates memory barriers with thread count information.

    This class processes memory barriers and adds annotations about the number of threads
    that will join each barrier. The thread count is determined based on the following rules:
    1. For barriers used in copy operations: thread count equals the number of threads
       working together in the copy operation.
    2. For barriers not used in copy operations: thread count equals the kernel's total
       thread count.
    3. For barriers in warp specialization context: thread count equals the number of
       threads in the current warp groups.

    Attributes:
        mbarrier2copies (Dict[MBarriers, List[Copy]]): Mapping from memory barriers to their associated copy operations.
        num_threads (int): Default number of threads to use when not specified by copy operations.
    """

    def __init__(
        self,
        mbarrier2copies: Dict[MBarriers, List[Copy]],
        mbarrier2context_threads: Dict[MBarriers, int],
        num_threads: int,
    ):
        super().__init__()
        self.mbarrier2copies: Dict[MBarriers, List[Copy]] = mbarrier2copies
        self.mbarrier2context_threads: Dict[MBarriers, int] = mbarrier2context_threads
        self.num_threads: int = num_threads

    def visit_MBarriers(self, mbarrier: MBarriers):
        """
        Visit a memory barrier operation and add thread count annotations.

        This method determines the appropriate thread count for a memory barrier based on its usage:
        - If the barrier is used in copy operations, the thread count is derived from the copy
          operation's thread requirements.
        - If the barrier is not used in any copy operation, the thread count is taken from the
          kernel's thread count.
        - In warp specialization context, the thread count is based on the current warp group size.

        Args:
            mbarrier (MBarriers): The memory barrier operation to process.

        Returns:
            MBarriers: The annotated memory barrier operation with thread count information.

        Raises:
            AssertionError: If memory barriers are used in different copies with different thread counts.
        """
        copies = self.mbarrier2copies.get(mbarrier, None)
        if copies is not None:
            num_threads = None
            for copy in copies:
                tiled_copy = copy.tiled_copy
                _, src_tv = tiled_copy.src_tv_layout()
                _, dst_tv = tiled_copy.dst_tv_layout()
                src_t, _ = canonicalize_thread_value_layout(src_tv)
                dst_t, _ = canonicalize_thread_value_layout(dst_tv)
                assert src_t.size() == dst_t.size()
                if num_threads is None:
                    num_threads = src_t.size()
                else:
                    if num_threads != src_t.size():
                        raise AssertionError(
                            f"MBarriers {mbarrier} is used in different copies with different"
                            f" number of threads. (got: {src_t.size()}, expected: {num_threads})"
                        )
        else:
            context_threads = self.mbarrier2context_threads.get(mbarrier, None)
            num_threads = context_threads if context_threads is not None else self.num_threads
            if num_threads is None:
                raise AssertionError(f"MBarriers {mbarrier} appears in a non-cuda function.")
        annotations = {}
        annotations["num_threads"] = num_threads
        return mbarrier.reforward([], annotations_update=annotations)


class AnnotateMBarrierPass(FunctionPass):
    """
    A function pass that annotates memory barriers in CUDA functions.

    This pass analyzes CUDA functions to identify memory barriers and their usage in copy operations,
    then adds appropriate annotations about the number of threads that will join each barrier.
    The thread count is determined based on the following rules:
    1. For barriers used in copy operations: thread count equals the number of threads
       working together in the copy operation.
    2. For barriers not used in copy operations: thread count equals the kernel's total
       thread count.
    3. For barriers in warp specialization context: thread count equals the number of
       threads in the current warp groups.
    """

    def __init__(self):
        super().__init__()

    def process_func(self, func: Function) -> Function:
        """
        Process a function to annotate its memory barriers.

        This method analyzes the function to identify memory barriers and their relationships
        with copy operations, then adds appropriate thread count annotations to each barrier.

        Args:
            func (Function): The CUDA function to process.

        Returns:
            Function: The processed function with annotated memory barriers.
        """
        func_attrs = func.attrs
        if "cuda.block_dim" in func_attrs:
            num_threads = func_attrs["cuda.block_dim"]
        else:
            num_threads = None

        tensor_alias_analysis = TensorAliasAnalysis()
        var2tensor = tensor_alias_analysis.analyze(func)

        mbarrier2copies, mbarrier2context_threads = MBarrierCollector(var2tensor).collect(func)
        return AnnotateMBarrierRewriter(mbarrier2copies, mbarrier2context_threads, num_threads)(func)


def annotate_mbarrier_pass() -> FunctionPass:
    """
    Create an instance of the memory barrier annotation pass.

    Returns:
        FunctionPass: An instance of AnnotateMBarrierPass.
    """
    return AnnotateMBarrierPass()
