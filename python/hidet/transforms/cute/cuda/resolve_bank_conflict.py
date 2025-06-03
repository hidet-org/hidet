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
from typing import List, Dict, Set

from hidet.ir.tools import TypeInfer
from hidet.ir.functors import IRRewriter
from hidet.ir.cute.expr import CallOp

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
from hidet.ir.expr import Var, var
from hidet.ir.stmt import DeclareStmt
from hidet.ir.cute import (
    Swizzle,
    LayoutBase,
    TensorLayout,
    ComposedTensorLayout,
    composition,
    canonicalize_thread_value_layout,
    group,
    make_swizzle,
)
from hidet.ir.cute.ops import Copy, TensorBase, Tensor, TensorView, Mma
from hidet.ir.cute.expr import CConst
from hidet.transforms.cute.analysis import TensorAliasAnalysis


class ResolveBankConflict(TensorAliasAnalysis):
    """
    ResolveBankConflict pass resolves bank conflicts in shared memory.

    In the CuTe IR, tensors are first-class citizens. All data is stored in tensors, and tile-level
    operations perform computations on these tensors. This design allows for simplified tensor alias analysis
    and the elimination of bank conflicts in shared memory accesses.

    After a tensor is created, it can be partitioned or tiled into smaller tiles, which can then be processed
    by operations such as `copy`, `arithmetic`, `reduce`, and `mma`.

    Example:
        ts_a = tensor("float32", (32, 32), "shared")
        txsa = partition_src(ts_a, tiled_copy)  # Partition the source tensor according to the tiled_copy
        txsa_p = txsa[:, :, 0]  # Extract the first slice of the partitioned tensor
        copy(txsa_p, txra_p, tiled_copy)  # Copy the partitioned tensor to another register tensor

    In the above example, alias analysis can determine that `txsa` and `txsa_p` are subtensors of the shared
    memory tensor `ts_a`.

    Process:
        The `ResolveBankConflict` pass follows these steps:

        1. Collection of Copy Operations:
           - Gather all copy operations that access shared memory.
           - Group these operations by checking if their source or destination tensors belong to the same
             shared memory tensor.

        2. Resolve Bank Conflicts:
           - Attempt to resolve bank conflicts by reordering the memory layout of the shared memory tensor.
           - Introduce a swizzle function to be performed on the original memory layout.
           - Iterate over all possible swizzle functions to find the one that minimizes bank conflicts by
           analyzing the memory access patterns of the threads within a warp.

    Swizzle Function:
        For the definition of the swizzle function, refer to the following code:
        `python/hidet/ir/cute/swizzle.py`

    Attributes:
        tensor2copys (Dict[Tensor, List[Copy]]): Mapping from tensors to their corresponding copy operations.
        copy2layout (Dict[Copy, TensorLayout]): Mapping from copy operations to their tensor layouts.
        tensor2layout (Dict[Tensor, TensorLayout]): Mapping from tensors to their final resolved tensor layouts.
    """

    def __init__(self):
        super().__init__()
        self.tensor2copys: Dict[Tensor, List[Copy]] = {}
        self.copy2layout: Dict[Copy, TensorLayout] = {}
        self.copy2conflicts: Dict[Copy, int] = {}
        self.immutable_tensors: Set[Tensor] = set()
        self.tensor2layout: Dict[Tensor, TensorLayout] = {}
        self.infer_type = TypeInfer()

    def _on_shared_tensor(self, v: Var, copy: Copy):
        tensor_info = self.get_tensor(v)
        tensor = tensor_info.tensor
        if not tensor.is_volatile():
            self.copy2layout[copy] = tensor_info.layout
            if tensor in self.tensor2copys:
                self.tensor2copys[tensor].append(copy)
            else:
                self.tensor2copys[tensor] = [copy]

    def visit_Copy(self, e: Copy):
        """
        Analyzes the copy operation and updates mappings for tensor layouts and copy operations.

        Args:
            e (Copy): The copy operation to analyze.
        """
        src = e.src
        src_ty = self.infer_type(src)
        dst = e.dst
        dst_ty = self.infer_type(dst)
        if src_ty.scope.is_shared():
            self._on_shared_tensor(src, e)
        elif dst_ty.scope.is_shared():
            self._on_shared_tensor(dst, e)

    def visit_Mma(self, e: Mma):
        """
        Note: Mma is a special case, we need to handle it separately. If
        wgmma instruction is used, the swizzle layout cannot be determined
        freely. We need to consider the swizzle determined by the wgmma
        instruction.
        """
        a = e.a
        a_ty = self.infer_type(a)
        b = e.b
        b_ty = self.infer_type(b)
        if a_ty.scope.is_shared():
            tensor_info = self.get_tensor(a)
            tensor = tensor_info.tensor
            self.immutable_tensors.add(tensor)
        elif b_ty.scope.is_shared():
            tensor_info = self.get_tensor(b)
            tensor = tensor_info.tensor
            self.immutable_tensors.add(tensor)

    def _bank_conflicts(self, phase: TensorLayout, elements_per_inst: int, banks: int, logger=None):
        """
        Determines the number of bank conflicts in the given tensor layout.

        Args:
            phase (TensorLayout): The tensor layout to analyze.
            elements_per_inst (int): Number of elements per instruction.
            banks (int): Number of memory banks.
            logger (Optional[Logger]): Logger for debugging output.

        Returns:
            int: The number of bank conflicts.
        """
        conflict_ways = 1
        bank2addr: Dict[int, Set[int]] = {}
        for tid in range(int(phase.size())):
            addr = phase(tid)
            bank_id = (addr // elements_per_inst) % banks
            if logger is not None:
                logger.debug(f"tid {tid} access shared memory bank {bank_id}")
            if bank_id in bank2addr:
                if addr not in bank2addr:
                    bank2addr[bank_id].add(addr)
                    ways = len(bank2addr[bank_id])
                    conflict_ways = max(conflict_ways, ways)
            else:
                bank2addr[bank_id] = set([addr])
        if logger is not None:
            logger.debug(f"bank conflict ways: {conflict_ways}")
        return conflict_ways

    def _resolve_banks(self, tensor: TensorBase, copys: List[Copy]):
        """
        Resolves bank conflicts for the given tensor and its associated copy operations.

        Args:
            tensor (TensorBase): The tensor to resolve bank conflicts for.
            copys (List[Copy]): The list of copy operations associated with the tensor.
        """
        BITS_PER_BYTE = 8
        BYTES_PER_BANK = 4
        NR_BANKS = 32

        need_resolve = False
        yshft_max = 0
        zshft_min = 0
        bits_max = 0

        verbose = False

        from hidet.logging import logger, setConsoleLevel, DEBUG

        origin_level = logger.level
        need_annotate = tensor in self.immutable_tensors

        if verbose:
            setConsoleLevel(DEBUG)
            logger.debug("====================================")
            logger.debug(f"resolve {tensor}")

        # cache the thread layout in this list to avoid re-invoking
        # src_tv_layout() or dst_tv_layout() repeatedly. These functions are
        # very time-consuming.
        tcache = []
        swizzle_candidates = []
        for copy in copys:
            annotations = copy.annotations
            bytes_per_phase = NR_BANKS * BYTES_PER_BANK
            assert "inst" in annotations
            inst = annotations["inst"]
            bytes_per_inst = inst.alignment
            threads_per_phase = bytes_per_phase // bytes_per_inst
            src_ty = self.infer_type(copy.src)
            dst_ty = self.infer_type(copy.dst)
            if src_ty.scope.is_shared():
                elements_per_inst = (bytes_per_inst * BITS_PER_BYTE) // src_ty.dtype.nbits
                _, tv = copy.tiled_copy.src_tv_layout()
                t, _ = canonicalize_thread_value_layout(tv)
                tcache.append(t)
            else:
                assert dst_ty.scope.is_shared()
                elements_per_inst = (bytes_per_inst * BITS_PER_BYTE) // dst_ty.dtype.nbits
                _, tv = copy.tiled_copy.dst_tv_layout()
                t, _ = canonicalize_thread_value_layout(tv)
                tcache.append(t)
            memory_layout = self.copy2layout[copy]
            phase_layout, _ = group(composition(memory_layout, t), threads_per_phase)
            banks = bytes_per_phase // bytes_per_inst
            zshft_min = max(zshft_min, elements_per_inst.bit_length())
            if verbose:
                logger.debug(f"copy {copy}, bank conlicts before resolving:")
            bank_conflicts = self._bank_conflicts(phase_layout, elements_per_inst, banks, logger if verbose else None)
            if need_annotate:
                self.copy2conflicts[copy] = bank_conflicts
            elif bank_conflicts > 1:
                need_resolve = True
                from .instruction_selection import TmaCopyInstruction

                # For TMA instructions, all four swizzling modes below are valid and can resolve
                # shared memory bank conflicts during the TMA copy itself.
                #
                # However, each swizzling function interacts differently with other copy operations
                # that use the same shared memory tensor. While all four options avoid bank conflicts
                # for the TMA transfer, they may introduce or reduce conflicts in other accesses.
                #
                # Therefore, we must evaluate all four swizzling configurations and select the one
                # that minimizes bank conflicts across all shared memory accesses—not just the TMA copy.
                if isinstance(inst, TmaCopyInstruction):
                    if len(swizzle_candidates) == 0:
                        logbits = (src_ty.dtype.nbits - 1).bit_length()
                        swizzle_candidates = [
                            Swizzle(0, 7 - logbits, 3),
                            Swizzle(1, 7 - logbits, 3),
                            Swizzle(2, 7 - logbits, 3),
                            Swizzle(3, 7 - logbits, 3),
                        ]
                yshft_max = max(int(phase_layout.cosize()).bit_length(), yshft_max)
                bits_max = max(int(phase_layout.cosize()).bit_length(), bits_max)

        if len(swizzle_candidates) == 0:
            for bits in range(1, bits_max):
                bits_mask = (1 << bits) - 1
                for zshft in range(zshft_min - 1, bits_max - bits):
                    z = bits_mask << zshft
                    for yshft in range(bits + zshft, yshft_max):
                        y = bits_mask << yshft
                        swizzle = make_swizzle(y, z)
                        swizzle_candidates.append(swizzle)

        import sys

        if need_resolve:
            min_conflicts = sys.maxsize
            best_swizzle = None
            num_copys = len(copys)
            for swizzle in swizzle_candidates:
                current_conflicts = 0
                for copy, t in zip(copys, tcache):
                    annotations = copy.annotations
                    bytes_per_phase = NR_BANKS * BYTES_PER_BANK
                    assert "inst" in annotations
                    inst = annotations["inst"]
                    bytes_per_inst = inst.alignment
                    threads_per_phase = bytes_per_phase // bytes_per_inst
                    src_ty = self.infer_type(copy.src)
                    dst_ty = self.infer_type(copy.dst)
                    if src_ty.scope.is_shared():
                        elements_per_inst = (bytes_per_inst * BITS_PER_BYTE) // src_ty.dtype.nbits
                    else:
                        assert dst_ty.scope.is_shared()
                        elements_per_inst = (bytes_per_inst * BITS_PER_BYTE) // dst_ty.dtype.nbits
                    memory_layout = self.copy2layout[copy]
                    phase_layout, _ = group(composition(memory_layout, t), threads_per_phase)
                    banks = bytes_per_phase // bytes_per_inst
                    composed_layout = ComposedTensorLayout(phase_layout, 0, swizzle)
                    bank_conflicts = self._bank_conflicts(composed_layout, elements_per_inst, banks)
                    current_conflicts += bank_conflicts
                if current_conflicts <= min_conflicts:
                    min_conflicts = current_conflicts
                    best_swizzle = swizzle
                    if min_conflicts == num_copys:
                        break

            swizzle = best_swizzle
            if verbose:
                for copy, t in zip(copys, tcache):
                    annotations = copy.annotations
                    bytes_per_phase = NR_BANKS * BYTES_PER_BANK
                    assert "inst" in annotations
                    inst = annotations["inst"]
                    bytes_per_inst = inst.alignment
                    threads_per_phase = bytes_per_phase // bytes_per_inst
                    src_ty = self.infer_type(copy.src)
                    dst_ty = self.infer_type(copy.dst)
                    if src_ty.scope.is_shared():
                        elements_per_inst = (bytes_per_inst * BITS_PER_BYTE) // src_ty.dtype.nbits
                    else:
                        assert dst_ty.scope.is_shared()
                        elements_per_inst = (bytes_per_inst * BITS_PER_BYTE) // dst_ty.dtype.nbits
                    memory_layout = self.copy2layout[copy]
                    phase_layout, _ = group(composition(memory_layout, t), threads_per_phase)
                    banks = bytes_per_phase // bytes_per_inst
                    composed_layout = ComposedTensorLayout(phase_layout, 0, swizzle)
                    logger.debug(f"copy {copy}, bank conlicts after resolving:")
                    bank_conflicts = self._bank_conflicts(composed_layout, elements_per_inst, banks, logger)
                logger.debug("resolve completed")
                logger.debug("====================================")
                setConsoleLevel(origin_level)
            self.tensor2layout[tensor] = ComposedTensorLayout(tensor.layout, 0, swizzle)

    def solve(self, func: Function):
        """
        Perform the bank conflict resolution pass on the given function.

        Args:
            func (Function): The function to process.

        Returns:
            Dict[Tensor, TensorLayout]: The mapping from tensors to their resolved tensor layouts.
        """
        self.visit(func)

        for tensor, copys in self.tensor2copys.items():
            self._resolve_banks(tensor, copys)

        return self.tensor2layout, self.copy2conflicts


class ApplySharedMemoryLayoutUpdate(IRRewriter):
    """
    ApplySharedMemoryLayoutUpdate applies the resolved memory layouts to the IR.

    This class updates the IR to reflect the new memory layouts determined by the
    ResolveBankConflict pass.

    Attributes:
        tensor2layout (Dict[TensorBase, LayoutBase]): The mapping from tensors to their new layouts.
        old2new (Dict[Var, Var]): The mapping from old variables to new variables.
    """

    def __init__(self, tensor2layout: Dict[TensorBase, LayoutBase], copy2conflicts: Dict[Copy, int]):
        """
        Initializes the ApplySharedMemoryLayoutUpdate instance.

        Args:
            tensor2layout (Dict[TensorBase, LayoutBase]): The mapping from tensors to their new layouts.
        """
        super().__init__(use_memo=False)
        self.tensor2layout: Dict[TensorBase, LayoutBase] = tensor2layout
        self.copy2conflicts: Dict[Copy, int] = copy2conflicts
        self.old2new: Dict[Var, Var] = {}
        self.infer_type = TypeInfer()

    def visit_Var(self, v: Var):
        """
        Visits and possibly updates a variable.

        Args:
            v (Var): The variable to visit.

        Returns:
            Var: The updated or original variable.
        """
        if v in self.old2new:
            return self.old2new[v]
        return super().visit_Var(v)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        """
        Visits and possibly updates a declaration statement.

        Args:
            stmt (DeclareStmt): The declaration statement to visit.

        Returns:
            DeclareStmt: The updated or original declaration statement.
        """
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = self.visit(call.op)
            if op is call.op:
                return stmt
            else:
                v = stmt.var
                init = op.make_call()
                v = var(v.hint, self.infer_type(init))
                self.old2new[stmt.var] = v
                return DeclareStmt(v, init, stmt.is_static, stmt.scope)
        return super().visit_DeclareStmt(stmt)

    def visit_Tensor(self, e: Tensor):
        """
        Visits and possibly updates a tensor.

        Args:
            e (Tensor): The tensor to visit.

        Returns:
            Tensor: The updated or original tensor.
        """
        dtype = self.visit(e.dtype)
        if e in self.tensor2layout:
            layout = self.tensor2layout[e]
        else:
            layout = self.visit_Layout(e.layout)
        if dtype is e.dtype and layout is e.layout:
            return e
        else:
            assert dtype is e.dtype
            return e.reforward([], attrs_update={"layout": layout})

    def visit_TensorView(self, e: TensorView):
        """
        Visits and possibly updates a tensor view.

        Args:
            e (TensorView): The tensor view to visit.

        Returns:
            TensorView: The updated or original tensor view.
        """
        x = self.visit(e.x)
        if e in self.tensor2layout:
            layout = self.tensor2layout[e]
        else:
            layout = self.visit_Layout(e.layout)
        if x is e.x and layout is e.layout:
            return e
        else:
            return e.reforward([x], attrs_update={"layout": layout})

    def visit_Copy(self, e: Copy):
        if e in self.copy2conflicts:
            bank_conflicts = self.copy2conflicts[e]
            args = [self.visit(arg) for arg in e.args]
            annotations: Dict[str, CConst] = {}
            annotations["bank_conflicts"] = bank_conflicts
            return e.reforward(args, annotations_update=annotations)
        return super().visit_Copy(e)


class ResolveBankConflictPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        solver = ResolveBankConflict()
        solution = solver.solve(func)

        rewriter = ApplySharedMemoryLayoutUpdate(*solution)
        return rewriter(func)


def resolve_bank_conflict_pass() -> FunctionPass:
    return ResolveBankConflictPass()
