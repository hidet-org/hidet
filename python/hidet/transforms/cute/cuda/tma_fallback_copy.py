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
TMA Fallback Copy Pass

This module implements a pass that handles fallback from TMA (Tensor Memory Access) instructions to regular cp_async
instructions when TMA cannot be used due to layout constraints. This maintains a unified programming interface while
allowing the compiler to automatically select the appropriate instruction type.

In the Hexcute design, the instruction selection pass chooses the optimal PTX instruction for each copy operation.
However, when some code is written to use TMA instructions intentionally, but the layouts prevent the TMA
engine from copying the data, this pass falls back to normal cp_async instructions. This approach reduces
software complexity by maintaining a single codebase instead of maintaining separate variations for different
instruction types (like CUTLASS's separate TMA and cp_async warp-specialized GEMM kernels).

The pass works in three main steps:
1. Collects copy operations that cannot use TMA instructions (those involving mbarrier without mask operations)
2. Deducts transaction count in mbarrier_arrive operations since data is not copied through TMA engine
3. Generates masks for copy operations using cp_async instructions
"""
from typing import Dict, List, Tuple, Optional

from hidet.ir.module import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.transforms.base import FunctionPass
from hidet.ir.tools import TypeInfer
from hidet.ir.builders import StmtBuilder

from hidet.ir.dtypes import i32
from hidet.ir.expr import Expr, Var, var, is_constant
from hidet.ir.stmt import DeclareStmt, EvaluateStmt, IfStmt

from hidet.ir.cute import TensorLayout, composition, flatten, product
from hidet.ir.cute.expr import CallOp
from hidet.ir.cute.ops import Copy, MBarriers, TensorView, TensorBase, MBarrierArrive, Partition, SubTensor, mask
from hidet.ir.cute.int_tuple import product_each


from hidet.transforms.cute.analysis import TensorInfo, TensorAliasAnalysis
from hidet.transforms.cute.cuda.instruction_selection import TmaCopyInstruction, MaskUsageMarker, MaskAnnotation


class TmaFallbackCollector(IRVisitor):
    """Collects information about copy operations that need to fall back from TMA to cp_async.

    This visitor analyzes the IR to identify:
    1. Copy operations that involve mbarriers but don't use TMA instructions
    2. The relationship between mbarriers and shared memory tensors
    3. The transaction counts for each copy operation
    """

    def __init__(self, var2tensor: Dict[Var, TensorInfo]):
        super().__init__()
        self.infer_type = TypeInfer()
        self.var2tensor: Dict[Var, TensorInfo] = var2tensor
        self.mbarrier_to_shared_tensor_and_tx: Dict[MBarriers, Dict[TensorBase, int]] = {}
        self.global_tensor_to_copies: Dict[TensorView, List[Copy]] = {}

    def visit_Copy(self, e: Copy):
        """Analyzes a Copy operation to determine if it needs to fall back from TMA to cp_async.

        Args:
            e: The Copy operation to analyze
        """
        if e.mbarrier is not None:
            tensor_info = self.var2tensor[e.mbarrier]
            bar_tensor = tensor_info.tensor
            if isinstance(bar_tensor, MBarriers):
                annotations = e.annotations
                assert "inst" in annotations
                inst = annotations["inst"]
                if not isinstance(inst, TmaCopyInstruction):
                    src_ty = self.infer_type(e.src)
                    dst_ty = self.infer_type(e.dst)
                    if src_ty.scope.is_shared():
                        shared_tensor = self.var2tensor[e.src].tensor
                        shared_layout = self.var2tensor[e.src].layout
                    elif dst_ty.scope.is_shared():
                        shared_tensor = self.var2tensor[e.dst].tensor
                        shared_layout = self.var2tensor[e.dst].layout
                    else:
                        raise ValueError(f"No shared memory tensor involved in the copy {e} with barrier {bar_tensor}")
                    if src_ty.scope.is_global():
                        global_tensor = self.var2tensor[e.src].tensor
                    elif dst_ty.scope.is_global():
                        global_tensor = self.var2tensor[e.dst].tensor
                    else:
                        raise ValueError(f"No global memory tensor involved in the copy {e} with barrier {bar_tensor}")
                    shared_layout = composition(shared_layout, TensorLayout(e.tiled_copy.shape))
                    flat_shape = flatten(shared_layout.shape_tuple)
                    flat_stride = flatten(shared_layout.stride_tuple)
                    flat_shape = tuple(s if d != 0 else 1 for s, d in zip(flat_shape, flat_stride))
                    size = product(flat_shape)
                    transaction_count = src_ty.dtype.nbits * size // 8
                    if bar_tensor not in self.mbarrier_to_shared_tensor_and_tx:
                        self.mbarrier_to_shared_tensor_and_tx[bar_tensor] = {shared_tensor: transaction_count}
                    else:
                        if shared_tensor not in self.mbarrier_to_shared_tensor_and_tx[bar_tensor]:
                            self.mbarrier_to_shared_tensor_and_tx[bar_tensor][shared_tensor] = transaction_count
                        else:
                            tx = self.mbarrier_to_shared_tensor_and_tx[bar_tensor][shared_tensor]
                            if transaction_count != tx:
                                raise ValueError(
                                    f"Shared tensor {shared_tensor} has different transaction count "
                                    f"{transaction_count} and {tx} for the same mbarrier {bar_tensor}"
                                )
                    if global_tensor not in self.global_tensor_to_copies:
                        self.global_tensor_to_copies[global_tensor] = [e]
                    else:
                        self.global_tensor_to_copies[global_tensor].append(e)

    def collect(self, func: Function):
        """Collects all copy operations that need to fall back from TMA to cp_async.

        Args:
            func: The function to analyze

        Returns:
            Tuple containing:
            - Dictionary mapping global tensors to their copy operations
            - Dictionary mapping mbarriers to their associated shared tensors and transaction counts
        """
        self.visit(func)
        return self.global_tensor_to_copies, self.mbarrier_to_shared_tensor_and_tx


class TmaFallbackCopyRewriter(IRRewriter):
    """Rewrites the IR to implement the TMA fallback functionality.

    This rewriter:
    1. Generates appropriate masks for cp_async operations
    2. Adjusts mbarrier transaction counts
    3. Modifies copy operations to use cp_async instead of TMA
    """

    def __init__(
        self,
        var2tensor: Dict[Var, TensorInfo],
        global_tensor_to_copies: Dict[TensorView, List[Copy]],
        mbarrier_to_shared_tensor_and_tx: Dict[MBarriers, Dict[TensorBase, int]],
    ):
        super().__init__()
        self.var2tensor = var2tensor
        self.mbarrier_to_shared_tensor_and_tx = mbarrier_to_shared_tensor_and_tx
        self.global_tensor_to_copies: Dict[TensorView, List[Copy]] = global_tensor_to_copies
        self.global_tensor_to_mask_cache: Dict[TensorView, Dict[str, Tuple[Var, Var]]] = {}
        self.var2coordinates: Dict[Var, Tuple[int, ...]] = {}
        self.infer_type = TypeInfer()

    def _declare(
        self, sb: StmtBuilder, v: Optional[Var] = None, hint: Optional[Expr] = None, init: Optional[Expr] = None
    ):
        """Helper method to declare variables in the statement builder.

        Args:
            sb: The statement builder
            v: Optional variable to declare
            hint: Optional hint for variable name
            init: Optional initial value

        Returns:
            The declaration statement
        """
        if v is None:
            assert init is not None and hint is not None
            v_ty = self.infer_type(init)
            v = var(hint, v_ty)
            return sb.declare(v, init)
        else:
            assert hint is None
            if init is not None:
                return sb.declare(v, init)
            else:
                return sb.declare(v)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        """Processes declaration statements, handling tensor views and generating masks.

        Args:
            stmt: The declaration statement to process

        Returns:
            The processed declaration statement
        """
        call = stmt.init
        if isinstance(call, CallOp):
            op = call.op
            if isinstance(op, TensorView):
                if op in self.global_tensor_to_copies:
                    # Step 1: Record the tile coordinates for this tensor view
                    self.var2coordinates[stmt.var] = op.tile_coords

                    tensor = op
                    # Step 3: Get the global layout and shape information
                    global_layout = tensor.layout
                    tile_coords = tensor.tile_coords
                    global_shape = product_each(global_layout.shape_tuple)

                    # Step 4: Initialize or get the mask cache for this tensor
                    if tensor not in self.global_tensor_to_mask_cache:
                        self.global_tensor_to_mask_cache[tensor] = {}
                    mask_cache = self.global_tensor_to_mask_cache[tensor]

                    # Step 5: Create a statement builder for generating mask declarations
                    sb = StmtBuilder()
                    var_ = self.visit(stmt.var)
                    init = self.visit(stmt.init)
                    sb.declare(var_, init)

                    # Step 6: Process each copy operation that needs mask generation
                    for copy in self.global_tensor_to_copies[op]:
                        # skip if the copy has a mask
                        if copy.mask is not None:
                            continue

                        tiled_copy = copy.tiled_copy
                        tiled_copy_str = tiled_copy.str_indented()
                        # Skip if we've already generated masks for this tiled copy
                        if tiled_copy_str in mask_cache:
                            continue

                        # Step 7: Get tile shape and verify dimensions match
                        tile_shape = tiled_copy.shape
                        assert len(global_shape) == len(tile_shape) and len(tile_coords) == len(tile_shape)

                        # Step 8: Calculate mask extents for both full and residue cases
                        mask_extents = []
                        mask_residue_extents = []
                        # Process all dimensions except the last one
                        for gs, ts, tc in zip(global_shape[:-1], tile_shape[:-1], tile_coords[:-1]):
                            if is_constant(gs) and gs % ts == 0:
                                # If global shape is divisible by tile shape, use tile shape
                                mask_extents.append(i32(ts))
                                mask_residue_extents.append(i32(ts))
                            else:
                                # Otherwise, use remaining elements from current coordinate
                                mask_extents.append(gs - tc)
                                mask_residue_extents.append(gs - tc)

                        # Step 9: Handle the last dimension
                        mask_extents.append(i32(tile_shape[-1]))
                        if is_constant(global_shape[-1]) and global_shape[-1] % tile_shape[-1] == 0:
                            # If last dimension is divisible by tile shape, use tile shape
                            mask_residue_extents.append(i32(tile_shape[-1]))
                        else:
                            # Otherwise, use remainder of last dimension
                            mask_residue_extents.append(global_shape[-1] % tile_shape[-1])

                        # Step 10: Generate full mask if any extent is constant
                        if any(not is_constant(e) for e in mask_extents):
                            mask_full = self._declare(sb, hint="mask_full", init=mask(tiled_copy, mask_extents))
                        else:
                            mask_full = None

                        # Step 11: Generate residue mask if any extent is constant
                        if any(not is_constant(e) for e in mask_residue_extents):
                            mask_residue = self._declare(
                                sb, hint="mask_residue", init=mask(tiled_copy, mask_residue_extents)
                            )
                        else:
                            mask_residue = None

                        # Step 12: Cache the generated masks
                        mask_cache[tiled_copy_str] = (mask_full, mask_residue)
                    return sb.finish()
            elif isinstance(op, Partition):
                src_coords = self.var2coordinates.get(op.x, None)
                if src_coords is not None:
                    self.var2coordinates[stmt.var] = src_coords
            elif isinstance(op, SubTensor):
                src_coords = self.var2coordinates.get(op.x, None)
                if src_coords is not None:
                    x_ty = self.infer_type(op.x)
                    tile_shape = x_ty.layout[0].shape_tuple
                    tile_shape = product_each(tile_shape)
                    rank = len(tile_shape)
                    # TODO: FIXME currently it's a hack
                    rank = 2
                    crd_layout = TensorLayout(x_ty.layout[rank:].shape_tuple)
                    crd = op.coord[rank:]
                    self.var2coordinates[stmt.var] = src_coords[:-1] + [
                        src_coords[-1] + crd_layout(crd) * tile_shape[-1]
                    ]
        return super().visit_DeclareStmt(stmt)

    def visit_MBarrierArrive(self, e: MBarrierArrive):
        """Adjusts mbarrier arrive operations to account for non-TMA copy operations.

        Args:
            e: The mbarrier arrive operation to process

        Returns:
            The processed mbarrier arrive operation
        """
        mbar = self.visit(e.mbarrier)
        count = self.visit(e.count)
        bar_tensor = self.var2tensor[e.mbarrier].tensor
        if bar_tensor in self.mbarrier_to_shared_tensor_and_tx:
            shared_tensor2tx = self.mbarrier_to_shared_tensor_and_tx[bar_tensor]
            total_transaction_count = 0
            for _, tx in shared_tensor2tx.items():
                total_transaction_count += tx
            assert total_transaction_count <= count
            count = count - total_transaction_count
            annotations = e.annotations
            annotations['tma_fallback_copy'] = True
            return e.reforward([mbar, count], annotations_update=annotations)
        return super().visit_MBarrierArrive(e)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        """Processes evaluate statements, handling copy operations and their masks.

        Args:
            stmt: The evaluate statement to process

        Returns:
            The processed evaluate statement
        """
        # Step 1: Extract the expression from the evaluate statement
        expr = stmt.expr
        if isinstance(expr, CallOp):
            op = expr.op
            # Step 2: Check if this is a Copy operation that needs processing
            if isinstance(op, Copy):
                # Step 3: Determine memory scopes of source and destination
                src_ty = self.infer_type(op.src)
                dst_ty = self.infer_type(op.dst)

                # Step 4: Identify the global tensor and get its coordinates
                if src_ty.scope.is_global():
                    # Case: Source is in global memory
                    global_tensor = self.var2tensor[op.src].tensor
                    last_dim_coords = self.var2coordinates.get(op.src, None)
                elif dst_ty.scope.is_global():
                    # Case: Destination is in global memory
                    global_tensor = self.var2tensor[op.dst].tensor
                    last_dim_coords = self.var2coordinates.get(op.dst, None)
                else:
                    global_tensor = None

                # Step 5: Check if this copy operation needs TMA fallback handling
                if global_tensor in self.global_tensor_to_copies and op.mask is None:
                    # Step 6: Process source and destination operands
                    src = self.visit(op.src)
                    dst = self.visit(op.dst)

                    # Step 7: Get the pre-generated masks from cache
                    tiled_copy_str = op.tiled_copy.str_indented()
                    mask_full, mask_residue = self.global_tensor_to_mask_cache[global_tensor][tiled_copy_str]

                    # Step 8: Get shape information for boundary checking
                    global_shape = product_each(global_tensor.layout.shape_tuple)
                    tile_shape = op.tiled_copy.shape
                    last_dim_extent = global_shape[-1]
                    assert last_dim_coords is not None
                    last_dim_coord = last_dim_coords[-1]

                    # Step 9: Handle different mask cases for the copy operation
                    if mask_full is None and mask_residue is None:
                        # Case 9.1: No masks needed (regular copy)
                        return EvaluateStmt(op.reforward([src, dst]).make_call())
                    elif mask_full is not None and mask_residue is None:
                        # Case 9.2: Only full mask needed (partial boundary)
                        return EvaluateStmt(op.reforward([src, dst, mask_full]).make_call())
                    elif mask_residue is not None:
                        # Case 9.3: Both full and residue masks needed (complex boundary)
                        assert mask_full is not None
                        # Step 10: Calculate boundary condition for last dimension
                        cond = last_dim_coord < last_dim_extent - tile_shape[-1]
                        # Step 11: Generate conditional copy with appropriate mask
                        return IfStmt(
                            cond,  # If not at boundary: use full mask
                            EvaluateStmt(op.reforward([src, dst, mask_full]).make_call()),
                            EvaluateStmt(
                                op.reforward([src, dst, mask_residue]).make_call()
                            ),  # At boundary: use residue mask
                        )
        # Step 12: Fall back to default handling for non-Copy operations
        return super().visit_EvaluateStmt(stmt)


class TmaFallbackCopyPass(FunctionPass):
    """A pass that implements the TMA fallback functionality.

    This pass coordinates the collection and rewriting of copy operations that need to fall back
    from TMA to cp_async instructions.
    """

    def __init__(self):
        super().__init__()

    def process_func(self, func: Function):
        """Processes a function to implement the TMA fallback functionality.

        Args:
            func: The function to process

        Returns:
            The processed function with TMA fallback implemented
        """
        var2tensor = TensorAliasAnalysis().analyze(func)
        collector = TmaFallbackCollector(var2tensor)
        global_tensor_to_copies, mbarrier_to_shared_tensor_and_tx = collector.collect(func)

        rewriter = TmaFallbackCopyRewriter(var2tensor, global_tensor_to_copies, mbarrier_to_shared_tensor_and_tx)
        func = rewriter(func)

        marker = MaskUsageMarker()
        mask2user = marker.mark(func)
        mask_annotation = MaskAnnotation(mask2user)
        func = mask_annotation(func)

        return func


def tma_fallback_copy_pass():
    """Creates a new instance of the TMA fallback copy pass.

    Returns:
        A new TmaFallbackCopyPass instance
    """
    return TmaFallbackCopyPass()
