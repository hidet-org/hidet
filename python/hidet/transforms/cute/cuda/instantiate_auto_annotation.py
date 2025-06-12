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
# pylint: disable=too-many-lines
"""
Module for Task Mapping and Layout Inference
============================================

This module infers the task mapping and the shared memory layouts for each tensor in the IR sequence. The process
is divided into four main steps:

1. **Infer Logical Shapes**:
    - The logical shape is the shape of each tile in the tensor.
    - The `InferLogicalShape` pass iterates over the IR sequence, collecting operators with unresolved shapes and
      resolving them.
    - A dictionary maps each variable to its logical shape internally.
    - If any inconsistency occurs during inference, an error is raised.

    Example:
    ```python
    a = tensor(float16, (128, 128), 'global')  # shape = (128, 128)
    b = partition_src(a, auto_copy())          # shape = (128, 32, 4)
    c = tensor(float16, (128, 128), 'shared')  # shape = (128, 128)
    d = partition_dst(c, auto_copy())          # shape = (128, 32, 4)
    copy(c[:, :, 0], d[:, :, 0], auto_copy((128, 32)))
    ```
    The logical shape of each tensor is annotated in the comments. The `InferLogicalShape` pass will iterate over
    the IR sequence and collect all the operators with unresolved shapes and those with resolved shapes. This pass
    maintains a dictionary that maps each variable to its logical shape internally. The unresolved operators are
    then visited, and their logical shapes are inferred. This process is repeated until all the operators have
    resolved shapes. If any inconsistency occurs, an error will be raised.

2. **Infer Logical Layouts**:
    - The logical layout is the layout of each tile, determining if a tensor is a vector or a matrix by analyzing
      strides.
    - The `InferLogicalLayout` pass, similar to `InferLogicalShape`, iterates over the IR sequence to resolve
      layouts.
    - A dictionary maps each variable to its logical layout internally.
    - If any inconsistency occurs during inference, an error is raised.

    Example:
    ```python
    a = tensor(float16, TensorLayout((128, 128), (1, 0)), 'global')  # layout = (128, 128):(1, 0), this tensor is
                                                                     # a column vector
    b = tensor(float16, TensorLayout((128, 128), (1, 128)), 'global') # layout = (128, 128):(1, 128), this tensor'
                                                                      # is a matrix
    ```
    The `InferLogicalLayout` pass is similar to the `InferLogicalShape` pass. It will iterate over the IR sequence
    and collect all the operators with unresolved layouts and those with resolved layouts. This pass maintains a
    dictionary that maps each variable to its logical layout internally. The unresolved operators are then visited,
    and their logical layouts are inferred. This process is repeated until all the operators have resolved layouts.
    If any inconsistency occurs, an error will be raised.

    Note that the logical shapes and layouts treat the tensors as a whole without considering how these tensors are
    distributed across the hardware units. The final step of this module is to infer the task mappings, the layouts
    representing how the tiles are distributed across the hardware units, and the memory layouts that lead to the
    most efficient memory access patterns.

    Note that the logical shapes and layouts will be used as context information for the task mapping and memory
    layout inference.

3. **Infer Task Mappings and Memory Layouts**:
    - The task mapping of a copy operation is defined by a `TiledCopy` object, which contains the TV layouts for
      both the source and destination tensors.
    - The task mapping of an `mma` operation is defined by a `TiledMma` object, which contains the TV layouts for
      the `a`, `b`, `c`, and `d` operands.
    - To find the optimal task mapping, we need to determine the TV layouts of the tensor nodes in the IR sequence.
    - The `ResolveAuto` pass iterates over the IR sequence and collects all the operators with unresolved task
      mappings and those with resolved task mappings. This pass maintains a dictionary that maps each tensor
      variable to its TV layout. The unresolved operators are then visited, and the constraints of the operators
      are applied to infer the TV layouts. Different operators have different constraints.

    For example, we assume an instruction can be used to execute the copy operation, and this instruction can be
    described by its source and destination TV layouts, `p` and `q`, respectively. Then, we assume the source and
    destination TV layouts of the copy operation are `f` and `g`, respectively. The constraints of the copy
    operation can be formalized as follows:
    ```
    f o p^{-1} = g o q^{-1}
    ```
    When applying the constraints, we will try to rewrite the above constraints as follows:
    ```
    f = g o q^{-1} o p
    ```
    where `p` and `q` are known, and we assume `g` has been resolved in the previous steps. Then, we can infer the
    TV layout of `f` by applying the constraints. Note that a copy operation may have multiple instructions
    available to execute it, so we could employ DFS to find all valid variants and choose the best one through a
    cost model. Alternatively, we could use some simple heuristics or beam search to prune the search space.

    Similarly, we can build the constraints for other operators and infer the TV layouts for them. We list the
    constraints for some operators below:
    - `Mma`: We assume the instruction can be used to execute the `mma` operation is described by its TV layouts
      `p`, `q`, `r`, and `s`, which correspond to the operands `a`, `b`, `c`, and `d`, respectively. The TV layouts
      of the `mma` operation are `a`, `b`, `c`, and `d`, respectively. To formalize the constraints, we first build
      three mapping functions:
      ```
      f1 = a o p^{-1} : (inst_m, inst_k) -> (tile_m, tile_k)
      f2 = b o q^{-1} : (inst_n, inst_k) -> (tile_n, tile_k)
      f3 = c o r^{-1} : (inst_m, inst_n) -> (tile_m, tile_n)
      ```
      where inst_m, inst_n, inst_k are coordinates of the instruction, and tile_m, tile_n, tile_k are coordinates
      of the tile to be copied. Then, we could extract the m, n, k modes from the three mapping functions:
      ```
      f1 = (f1_m, f1_k)
      f2 = (f2_n, f2_k)
      f3 = (f3_m, f3_n)
      ```
      Then, we could build the constraints as follows:
      ```
      f1_m = f3_m
      f2_n = f3_n
      f1_k = f2_k
      ```
      We can then infer the TV layouts of `a`, `b`, `c`, and `d` by applying the constraints. Intuitively, the
      above constraints should be satisfied because the shapes of the matrices A, B, and C should be compatible for
      matrix multiplication. However, for tensor core programming, we replace the normal shapes with hierarchical TV
      layouts.

    - `Elementwise`: The constraints for the elementwise operation are simple. We assume the TV layouts of the input
      tensors are `a1`, `a2`, ..., `an`. We can build the constraints as follows:
      ```
      a1.shape = a2.shape = ... = an.shape
      ```
      The non-zero strides of these TV layouts should be identical.

4. **Infer Memory Layouts**:
    - For each tensor residing in shared memory, multiple copy operations might access it. Each copy operation has
      its own memory constraints. The inference of memory layouts is to find a memory layout substitution that
      satisfies all the constraints. This process is similar to the unification process in type inference.

    Example:
    ```python
    a = tensor(float16, (128, 128), 'global')  # shape = (128, 128)
    b = partition_src(a, auto_copy())          # shape = (128, 32, 4)
    c = tensor(float16, (128, 128), 'shared')  # shape = (128, 128)
    d = partition_dst(c, auto_copy())          # shape = (128, 32, 4)
    copy(b[:, :, 0], d[:, :, 0], auto_copy((128, 32)))  # copy_1
    e = partition_src(c, auto_copy())          # shape = (128, 16, 8)
    f = tensor(float16, (128, 32), "register") # shape = (128, 32)
    g = partition_dst(f, auto_copy())          # shape = (128, 16, 2)
    copy(e[:, :, 0], g[:, :, 0], auto_copy((128, 16)))  # copy_2
    ```

    Before inferring the memory layouts, we only know the shapes of the shared memory tensor `c` without knowing the
    strides, so the memory layout of `c` would look like (128, 128):(v, v). The strides remain undetermined. The
    memory layout would be refined every time we apply the constraints of the copy operations.

    For example, consider the constraints of the `copy_1` operation. Assume we have 32 threads in a thread block. In
    the task mapping inference, the TV layout of `d[:, :, 0]` is inferred as
    ```
    p = ((4, 8), (8, 16)):((1024, 1), (128, 8))
    ```
    and the `copy_1` operation is executed by a `cp_async` instruction. This instruction requires the memory address
    to be aligned to 16 bytes, which means the result of composing the memory layout of `c` and value mode,
    (8, 16):(128, 8), should be like (8, ...):(1, ...). This further indicates the memory layouts should be in the
    following form:
    ```
    (128, (8, 16)):(v, (1, v))
    ```
    Thus, the memory layout is refined after applying the constraints of the `copy_1` operation. Similarly, we can
    apply the constraints of the `copy_2` operation to refine the memory layout of `c`. If any conflict occurs, this
    may indicate the constraints are not satisfiable, and the solution will be backtracked. We will keep doing this
    process until all the constraints are satisfied. Then, we use a heuristic to determine the remaining undetermined
    strides.

Note:
-----
    - The task mapping and layout inference ideally can help us find all valid configurations of instructions and
      layouts for the tensor program. For example, for GEMM kernel, we can find the variants like:
        mma: mma.m16n8k16, global2shared: cp_async, shared2register: ldmatrix
        mma: mma.m16n8k16, global2shared: cp_async, shared2register: lds128
      ...

Classes:
--------
- MarkUnresolved(IRVisitor): Identifies unresolved operations in a function and marks them.
- InferLogicalShape(IRVisitor): Infers the logical shapes of variables in a function.
- InferLogicalLayout(IRVisitor): Infers the logical layouts of variables in a function.
- InferContext: Stores inference context, including variable layouts, tensor information, and solution mappings.
- MemoryConstraint: Represents memory constraints for a tensor.
- InferResult: Stores inference results, including logical encodings and memory constraints.
- InferRule: Represents a single inference rule.
- InferRules: Manages a collection of inference rules.
- MemoryConstraintsUnifier: Unifies memory constraints to find a common memory layout.
- StackFrame: Represents a single frame in the state stack during inference.
- ResolveAuto(IRVisitor): Resolves automatic annotations in a function using inferred information.
- MaterializeAuto(IRRewriter): Materializes the inferred layouts and operations in a function.
- InstantiateAutoAnnotationPass(FunctionPass): Applies the auto-annotation pass to a function.

Functions:
----------
- infer_context(var2layout, var2tensor, solution): Creates an inference context.
- constraint(tensor, memory_constraints): Creates a memory constraint.
- infer_result(*encs, memory_constraint=None): Creates an inference result.
- register_infer_rules(op_cls): Decorator to register inference rules for an operation class.
- get_infer_rules(op_or_op_cls): Retrieves the inference rules for a given operation or operation class.
- forward(op, args, ctx, input_vars=None, output_var=None): Infers logical encoding by forwarding it to the output.
- make_constraint(inputs, output, op, name): Creates a constraint.
- is_surjective(a): Checks if a tensor layout is surjective.
- infer_memory_constraints(ctx, tensor_info, value, elements_per_inst): Infers memory constraints for a tensor.
- validate_alignment(value, memory, elements_per_inst): Validates the alignment of a memory layout.
- instantiate_auto_annotation_pass(): Creates an instance of the auto-annotation pass.
"""
from typing import Type, Tuple, List, Dict, Union, Optional, Callable

from hidet.ir.type import DataType
from hidet.ir.expr import Var, Expr, var, is_constant
from hidet.ir.tools import TypeInfer, infer_type
from hidet.ir.functors import IRVisitor, IRRewriter

from hidet.ir.cute.expr import Op, CallOp
from hidet.ir.cute.type import TiledTensorType, LogicalEncoding, logical_encoding

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
from hidet.ir.stmt import DeclareStmt, AssignStmt

from hidet.ir.cute.layout import (
    CopyAtom,
    ThrValAtom,
    filter,
    filter_lo_hi,
    remove_lo_hi,
    group,
    compact_col_major,
    slice_and_offset,
)
from hidet.ir.cute import (
    TensorLayout,
    ComposedTensorLayout,
    TiledTensorLayout,
    auto_layout,
    is_auto_layout,
    max_common_vector,
    flatten,
    is_tuple,
    rank,
    prefix_product,
)
from hidet.ir.cute.ops import (
    TensorBase,
    Tensor,
    TensorView,
    Partition,
    PartitionA,
    PartitionB,
    PartitionSrc,
    PartitionDst,
    Copy,
    Mask,
    Mma,
    Rearrange,
    Arithmetic,
    SubTensor,
    Reduce,
    Broadcast,
    Transpose,
    Atomic,
    MBarriers,
)
from hidet.ir.cute.algorithm import TiledCopy, TiledMma, is_auto_copy, is_auto_mma
from hidet.ir.cute import coalesce, composition, left_inverse, make_layout, product_each
from hidet.transforms.cute.analysis import TensorAliasAnalysis, TensorInfo

from hidet.logging import logger, stderr_handler, setConsoleLevel, DEBUG
from hidet.utils.py import gcd

from .instruction_selection import (
    CopyInstruction,
    memory_instructions,
    MmaInstruction,
    get_mma_instructions,
    expr_to_buffer,
)
from .tma_layout_utils import (
    get_last_dim_strides,
    coalesce_per_dim,
    common_reshape_per_dim,
    coalesce_gmem_shape_and_smem_shape,
    split_shapes,
    sort_dims,
    make_contiguous_stride,
    construct_memory_constraint,
)


verbose = False


ShapeVar = Union[int, Expr]


class MarkUnresolved(IRVisitor):
    def __init__(self):
        super().__init__()
        self.ops_unresolved: List[Op] = []
        self.ops_resolved: List[Op] = []
        self.op2vars: Dict[Op, List[Var]] = {}
        self.infer_type = TypeInfer()

    def mark(self, func: Function):
        self.visit(func)

        return self.ops_resolved, self.ops_unresolved, self.op2vars

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = stmt.var
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            self.visit(call)
            op = call.op
            if op in self.op2vars:
                self.op2vars[op].append(v)
            else:
                self.op2vars[op] = [v]

    def visit_AssignStmt(self, stmt: AssignStmt):
        v = stmt.var
        if isinstance(stmt.value, CallOp):
            call = stmt.value
            self.visit(call)
            op = call.op
            if op in self.op2vars:
                self.op2vars[op].append(v)
            else:
                self.op2vars[op] = [v]

    def visit_Copy(self, e: Copy):
        self.visit(e.src)
        self.visit(e.dst)
        if e.mask is not None:
            self.visit(e.mask)
        if e.mbarrier is not None:
            self.visit(e.mbarrier)
        if is_auto_copy(e.tiled_copy):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.src, e.dst]

    def visit_Tensor(self, e: Tensor):
        if is_auto_layout(e.layout):
            assert e.scope.is_register() or e.scope.is_shared()
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)

    def visit_TensorView(self, e: TensorView):
        if is_auto_layout(e.layout):
            assert e.scope.is_register() or e.scope.is_shared()
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_PartitionA(self, e: PartitionA):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_mma(e.tiled_mma) or is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_PartitionB(self, e: PartitionB):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_mma(e.tiled_mma) or is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_PartitionSrc(self, e: PartitionSrc):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_copy(e.tiled_copy) or is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_PartitionDst(self, e: PartitionDst):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_copy(e.tiled_copy) or is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Mma(self, e: Mma):
        self.visit(e.d)
        self.visit(e.a)
        self.visit(e.b)
        self.visit(e.c)
        if is_auto_mma(e.tiled_mma):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.d, e.a, e.b, e.c]

    def visit_SubTensor(self, e: SubTensor):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Rearrange(self, e: Rearrange):
        self.visit(e.x)
        if is_auto_layout(e.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Arithmetic(self, e: Arithmetic):
        for i in e.inputs:
            self.visit(i)
        input_tys = [self.infer_type(i) for i in e.inputs]
        if any(is_auto_layout(input_ty.layout) for input_ty in input_tys):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [i for i in e.inputs]

    def visit_Reduce(self, e: Reduce):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Broadcast(self, e: Broadcast):
        self.visit(e.x)
        self.visit(e.target)
        x_ty = self.infer_type(e.x)
        tgt_ty = self.infer_type(e.target)
        if is_auto_layout(x_ty.layout) or is_auto_layout(tgt_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x, e.target]

    def visit_Transpose(self, e: Transpose):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Atomic(self, e: Atomic):
        # Theoretically, it's fine that if the atomic operation is not
        # incoorperated in the task mapping inference system, because the
        # output tensor is always a global memory tensor, and this operation
        # doesn't impose any layout constraints on its input tensor node.
        # Currently, we just implement a basic version of inference for atomic
        # operation. We could extend it to more complicated situations later
        # TODO:
        self.visit(e.src)
        src_ty = self.infer_type(e.src)
        if is_auto_layout(src_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.src, e.dst]


class InferLogicalShape(IRVisitor):
    def __init__(self, ops_resolved: List[Op], ops_unresolved: List[Op], op2vars: Dict[Op, List[Var]]):
        super().__init__(False)
        self.var2shape: Dict[Var, Tuple[ShapeVar]] = {}

        self.ops_resolved: List[Op] = ops_resolved
        self.ops_unresolved: List[Op] = ops_unresolved.copy()
        self.op2vars: Dict[Op, List[Var]] = op2vars
        self.infer_type = TypeInfer()

    def _shape_resolved(self, shape: Tuple[ShapeVar]):
        return shape is not None and all(is_constant(s) for s in shape)

    def _update_var_shape(self, v: Var, shape: Tuple[ShapeVar]):
        if shape is not None:
            self.var2shape[v] = shape

    def _get_var_shape(self, v: Var):
        if v in self.var2shape:
            return self.var2shape[v]
        else:
            return None

    def _align_shape(
        self, in_shape: Tuple[ShapeVar], out_shape: Tuple[ShapeVar], shape: Optional[Tuple[ShapeVar]] = None
    ):
        if shape is not None:
            shape_rank = len(shape)
            if in_shape is not None:
                in_shape = list(in_shape)
                assert len(in_shape) == shape_rank
                for i in range(shape_rank):
                    s = shape[i]
                    si = in_shape[i]
                    if is_constant(si):
                        assert si == s
                    else:
                        in_shape[i] = shape[i]
                in_shape = tuple(in_shape)
            else:
                in_shape = shape
            if out_shape is not None:
                out_shape = list(out_shape)
                assert len(out_shape) == shape_rank
                for i in range(shape_rank):
                    s = shape[i]
                    so = out_shape[i]
                    if is_constant(so):
                        assert so == s
                    else:
                        out_shape[i] = shape[i]
                out_shape = tuple(out_shape)
            else:
                out_shape = shape
        else:
            if in_shape is not None:
                if out_shape is not None:
                    in_shape = list(in_shape)
                    out_shape = list(out_shape)
                    ri = len(in_shape)
                    ro = len(out_shape)
                    assert ri == ro
                    for i in range(ri):
                        si = in_shape[i]
                        so = out_shape[i]
                        if is_constant(si):
                            if is_constant(so):
                                assert si == so
                            else:
                                out_shape[i] = si
                        else:
                            if is_constant(so):
                                in_shape[i] = so
                    in_shape = tuple(in_shape)
                    out_shape = tuple(out_shape)
                else:
                    out_shape = in_shape
            else:
                if out_shape is not None:
                    in_shape = out_shape
        return in_shape, out_shape

    def visit_Copy(self, op: Copy):
        vars = self.op2vars[op]
        in_var, out_var = vars
        shape = None
        shape = op.tiled_copy.shape
        if shape is not None:
            shape = tuple(shape)
        in_shape = self._get_var_shape(in_var)
        out_shape = self._get_var_shape(out_var)
        in_shape, out_shape = self._align_shape(in_shape, out_shape, shape)
        self._update_var_shape(in_var, in_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(in_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def visit_Rearrange(self, op: Rearrange):
        vars = self.op2vars[op]
        in_var, out_var = vars
        in_shape = self._get_var_shape(in_var)
        out_shape = self._get_var_shape(out_var)
        in_shape, out_shape = self._align_shape(in_shape, out_shape)
        self._update_var_shape(in_var, in_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(in_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def visit_SubTensor(self, op: SubTensor):
        vars = self.op2vars[op]
        in_var, out_var = vars
        crd = op.coord
        rc = len(crd)
        in_shape = self._get_var_shape(in_var)
        out_shape = self._get_var_shape(out_var)
        if in_shape is not None:
            if out_shape is not None:
                ri = len(in_shape)
                ro = len(out_shape)
                in_shape = list(in_shape)
                out_shape = list(out_shape)
                assert rc == ri and ro <= ri
                o_idx = 0
                for i_idx in range(ri):
                    if crd[i_idx] is None:
                        so = out_shape[o_idx]
                        si = in_shape[i_idx]
                        if is_constant(si):
                            if is_constant(so):
                                assert si == so
                            else:
                                out_shape[o_idx] = si
                        else:
                            if is_constant(so):
                                in_shape[i_idx] = so
                        o_idx += 1
                in_shape = tuple(in_shape)
                out_shape = tuple(out_shape)
            else:
                ri = len(in_shape)
                assert rc == ri
                out_shape = []
                for i_idx in range(ri):
                    if crd[i_idx] is None:
                        out_shape.append(in_shape[i_idx])
                out_shape = tuple(out_shape)
        else:
            if out_shape is not None:
                ro = len(out_shape)
                assert ro <= rc
                o_idx = 0
                in_shape = [var("v") for _ in range(rc)]
                for i_idx in range(rc):
                    if crd[i_idx] is None:
                        in_shape[i_idx] = out_shape[o_idx]
                        o_idx += 1
                in_shape = tuple(in_shape)
        self._update_var_shape(in_var, in_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(in_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def visit_Partition(self, op: Partition):
        vars = self.op2vars[op]
        in_var, out_var = vars
        in_shape = self._get_var_shape(in_var)
        out_shape = self._get_var_shape(out_var)
        if in_shape is not None:
            if out_shape is not None:
                in_shape = list(in_shape)
                out_shape = list(out_shape)
                ri = len(in_shape)
                ro = len(out_shape)
                assert ro >= ri
                i_idx = 0
                cur = in_shape[i_idx]
                for o_idx in range(ro):
                    so = out_shape[o_idx]
                    if is_constant(cur):
                        if is_constant(so):
                            if cur == so:
                                i_idx += 1
                                if i_idx < ri:
                                    cur = in_shape[i_idx]
                            else:
                                assert cur % so == 0
                                cur = cur // so
                        else:
                            out_shape[o_idx] = cur
                            i_idx += 1
                            if i_idx < ri:
                                cur = in_shape[i_idx]
                    else:
                        if is_constant(so):
                            in_shape[i_idx] = so
                        i_idx += 1
                        if i_idx < ri:
                            cur = in_shape[i_idx]
                assert i_idx == ri
                in_shape = tuple(in_shape)
                out_shape = tuple(out_shape)
        else:
            if out_shape is not None:
                in_shape = out_shape
        self._update_var_shape(in_var, in_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(in_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def visit_PartitionA(self, op: Partition):
        self.visit_Partition(op)

    def visit_PartitionB(self, op: Partition):
        self.visit_Partition(op)

    def visit_PartitionSrc(self, op: PartitionSrc):
        self.visit_Partition(op)

    def visit_PartitionDst(self, op: PartitionDst):
        self.visit_Partition(op)

    def visit_TensorBase(self, op: TensorBase):
        vars = self.op2vars[op]
        out_var = vars[-1]
        out_shape = self._get_var_shape(out_var)
        shape = op.layout.shape
        if shape is not None:
            shape = product_each(shape)
            if out_shape is not None:
                # just validate the partition can be done
                out_shape = list(out_shape)
                r = len(shape)
                ro = len(out_shape)
                assert ro >= r
                idx = 0
                cur = shape[idx]
                for o_idx in range(ro):
                    assert is_constant(cur)
                    so = out_shape[o_idx]
                    if is_constant(so):
                        if cur == so:
                            idx += 1
                            if idx < r:
                                cur = shape[idx]
                        else:
                            assert cur % so == 0
                            cur = cur // so
                    else:
                        out_shape[o_idx] = cur
                        idx += 1
                        if idx < r:
                            cur = shape[idx]
                assert idx == r
                out_shape = shape
            else:
                out_shape = shape
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def visit_Tensor(self, op: Tensor):
        self.visit_TensorBase(op)

    def visit_TensorView(self, op: TensorView):
        self.visit_TensorBase(op)

    def visit_Arithmetic(self, op: Arithmetic):
        vars = self.op2vars[op]
        shape = None
        for v in vars:
            cur = self._get_var_shape(v)
            cur, shape = self._align_shape(cur, shape)
        if shape is not None:
            for v in vars:
                self._update_var_shape(v, shape)
        if all(self._shape_resolved(self._get_var_shape(v)) for v in vars):
            self.ops_resolved.append(op)

    def visit_Reduce(self, op: Reduce):
        vars = self.op2vars[op]
        in_var, out_var = vars
        in_shape = self._get_var_shape(in_var)
        out_shape = self._get_var_shape(out_var)
        in_shape, out_shape = self._align_shape(in_shape, out_shape)
        self._update_var_shape(in_var, in_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(in_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def _broadcast_shape(self, shape: Tuple[ShapeVar], broadcast_shape: Tuple[ShapeVar]):
        broadcast_dim = None
        ls = len(shape)
        lb = len(broadcast_shape)
        assert ls == lb
        for i in range(ls):
            s = shape[i]
            s1 = broadcast_shape[i]
            if is_constant(s):
                if is_constant(s1):
                    if s != s1:
                        assert broadcast_dim is None
                        broadcast_dim = i
        if broadcast_dim is not None:
            shape = list(shape)
            broadcast_shape = list(broadcast_shape)
            for i in range(ls):
                s = shape[i]
                s1 = broadcast_shape[i]
                if i != broadcast_dim:
                    if is_constant(s):
                        if is_constant(s1):
                            assert s == s1
                        else:
                            broadcast_shape[i] = s
                    else:
                        if is_constant(s1):
                            shape[i] = s1
        return tuple(shape), tuple(broadcast_shape)

    def visit_Broadcast(self, op: Broadcast):
        """
        When inferring the logical shape, all shapes are broadcasted beforehand.
        However, we will encounter cases where a tensor will be broadcasted twice.
        The Broadcast op is introduced to resolve the conflicts.
        The operator allows a tensor to be broadcasted to a different dimensionality from
        the former broadcasted dimensionality.
        TODO: need a more elegant solution.
        """
        vars = self.op2vars[op]
        x_var, tgt_var, out_var = vars
        x_shape = self._get_var_shape(x_var)
        tgt_shape = self._get_var_shape(tgt_var)
        out_shape = self._get_var_shape(out_var)
        if x_shape is not None:
            if tgt_shape is not None:
                if out_shape is not None:
                    tgt_shape, out_shape = self._align_shape(tgt_shape, out_shape)
                    x_shape, tgt_shape = self._broadcast_shape(x_shape, tgt_shape)
                    tgt_shape, out_shape = self._align_shape(tgt_shape, out_shape)
                else:
                    x_shape, tgt_shape = self._broadcast_shape(x_shape, tgt_shape)
                    out_shape = tgt_shape
            else:
                if out_shape is not None:
                    x_shape, out_shape = self._broadcast_shape(x_shape, out_shape)
                    tgt_shape = out_shape
        else:
            tgt_shape, out_shape = self._align_shape(tgt_shape, out_shape)
        self._update_var_shape(x_var, x_shape)
        self._update_var_shape(tgt_var, tgt_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(x_shape) and self._shape_resolved(tgt_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def visit_Transpose(self, op: Transpose):
        vars = self.op2vars[op]
        x_var, out_var = vars
        x_shape = self._get_var_shape(x_var)
        out_shape = self._get_var_shape(out_var)
        if x_shape is not None:
            if out_shape is not None:
                x_shape = list(x_shape)
                out_shape = list(out_shape)
                lx = len(x_shape)
                lo = len(out_shape)
                assert lx == lo
                o_shape = []
                for i in op.dims:
                    o_shape.append(x_shape[i])
                o_shape, out_shape = self._align_shape(o_shape, out_shape)
                for i, d in enumerate(op.dims):
                    x_shape[d] = o_shape[i]
                x_shape = tuple(x_shape)
                out_shape = tuple(out_shape)
            else:
                assert len(x_shape) == len(op.dims)
                out_shape = tuple(x_shape[d] for d in op.dims)
        else:
            if out_shape is not None:
                assert len(out_shape) == len(op.dims)
                x_shape = []
                for i, d in enumerate(op.dims):
                    x_shape[d] = out_shape[i]
                x_shape = tuple(x_shape)
        self._update_var_shape(x_var, x_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(x_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def visit_Atomic(self, op: Atomic):
        vars = self.op2vars[op]
        in_var, out_var = vars
        in_shape = self._get_var_shape(in_var)
        out_shape = self._get_var_shape(out_var)
        in_shape, out_shape = self._align_shape(in_shape, out_shape)
        self._update_var_shape(in_var, in_shape)
        self._update_var_shape(out_var, out_shape)
        if self._shape_resolved(in_shape) and self._shape_resolved(out_shape):
            self.ops_resolved.append(op)

    def _resolve(self):
        for op in self.ops_unresolved:
            self.visit(op)
        for op in self.ops_resolved:
            self.ops_unresolved.remove(op)
        self.ops_resolved = []

    def infer(self):
        for op in self.ops_resolved:
            if isinstance(op, TensorBase):
                ty = self.infer_type(op.make_call())
                shape = (
                    ty.layout.shape
                    if isinstance(ty.layout, (TensorLayout, ComposedTensorLayout))
                    else ty.layout.shape()
                )
                if shape is not None:
                    shape = product_each(shape)
                vars = self.op2vars[op]
                out_var = vars[-1]
                self._update_var_shape(out_var, shape)
            elif isinstance(op, Transpose):
                vars = self.op2vars[op]
                x_var, out_var = vars
                x_type = self.infer_type(op.x)
                x_shape = product_each(x_type.layout.shape_tuple)
                out_type = self.infer_type(op.make_call())
                out_shape = product_each(out_type.layout.shape_tuple)
                x_shp = self._get_var_shape(x_var)
                out_shp = self._get_var_shape(out_var)
                x_shape, x_shp = self._align_shape(x_shape, x_shp)
                out_shape, out_shp = self._align_shape(out_shape, out_shp)
                self._update_var_shape(x_var, x_shape)
                self._update_var_shape(out_var, out_shape)
            elif isinstance(op, (PartitionSrc, PartitionDst, SubTensor)):
                pass
            else:
                res = op.resolve_logical_encoding()
                if isinstance(res, NotImplementedError):
                    raise NotImplementedError(
                        f"Missing resolve_logical_encoding method for the following operator: \n{type(op).__name__}"
                    )
                else:
                    for i, v in enumerate(self.op2vars[op]):
                        if res[i] is not None:
                            self._update_var_shape(v, res[i].shape)
        self.ops_resolved = []

        while len(self.ops_unresolved) > 0:
            self._resolve()

        return self.var2shape


class InferLogicalLayout(IRVisitor):
    def __init__(
        self,
        var2shape: Dict[Var, Tuple[int]],
        ops_resolved: List[Op],
        ops_unresolved: List[Op],
        op2vars: Dict[Op, List[Var]],
    ):
        super().__init__(False)
        self.var2shape: Dict[Var, Tuple[int]] = var2shape
        self.var2layout: Dict[Var, TensorLayout] = {}

        self.ops_resolved: List[Op] = ops_resolved
        self.ops_unresolved: List[Op] = ops_unresolved
        self.op2vars: Dict[Op, List[Var]] = op2vars
        self.infer_type = TypeInfer()

    def _layout_resolved(self, layout: TensorLayout):
        return layout is not None and all(
            is_constant(s) and is_constant(d) for s, d in zip(layout.shape_tuple, layout.stride_tuple)
        )

    def _update_var_layout(self, v: Var, layout: TensorLayout):
        if layout is not None:
            self.var2layout[v] = layout

    def _get_var_layout(self, v: Var):
        if v in self.var2layout:
            return self.var2layout[v]
        else:
            return None

    def _infer_layout(
        self, in_shape: Tuple[int], out_shape: Tuple[int], in_layout: TensorLayout, out_layout: TensorLayout
    ):
        from hidet.ir.cute.layout import common_reshape

        if in_layout is not None:
            if out_layout is not None:
                inlayout, outlayout = common_reshape(in_layout, out_layout)
                inshape = inlayout.shape_tuple
                instride = list(inlayout.stride_tuple)
                outshape = outlayout.shape_tuple
                outstride = list(outlayout.stride_tuple)
                assert len(instride) == len(outstride)
                ri = len(instride)
                for i in range(ri):
                    di = instride[i]
                    do = outstride[i]
                    if is_constant(di):
                        if is_constant(do):
                            assert di == do
                        else:
                            outstride[i] = di
                    else:
                        if is_constant(do):
                            instride[i] = do
                inlayout = TensorLayout(inshape, tuple(instride))
                outlayout = TensorLayout(outshape, tuple(outstride))
                in_layout = composition(inlayout, TensorLayout(in_shape))
                out_layout = composition(outlayout, TensorLayout(out_shape))
            else:
                out_layout = composition(in_layout, TensorLayout(out_shape))
        else:
            if out_layout is not None:
                in_layout = composition(out_layout, TensorLayout(in_shape))
        return in_layout, out_layout

    def _identical_io(self, op: Op):
        vars = self.op2vars[op]
        in_var, out_var = vars
        assert in_var in self.var2shape and out_var in self.var2shape
        in_shape = self.var2shape[in_var]
        out_shape = self.var2shape[out_var]
        in_layout = self._get_var_layout(in_var)
        out_layout = self._get_var_layout(out_var)
        in_layout, out_layout = self._infer_layout(in_shape, out_shape, in_layout, out_layout)
        self._update_var_layout(in_var, in_layout)
        self._update_var_layout(out_var, out_layout)
        if self._layout_resolved(in_layout) and self._layout_resolved(out_layout):
            self.ops_resolved.append(op)

    def visit_Copy(self, op: Copy):
        self._identical_io(op)

    def visit_Rearrange(self, op: Rearrange):
        self._identical_io(op)

    def visit_Partition(self, op: Partition):
        self._identical_io(op)

    def visit_PartitionA(self, op: PartitionA):
        self.visit_Partition(op)

    def visit_PartitionB(self, op: PartitionB):
        self.visit_Partition(op)

    def visit_PartitionSrc(self, op: PartitionSrc):
        self.visit_Partition(op)

    def visit_PartitionDst(self, op: PartitionDst):
        self.visit_Partition(op)

    def visit_SubTensor(self, op: SubTensor):
        vars = self.op2vars[op]
        in_var, out_var = vars
        crd = op.coord
        assert in_var in self.var2shape and out_var in self.var2shape
        in_shape = self.var2shape[in_var]
        out_shape = self.var2shape[out_var]
        in_layout = self._get_var_layout(in_var)
        out_layout = self._get_var_layout(out_var)
        if in_layout is not None:
            if out_layout is not None:
                inshape = in_layout.shape_tuple
                instride = list(in_layout.stride_tuple)
                outshape = out_layout.shape_tuple
                outstride = list(out_layout.stride_tuple)
                assert inshape == in_shape and outshape == out_shape
                in_stride = compact_col_major(in_shape)
                out_stride = compact_col_major(out_shape)
                ri = len(inshape)
                ro = len(outshape)
                rc = len(crd)
                assert rc == ri and ri >= ro
                o_idx = 0
                for i_idx in range(ri):
                    if crd[i_idx] is None:
                        di = instride[i_idx]
                        do = outstride[o_idx]
                        if is_constant(di):
                            if is_constant(do):
                                assert (di == 0 and do == 0) or (di == in_stride[i_idx] and do == out_stride[o_idx])
                            else:
                                if di == 0:
                                    outstride[o_idx] = 0
                                else:
                                    outstride[o_idx] = out_stride[o_idx]
                        else:
                            if is_constant(do):
                                if do == 0:
                                    instride[i_idx] = 0
                                else:
                                    instride[i_idx] = in_stride[i_idx]
                        o_idx += 1
                in_layout = TensorLayout(inshape, tuple(instride))
                out_layout = TensorLayout(outshape, tuple(outstride))
            else:
                out_layout, _ = slice_and_offset(crd, in_layout)
        else:
            if out_layout is not None:
                in_stride = list(compact_col_major(in_shape))
                outshape = out_layout.shape_tuple
                outstride = list(out_layout.stride_tuple)
                assert outshape == out_shape
                ri = len(in_shape)
                ro = len(outshape)
                rc = len(crd)
                assert rc == ri and ri >= ro
                o_idx = 0
                for i_idx in range(ri):
                    if crd[i_idx] is None:
                        di = in_stride[i_idx]
                        do = outstride[o_idx]
                        if is_constant(do):
                            if do == 0:
                                in_stride[i_idx] = 0
                        else:
                            in_stride[i_idx] = var("v")
                        o_idx += 1
                    else:
                        in_stride[i_idx] = var("v")
                out_layout = TensorLayout(outshape, tuple(outstride))
        self._update_var_layout(in_var, in_layout)
        self._update_var_layout(out_var, out_layout)
        if self._layout_resolved(in_layout) and self._layout_resolved(out_layout):
            self.ops_resolved.append(op)

    def visit_TensorBase(self, op: TensorBase):
        from hidet.ir.cute.layout import common_reshape

        vars = self.op2vars[op]
        out_var = vars[-1]
        out_layout = self._get_var_layout(out_var)
        layout = op.layout
        if layout is not auto_layout:
            flat_shape = flatten(layout.shape_tuple)
            flat_stride = flatten(layout.stride_tuple)
            stride = list(compact_col_major(flat_shape))
            for i, d in enumerate(flat_stride):
                if is_constant(d) and d == 0:
                    stride[i] = 0
            layout = TensorLayout(flat_shape, tuple(stride))
            if out_layout is not None:
                layout1, out_layout1 = common_reshape(layout, out_layout)
                stride1 = list(layout1.stride_tuple)
                outstride1 = list(out_layout1.stride_tuple)
                assert len(stride1) == len(outstride1)
                r = len(stride1)
                for i in range(r):
                    di = stride1[i]
                    do = outstride1[i]
                    if is_constant(do):
                        assert di == do
                out_layout = layout
            else:
                out_layout = layout
        self._update_var_layout(out_var, out_layout)
        if self._layout_resolved(out_layout):
            self.ops_resolved.append(op)

    def visit_Tensor(self, op: Tensor):
        self.visit_TensorBase(op)

    def visit_TensorView(self, op: TensorView):
        self.visit_TensorBase(op)

    def visit_Arithmetic(self, op: Arithmetic):
        from hidet.ir.cute.ops.arithmetic import broadcast_layout

        vars = self.op2vars[op]
        inputs = vars[:-1]
        out_var = vars[-1]
        inp_layouts = []
        inferable = True
        for inp in inputs:
            inp_layout = self._get_var_layout(inp)
            if inp_layout is not None and self._layout_resolved(inp_layout):
                inp_layouts.append(inp_layout)
            else:
                inferable = False
                break
        if inferable:
            out_layout1 = broadcast_layout(inp_layouts)
            out_layout = self._get_var_layout(out_var)
            out_shape = self.var2shape[out_var]
            out_layout, out_layout1 = self._infer_layout(out_shape, out_shape, out_layout, out_layout1)
            self._update_var_layout(out_var, out_layout)
        if all(self._layout_resolved(self._get_var_layout(v)) for v in vars):
            self.ops_resolved.append(op)

    def visit_Reduce(self, op: Reduce):
        from hidet.ir.cute import product

        vars = self.op2vars[op]
        in_var, out_var = vars
        in_layout = self._get_var_layout(in_var)
        out_layout = self._get_var_layout(out_var)
        if in_layout is not None:
            shape = self.var2shape[out_var]
            ax = op.axis
            assert 0 <= ax < len(shape)
            lo = product(shape[:ax])
            hi = lo * shape[ax]
            out_layout1 = filter_lo_hi(in_layout, lo, hi)
            out_layout, out_layout1 = self._infer_layout(shape, shape, out_layout, out_layout1)
        else:
            if out_layout is not None:
                in_shape = self.var2shape[in_var]
                in_stride = list(compact_col_major(in_shape))
                out_stride = out_layout.stride_tuple
                ri = len(in_stride)
                ro = len(out_stride)
                assert ri == ro
                for i in range(ri):
                    di = in_stride[i]
                    if di < lo or di >= hi:
                        in_stride[i] = out_stride[i]
        self._update_var_layout(in_var, in_layout)
        self._update_var_layout(out_var, out_layout)
        if self._layout_resolved(in_layout) and self._layout_resolved(out_layout):
            self.ops_resolved.append(op)

    def visit_Broadcast(self, op: Broadcast):
        vars = self.op2vars[op]
        x_var, tgt_var, out_var = vars
        x_shape = self.var2shape[x_var]
        out_shape = self.var2shape[out_var]
        x_layout = self._get_var_layout(x_var)
        tgt_layout = self._get_var_layout(tgt_var)
        out_layout = self._get_var_layout(out_var)
        lx = len(x_shape)
        lo = len(out_shape)
        cont_x_stride = compact_col_major(x_shape)
        cont_out_stride = compact_col_major(out_shape)
        if x_layout is not None:
            x_stride = list(x_layout.stride_tuple)
            if out_layout is not None:
                out_stride = list(out_layout.stride_tuple)
                assert len(x_stride) == lx and len(out_stride) == lo
                for i in range(lx):
                    dx = x_stride[i]
                    do = out_stride[i]
                    if is_constant(dx):
                        if is_constant(do):
                            if dx == 0:
                                assert do == 0
                            else:
                                assert do != 0
                        else:
                            if dx == 0:
                                out_stride[i] = 0
                            else:
                                out_stride[i] = cont_out_stride[i]
                    else:
                        if is_constant(do):
                            if do == 0:
                                x_stride[i] = 0
                            else:
                                x_stride[i] = cont_x_stride[i]
                x_layout = TensorLayout(x_layout.shape_tuple, tuple(x_stride))
                out_layout = TensorLayout(out_layout.shape_tuple, tuple(out_stride))
            else:
                out_stride = [var("v") for _ in range(lx)]
                for i in range(lx):
                    dx = x_stride[i]
                    if is_constant(dx):
                        if dx == 0:
                            out_stride[i] = 0
                        else:
                            out_stride[i] = cont_out_stride[i]
                out_layout = TensorLayout(out_shape, tuple(out_stride))
        else:
            if out_layout is not None:
                out_stride = out_layout.stride_tuple
                x_stride = [var("v") for _ in range(lx)]
                lo = len(out_stride)
                assert lx == lo
                for i in range(lx):
                    do = out_stride[i]
                    if is_constant(do):
                        if do == 0:
                            x_stride[i] = 0
                        else:
                            x_stride[i] = cont_x_stride[i]
                x_layout = TensorLayout(x_shape, tuple(x_stride))
        self._update_var_layout(x_var, x_layout)
        self._update_var_layout(tgt_var, tgt_layout)
        self._update_var_layout(out_var, out_layout)
        if self._layout_resolved(x_layout) and self._layout_resolved(tgt_layout) and self._layout_resolved(out_layout):
            self.ops_resolved.append(op)

    def visit_Transpose(self, op: Transpose):
        vars = self.op2vars[op]
        x_var, out_var = vars
        x_shape = self.var2shape[x_var]
        x_layout = self._get_var_layout(x_var)
        out_shape = self.var2shape[out_var]
        out_layout = self._get_var_layout(out_var)
        if x_layout is not None:
            lx = rank(x_layout.shape)
            num_dims = len(op.dims)
            assert lx == num_dims
            nonzero_strides = set()
            zero_strides = set()
            for i, d in enumerate(x_layout.stride_tuple):
                if is_constant(d):
                    if d == 0:
                        zero_strides.add(i)
                    else:
                        nonzero_strides.add(i)
            if out_layout is not None:
                lo = rank(out_layout.shape)
                assert lx == lo
                for i, d in zip(op.dims, out_layout.stride_tuple):
                    if is_constant(d):
                        if d == 0:
                            zero_strides.add(i)
                            assert i not in nonzero_strides
                        else:
                            nonzero_strides.add(i)
                            assert i not in zero_strides
                x_stride = list(compact_col_major(x_shape))
                for i, d in enumerate(x_stride):
                    if i in zero_strides:
                        x_stride[i] = 0
                    elif i not in nonzero_strides:
                        x_stride[i] = var("v")
                out_stride = list(compact_col_major(out_shape))
                for i, (j, d) in enumerate(zip(op.dims, out_stride)):
                    if j in zero_strides:
                        out_stride[i] = 0
                    elif j not in nonzero_strides:
                        out_stride[i] = var("v")
                x_stride = tuple(x_stride)
                out_stride = tuple(out_stride)
                x_layout = TensorLayout(x_shape, x_stride)
                out_layout = TensorLayout(out_shape, out_stride)
            else:
                out_stride = list(compact_col_major(out_shape))
                for i, (j, d) in enumerate(zip(op.dims, out_stride)):
                    if j in zero_strides:
                        out_stride[i] = 0
                    elif j not in nonzero_strides:
                        out_stride[i] = var("v")
                out_stride = tuple(out_stride)
                out_layout = TensorLayout(out_shape, out_stride)
        else:
            if out_layout is not None:
                lo = rank(out_layout.shape)
                num_dims = len(op.dims)
                assert lo == num_dims
                nonzero_strides = set()
                zero_strides = set()
                for i, d in zip(op.dims, out_layout.stride_tuple):
                    if is_constant(d):
                        if d == 0:
                            zero_strides.add(i)
                        else:
                            nonzero_strides.add(i)
                x_stride = list(compact_col_major(x_shape))
                for i, d in enumerate(x_stride):
                    if i in zero_strides:
                        x_stride[i] = 0
                    elif i not in nonzero_strides:
                        x_stride[i] = var("v")
                x_stride = tuple(x_stride)
                x_layout = TensorLayout(x_shape, x_stride)
        self._update_var_layout(x_var, x_layout)
        self._update_var_layout(out_var, out_layout)
        if self._layout_resolved(x_layout) and self._layout_resolved(out_layout):
            self.ops_resolved.append(op)

    def visit_Atomic(self, op: Atomic):
        vars = self.op2vars[op]
        in_var, out_var = vars
        assert in_var in self.var2shape and out_var in self.var2shape
        in_shape = self.var2shape[in_var]
        out_shape = self.var2shape[out_var]
        in_layout = self._get_var_layout(in_var)
        out_layout = self._get_var_layout(out_var)
        in_layout, out_layout = self._infer_layout(in_shape, out_shape, in_layout, out_layout)
        self._update_var_layout(in_var, in_layout)
        self._update_var_layout(out_var, out_layout)
        if self._layout_resolved(in_layout) and self._layout_resolved(out_layout):
            self.ops_resolved.append(op)

    def _resolve(self):
        for op in self.ops_unresolved:
            self.visit(op)
        for op in self.ops_resolved:
            self.ops_unresolved.remove(op)
        self.ops_resolved = []

    def _inst_logical_layout_for_tensor(self, layout):
        if isinstance(layout, (TensorLayout, ComposedTensorLayout)):
            flat_shape = flatten(layout.shape_tuple)
            flat_stride = flatten(layout.stride_tuple)
            stride = list(compact_col_major(flat_shape))
            for i, d in enumerate(flat_stride):
                if is_constant(d) and d == 0:
                    stride[i] = 0
            layout = TensorLayout(flat_shape, tuple(stride))
            return layout
        else:
            assert isinstance(layout, TiledTensorLayout)
            shape = layout.shape()
            tv_layout = make_layout(layout.thr_layout(), layout.val_layout())
            from hidet.ir.cute import codomain_from_shape_and_tv_layout

            return codomain_from_shape_and_tv_layout(shape, tv_layout)

    def infer(self):
        for op in self.ops_resolved:
            if isinstance(op, Mma):
                d, a, b, c = op.args
                shape_mn, _ = op.tiled_mma.c_tv_layout()
                shape_mk, _ = op.tiled_mma.a_tv_layout()
                shape_nk, _ = op.tiled_mma.b_tv_layout()
                a_layout = TensorLayout(shape_mk)
                b_layout = TensorLayout(shape_nk)
                c_layout = TensorLayout(shape_mn)
                self._update_var_layout(d, c_layout)
                self._update_var_layout(a, a_layout)
                self._update_var_layout(b, b_layout)
                self._update_var_layout(c, c_layout)
            elif isinstance(op, TensorBase):
                out_var = self.op2vars[op][-1]
                assert op.layout is not auto_layout
                ty = self.infer_type(op.make_call())
                layout = self._inst_logical_layout_for_tensor(ty.layout)
                self._update_var_layout(out_var, layout)
            elif isinstance(op, Transpose):
                vars = self.op2vars[op]
                x_var, out_var = vars
                x_layout = self._get_var_layout(x_var)
                if x_layout is None:
                    x_type = self.infer_type(op.x)
                    x_layout = x_type.layout
                    layout = self._inst_logical_layout_for_tensor(x_layout)
                    self._update_var_layout(x_var, layout)
                out_layout = self._get_var_layout(out_var)
                if out_layout is None:
                    out_type = self.infer_type(op.make_call())
                    out_layout = out_type.layout
                    layout = self._inst_logical_layout_for_tensor(out_layout)
                    self._update_var_layout(out_var, layout)
            elif isinstance(op, SubTensor):
                pass
            else:
                res = op.resolve_logical_encoding()
                if isinstance(res, NotImplementedError):
                    raise NotImplementedError(
                        f"Missing resolve_logical_encoding method for the following operator: \n{type(op).__name__}"
                    )
                else:
                    for i, v in enumerate(self.op2vars[op]):
                        if res[i] is not None:
                            from hidet.ir.cute import codomain_from_shape_and_tv_layout

                            logical_layout = codomain_from_shape_and_tv_layout(res[i].shape, res[i].layout)
                            self._update_var_layout(v, logical_layout)
        self.ops_resolved = []

        while len(self.ops_unresolved) > 0:
            self._resolve()

        return self.var2layout


class InferContext:
    def __init__(
        self,
        var2layout: Dict[Var, TensorLayout],
        var2tensor: Dict[Var, TensorInfo],
        solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]],
    ):
        self.var2layout: Dict[Var, TensorLayout] = var2layout
        self.var2tensor: Dict[Var, TensorInfo] = var2tensor
        self.solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]] = solution


def infer_context(
    var2layout: Dict[Var, TensorLayout],
    var2tensor: Dict[Var, TensorBase],
    solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]],
):
    return InferContext(var2layout, var2tensor, solution)


class MemoryConstraint:
    def __init__(self, tensor: TensorBase, memory_constraints: TensorLayout):
        self.tensor = tensor
        self.memory_constraints = memory_constraints


def constraint(tensor: TensorBase, memory_constraints: TensorLayout):
    return MemoryConstraint(tensor, memory_constraints)


class InferResult:
    def __init__(
        self, encs: List[LogicalEncoding], memory_constraints: Union[List[MemoryConstraint], MemoryConstraint]
    ):
        self.encs = encs
        self.memory_constraints = memory_constraints

    def __getitem__(self, i):
        return self.encs[i]

    def empty(self):
        return len(self.encs) == 0


def infer_result(*encs: LogicalEncoding, memory_constraints: Union[List[MemoryConstraint], MemoryConstraint] = None):
    return InferResult(encs, memory_constraints)


class InferRule:
    def __init__(self, name: str, call: Callable):
        self.name: str = name
        self.call: Callable = call

    def __call__(
        self,
        op: Op,
        args: List[LogicalEncoding],
        ctx: InferContext,
        input_vars: Optional[List[Var]] = None,
        output_var: Optional[Var] = None,
    ):
        return self.call(op, args, ctx, input_vars, output_var)


class InferRules:
    def __init__(self):
        self.infer_rules = {}

    def update_infer_rules(self, name: str, call: Callable):
        self.infer_rules.update({name: InferRule(name, call)})
        return self

    def get_infer_func(self, name: str):
        if name in self.infer_rules:
            return self.infer_rules[name]
        else:
            return None


_infer_rules: Dict[Type[Op], Type[InferRules]] = {}


def register_infer_rules(op_cls: Type[Op]):
    def decorator(infer_rules_cls: Type[InferRules]):
        _infer_rules[op_cls] = infer_rules_cls

    return decorator


def get_infer_rules(op_or_op_cls: Union[Op, Type[Op]]) -> Type[InferRules]:
    if isinstance(op_or_op_cls, Op):
        op_cls = type(op_or_op_cls)
    elif issubclass(op_or_op_cls, Op):
        op_cls = op_or_op_cls
    else:
        raise RuntimeError(f"Cannot get infer rules for {op_or_op_cls}")

    import inspect

    if op_cls not in _infer_rules:
        parent_classes: Tuple = inspect.getmro(op_cls)
        for cls in parent_classes:
            if cls in _infer_rules:
                _infer_rules[op_cls] = _infer_rules[cls]
                break
        else:
            raise RuntimeError(f"Cannot get infer rules for {op_cls.op_name()}")
    infer_rules = _infer_rules[op_cls]

    return infer_rules()


class MemoryConstraintsUnifier:
    """
    A memory constraints unifier that attempts to find a memory substitution that satisfies the alignment requirement.

    Example:
    --------
    For a tensor `A` defined as:
    ```python
    A = tensor(f16, [32, 32])
    ```
    If using ldg128 for memory load, infer a TV layout `tv`:
    ```python
    tv = ((4, 8), (8, 4)):((128, 1), (32, 8))
    ```
    This requires memory addresses to be multiples of 16-bytes. The memory constraints for tensor A can
    be represented as:
    ```python
    layout_A = (32, (8, 4)):(v, (1, v))
    ```
    Here, `v` represents a stride that is not yet determined and is free in the inference process.

    The memory inference algorithm infers memory constraints for each copy operation and represents them
    in the above form. It then unifies the memory constraints of each tensor to find a layout that satisfies
    all requirements. If no such layout can be found, the inference fails, and the current solution is rejected.

    Methods:
    --------
    unify(memory_constraints1: TensorLayout, memory_constraints2: TensorLayout) -> TensorLayout
        Unifies two memory constraints by aligning their shapes and strides. Returns the unified memory
        constraint or None if they cannot be unified.

    infer(layout_hint: TensorLayout, val: TensorLayout) -> TensorLayout
        Infers a memory layout from a given layout hint and a value layout. Returns the inferred memory
        layout or None if it cannot be inferred.

    """

    def _get_stride(self, d1: Union[Expr, int], d2: Union[Expr, int]):
        """
        Determines the stride to use based on two potential stride values.

        Parameters:
        ----------
        d1 : Union[Expr, int]
            The first stride value.
        d2 : Union[Expr, int]
            The second stride value.

        Returns:
        -------
        Union[Expr, int]
            The selected stride value or None if they cannot be unified.
        """
        if is_constant(d1):
            if is_constant(d2):
                return d1 if d1 == d2 else None
            else:
                return d1
        else:
            return d2

    def unify(self, memory_constraints1: TensorLayout, memory_constraints2: TensorLayout):
        """
        Unifies two memory constraints by aligning their shapes and strides.

        Parameters:
        ----------
        memory_constraints1 : TensorLayout
            The first memory constraint to unify.
        memory_constraints2 : TensorLayout
            The second memory constraint to unify.

        Returns:
        -------
        TensorLayout
            The unified memory constraint or None if unification fails.
        """
        if memory_constraints1 is None:
            return memory_constraints2
        elif memory_constraints2 is None:
            return memory_constraints1
        rank1 = rank(memory_constraints1.shape_tuple)
        rank2 = rank(memory_constraints2.shape_tuple)
        assert rank1 == rank2

        result_shape = []
        result_stride = []
        for mode1, mode2 in zip(memory_constraints1, memory_constraints2):
            cur_shape = []
            cur_stride = []
            shape1 = list(mode1.shape_tuple)
            stride1 = list(mode1.stride_tuple)
            shape2 = list(mode2.shape_tuple)
            stride2 = list(mode2.stride_tuple)
            i = 0
            j = 0
            while i < len(shape1) and j < len(shape2):
                if shape1[i] == shape2[j]:
                    cur_shape.append(shape1[i])
                    d = self._get_stride(stride2[j], stride1[i])
                    if d is None:
                        return None
                    cur_stride.append(d)
                    i = i + 1
                    j = j + 1
                elif shape1[i] < shape2[j]:
                    assert shape2[j] % shape1[i] == 0
                    cur_shape.append(shape1[i])
                    shape2[j] //= shape1[i]
                    d = self._get_stride(stride2[j], stride1[i])
                    if d is None:
                        return None
                    cur_stride.append(d)
                    if is_constant(stride2[j]):
                        stride2[j] *= shape1[i]
                    i = i + 1
                else:
                    assert shape1[i] % shape2[j] == 0
                    cur_shape.append(shape2[j])
                    shape1[i] //= shape2[j]
                    d = self._get_stride(stride2[j], stride1[i])
                    if d is None:
                        return None
                    cur_stride.append(d)
                    if is_constant(stride1[i]):
                        stride1[i] *= shape2[j]
                    j = j + 1
            result_shape.append(tuple(cur_shape))
            result_stride.append(tuple(cur_stride))
        result = TensorLayout(tuple(result_shape), tuple(result_stride))
        flat_shape = flatten(result.shape)
        flat_stride = flatten(result.stride)
        shape = []
        stride = []
        for s, d in zip(flat_shape, flat_stride):
            if is_constant(d):
                shape.append(s)
                stride.append(d)
        if len(shape) == 0:
            return result
        stride, shape = list(zip(*list(sorted(zip(stride, shape)))))
        layout = TensorLayout(shape, stride)
        layout = coalesce(layout)
        if len(layout.stride_tuple) > 1:
            return None
        return result

    def infer(self, layout_hint: TensorLayout, val: TensorLayout):
        """
        Infers a memory layout from a given layout hint and a value layout.

        Parameters:
        ----------
        layout_hint : TensorLayout
            The layout hint to guide the inference.
        val : TensorLayout
            The value layout to be inferred.

        Returns:
        -------
        TensorLayout
            The inferred memory layout or None if it cannot be inferred.
        """
        from hidet.ir.cute import shape_div

        flat_shape = list(flatten(layout_hint.shape_tuple))
        # TODO: handle broadcast
        # flat_stride = list(flatten(layout_hint.stride_tuple))
        val = coalesce(val)
        val_shape = val.shape_tuple
        val_stride = val.stride_tuple
        cont_stride = compact_col_major(val_shape)
        shapes = []
        strides = []
        current_idx = 1
        for d, s, d1 in sorted(zip(val_stride, val_shape, cont_stride)):
            if d > current_idx:
                s1 = shape_div(d, current_idx)
                shapes.append(s1)
                strides.append(var("v"))
                current_idx *= s1
            shapes.append(s)
            strides.append(d1)
            current_idx = d * s
        from hidet.utils import prod

        if prod(flat_shape) < prod(shapes):
            return None
        result_shape = []
        result_stride = []
        cur_shape = []
        cur_stride = []
        i = 0
        size = flat_shape[i]
        current_idx = 1
        for s, d in zip(shapes, strides):
            while current_idx * s > size:
                if size % current_idx != 0:
                    return None
                s1 = shape_div(size, current_idx)
                s = shape_div(s, s1)
                cur_shape.append(s1)
                cur_stride.append(d)
                d = var("v") if isinstance(d, Var) else d * s1
                result_shape.append(tuple(cur_shape))
                result_stride.append(tuple(cur_stride))
                cur_shape = []
                cur_stride = []
                current_idx *= s1
                i = i + 1
                if i < len(flat_shape):
                    size *= flat_shape[i]
            if s > 1:
                cur_shape.append(s)
                cur_stride.append(d)
                current_idx *= s
                if current_idx == size:
                    result_shape.append(tuple(cur_shape))
                    result_stride.append(tuple(cur_stride))
                    cur_shape = []
                    cur_stride = []
                    i = i + 1
                    if i < len(flat_shape):
                        size *= flat_shape[i]
        if size > current_idx:
            cur_shape.append(shape_div(size, current_idx))
            cur_stride.append(var("v"))
            result_shape.append(tuple(cur_shape))
            result_stride.append(tuple(cur_stride))
            i = i + 1
        while i < len(flat_shape):
            result_shape.append(flat_shape[i])
            result_stride.append(var("v"))
            i = i + 1
        return TensorLayout(tuple(result_shape), tuple(result_stride))


def infer_memory_constraints(ctx: InferContext, tensor_info: TensorInfo, value: TensorLayout, elements_per_inst):
    tensor = tensor_info.tensor
    tensor_memory = tensor_info.layout
    tensor_memory_constraints = ctx.solution.get(tensor, None)
    value_inst, _ = group(value, elements_per_inst)
    unifier = MemoryConstraintsUnifier()
    memory_constraints = unifier.infer(tensor_memory, value_inst)
    memory_constraints = unifier.unify(memory_constraints, tensor_memory_constraints)
    if memory_constraints is None:
        return None
    try:
        composition(TensorLayout(memory_constraints.shape), value)
    except AssertionError:
        return None
    constr = constraint(tensor, memory_constraints)
    return constr


def validate_alignment(value: TensorLayout, memory: TensorLayout, elements_per_inst: int):
    value_layout = composition(memory, value)
    value_reg = TensorLayout(value.shape)
    alignment = max_common_vector(value_layout, value_reg)
    return alignment % elements_per_inst == 0


# currently, hard-coded number, may be designed as a configurable argument
early_stop_candidates = 4


@register_infer_rules(Mma)
class MmaInferRules(InferRules):
    def __init__(self):
        super().__init__()

        def infer_shared_ab(
            op: Mma, args: List[LogicalEncoding], ctx: InferContext, input_vars: List[Var], output_var: Var
        ):
            a: Expr = op.a
            b: Expr = op.b
            a_ty: TiledTensorType = infer_type(a)
            b_ty: TiledTensorType = infer_type(b)
            candidates: List[MmaInstruction] = []
            arg_tv = [arg.layout[1] for arg in args]

            for inst in get_mma_instructions():
                bufs = [expr_to_buffer(arg, tv) for arg, tv in zip(op.args, arg_tv)]
                result = inst.match(*bufs, op.tiled_mma)
                if result is not None:
                    candidates.append((inst))

            a_v = arg_tv[1]
            b_v = arg_tv[2]
            a_shape = op.tiled_mma.a_shape
            b_shape = op.tiled_mma.b_shape
            infers = []
            for inst in candidates:
                m, n, k = inst.shape_mnk
                constr_a: List[MemoryConstraint] = []
                if a_ty.scope.is_shared():
                    a_tensor_info = ctx.var2tensor[a]
                    a_memory_layout = a_tensor_info.layout
                    core_matrix_layouts = inst.get_core_matrix_layouts_A()
                    for core_matrix_layout in core_matrix_layouts:
                        if core_matrix_layout is None:
                            continue
                        a_v_reorder = self.get_equivalent_value_layout(a_v, core_matrix_layout, (m, k), a_shape)
                        if is_auto_layout(a_memory_layout):
                            constr = infer_memory_constraints(
                                ctx, a_tensor_info, a_v_reorder, core_matrix_layout.size()
                            )
                            if constr is not None:
                                constr_a.append(constr)
                constr_b: List[MemoryConstraint] = []
                if b_ty.scope.is_shared():
                    b_tensor_info = ctx.var2tensor[b]
                    b_memory_layout = b_tensor_info.layout
                    core_matrix_layouts = inst.get_core_matrix_layouts_B()
                    for core_matrix_layout in core_matrix_layouts:
                        if core_matrix_layout is None:
                            continue
                        b_v_reorder = self.get_equivalent_value_layout(b_v, core_matrix_layout, (n, k), b_shape)
                        if is_auto_layout(b_memory_layout):
                            constr = infer_memory_constraints(
                                ctx, b_tensor_info, b_v_reorder, core_matrix_layout.size()
                            )
                            if constr is not None:
                                constr_b.append(constr)
                from itertools import product

                if len(constr_a) != 0 and len(constr_b) != 0:
                    for c1, c2 in product(constr_a, constr_b):
                        infers.append(infer_result([], memory_constraints=[c1, c2]))
                elif len(constr_a) != 0:
                    for c in constr_a:
                        infers.append(infer_result([], memory_constraints=c))
                elif len(constr_b) != 0:
                    for c in constr_b:
                        infers.append(infer_result([], memory_constraints=c))
            if len(infers) == 0:
                raise NotImplementedError(f"Cannot find an instruction to execute mma op{op}")
            return infers

        self.update_infer_rules("shared_ab", infer_shared_ab)

    @staticmethod
    def get_equivalent_value_layout(
        v: TensorLayout, core_matrix_layout: TensorLayout, inst_shape: Tuple[int, int], value_shape: Tuple[int, int]
    ):
        # The constraints on shared memory layout imposed by WG-MMA (Warp-Group Matrix Multiply-Accumulate)
        # can be described as follows:
        #
        # 1. Define a mapping function `f` from coordinates in the *core matrix* to logical positions in the shared
        #    memory tensor.
        # 2. The composition of this mapping function and the shared memory layout `M` must reconstruct the core matrix
        #    layout.
        #
        # Formally, if `f` denotes the mapping function and `M` denotes the shared memory layout, then:
        #     core_matrix_layout = composition(M, f)
        #
        # By checking whether M(f) matches the core matrix layout, we can verify if the shared memory layout satisfies
        # the constraints required by WG-MMA. Alternatively, we can derive the necessary constraints on `M` by inverting
        # the function `f`. Thus, the central challenge becomes constructing the mapping function `f`.
        #
        # Suppose we have:
        # - A value layout `v`, which represents the logical portion of the shared matrix accessed by each thread.
        #
        # In most cases, the coordinates of the core matrix lie within the bounds of the value layout `v`.
        # In these scenarios, we can directly construct the mapping function `f` as:
        #     f = v
        #
        # However, for certain layouts—such as SW32_N, SW64_N, and SW128_N—the coordinate range of the core matrix
        # exceeds the boundary of the value layout. Notably, these overflows occur exclusively along the **K**
        # dimension.
        #
        # Assume the following:
        # - `core_matrix_mn`, `core_matrix_k`: dimensions of the full core matrix
        # - `inst_mn`, `inst_k`: dimensions of each individual hardware instruction tile
        #
        # In these extended cases:
        # - For coordinates **within** the value layout, the mapping function `f` can be constructed as:
        #     f = composition(group(v, core_matrix_mn * inst_k), Layout(core_matrix_mn, inst_k))
        #
        # - For coordinates **outside** the value layout:
        #     - When `core_matrix_k > inst_k` and `core_matrix_mn == inst_mn`,
        #       the mapping along the **K** dimension can be expressed as:
        #           f = (core_matrix_k // inst_k,)
        #
        # This corresponds to a logical layout with strides:
        #     (shape_mn * inst_k,)
        # where `shape_mn` denotes the shape in the **M/N** dimension for matrix `A` or `B`.
        #
        # The reason for this stride pattern is that WG-MMA instructions are **serially executed along the K dimension**
        # within each warp group.
        #
        # The instruction scheduling follows a pattern similar to:
        #
        #   ------------------------------------------------------ K
        #   | inst0, inst1, inst2 --> executed by warp-group 0
        #   | inst3, inst4, inst5 --> executed by warp-group 1
        #   |
        #   M / N
        mn_mode, k_mode = core_matrix_layout
        assert mn_mode.size() == inst_shape[0]
        k_mode_inner, k_mode_outer = group(k_mode, inst_shape[1])
        core_matrix_shape_inner = flatten(mn_mode.shape_tuple + k_mode_inner.shape_tuple)
        stride_outer = value_shape[0] * k_mode_inner.size()
        k_mode_outer_shape = flatten(k_mode_outer.shape_tuple)
        k_mode_outer_stride = prefix_product(k_mode_outer_shape, stride_outer)
        v_reshape = composition(v, TensorLayout(core_matrix_shape_inner))
        shape = v_reshape.shape_tuple
        stride = v_reshape.stride_tuple
        if k_mode_outer.size() > 1:
            shape = shape + k_mode_outer_shape
            stride = stride + k_mode_outer_stride
            core_matrix_stride = flatten(mn_mode.stride_tuple + k_mode_inner.stride_tuple + k_mode_outer.stride_tuple)
        else:
            core_matrix_stride = flatten(mn_mode.stride_tuple + k_mode_inner.stride_tuple)
        assert len(shape) == len(stride) == len(core_matrix_stride)
        sorted_DS = sorted(zip(core_matrix_stride, shape, stride))
        result_shape = flatten(tuple(s for _, s, _ in sorted_DS))
        result_stride = flatten(tuple(d for _, _, d in sorted_DS))
        v_reorder = TensorLayout(result_shape, result_stride)
        return v_reorder


@register_infer_rules(Copy)
class CopyInferRules(InferRules):
    def __init__(self):
        super().__init__()

        def infer(op: Copy, args: List[LogicalEncoding], ctx: InferContext, input_vars: List[Var], output_var: Var):
            candidates: List[CopyInstruction] = []
            src: Expr = op.src
            dst: Expr = op.dst

            src_ty: TiledTensorType = infer_type(src)
            dst_ty: TiledTensorType = infer_type(dst)
            for inst in memory_instructions:
                if inst.src_scope == src_ty.scope and inst.dst_scope == dst_ty.scope:
                    candidates.append(inst)

            candidates = sorted(candidates, key=lambda x: -x.bytes_per_inst)

            dtype_nbits = src_ty.dtype.nbits

            inp: Var = input_vars[0]
            out: Var = output_var
            assert inp is src or inp is dst
            assert out is src or out is dst
            inp_idx: int = 0 if inp is src else 1
            out_idx: int = 1 if out is dst else 0

            arg_shape = args[0].shape
            arg_tv = args[0].layout
            thr, val = arg_tv[0], arg_tv[1]

            if verbose:
                logger.debug("====================================")
                logger.debug(f"infer {op}")

            infers = []
            for inst in candidates:
                layouts = (inst.src_layout, inst.dst_layout)
                inp_layout: TensorLayout = layouts[inp_idx]
                out_layout: TensorLayout = layouts[out_idx]

                inp_inst: TensorLayout = inst.get_layout_in_element(inp, inp_layout)
                out_inst: TensorLayout = inst.get_layout_in_element(out, out_layout)
                if verbose:
                    logger.debug(f"inst: {inst.apply}")
                    logger.debug(f"out: {out_inst}")
                    logger.debug(f"inp: {inp_inst}")
                    logger.debug(f"val: {val}")
                inp_thr_inst, inp_val_inst = inp_inst[0], inp_inst[1]

                if thr.size() < inp_thr_inst.size():
                    continue
                if val.count() < inp_val_inst.size():
                    continue
                # TODO
                if inst.alignment * 8 < dtype_nbits:
                    continue

                thr_inst, thr_rest = group(thr, inp_thr_inst.size(), filter_zero=False)
                val_inst, val_rest = group(val, inp_val_inst.size(), filter_zero=True)
                if any(v is None for v in [thr_inst, val_inst]):
                    continue

                cvt = coalesce(composition(make_layout(thr_inst, filter(val_inst)), left_inverse(inp_inst)))
                if verbose:
                    logger.debug(f"cvt: {cvt}")
                    logger.debug(f"thr_inst: {thr_inst}")
                    logger.debug(f"val_inst: {val_inst}")
                result_tv = composition(cvt, out_inst)
                result_thr, result_val = result_tv[0], result_tv[1]

                result_thr = coalesce(make_layout(result_thr, thr_rest))
                result_val = coalesce(make_layout(result_val, val_rest))
                result_tv = make_layout(result_thr, result_val)

                if inp_idx == 0:
                    src_val = val
                    dst_val = result_val
                elif out_idx == 0:
                    src_val = result_val
                    dst_val = val

                elems_per_inst = (inst.alignment * 8) // dtype_nbits

                constr = None
                if inst.src_scope.is_memory():
                    src_tensor_info = ctx.var2tensor[src]
                    src_memory_layout = src_tensor_info.layout
                    if inst.src_scope.is_global() and is_auto_layout(src_memory_layout):
                        raise ValueError(f"Global memory for var({src}) of copy op({op}) must be specified")
                    if is_auto_layout(src_memory_layout):
                        constr = infer_memory_constraints(ctx, src_tensor_info, src_val, elems_per_inst)
                        if constr is None:
                            continue
                    else:
                        if not validate_alignment(src_val, src_memory_layout, elems_per_inst):
                            continue
                if inst.dst_scope.is_memory():
                    dst_tensor_info = ctx.var2tensor[dst]
                    dst_memory_layout = dst_tensor_info.layout
                    if inst.dst_scope.is_global() and is_auto_layout(dst_memory_layout):
                        raise ValueError(f"Global memory for var({src}) of copy op({op}) must be specified")
                    if is_auto_layout(dst_memory_layout):
                        if constr is not None:
                            raise NotImplementedError(
                                "Infer memory for both input and output not supported, need to extend the algorithm"
                            )
                        constr = infer_memory_constraints(ctx, dst_tensor_info, dst_val, elems_per_inst)
                        if constr is None:
                            continue
                    else:
                        if not validate_alignment(dst_val, dst_memory_layout, elems_per_inst):
                            continue

                if verbose:
                    logger.debug(f"result tv: {result_tv}")
                    logger.debug(f"memory constraints: {constr.memory_constraints if constr is not None else None}")
                    logger.debug("====================================")
                if constr is None:
                    return [infer_result(logical_encoding(arg_shape, result_tv))]
                infers.append(infer_result(logical_encoding(arg_shape, result_tv), memory_constraints=constr))

            if len(infers) == 0:
                raise NotImplementedError(f"Cannot find an instruction to execute copy op{op}")
            return infers[:early_stop_candidates]

        self.update_infer_rules("i2o", infer).update_infer_rules("o2i", infer)


@register_infer_rules(SubTensor)
class SubTensorInferRules(InferRules):
    def __init__(self):
        super().__init__()

        def infer_input(
            op: SubTensor,
            args: List[LogicalEncoding],
            ctx: InferContext,
            input_vars: Optional[List[Var]] = None,
            output_var: Optional[Var] = None,
        ):
            crd = op.coord
            enc = args[0]
            shp = enc.shape
            tv = enc.layout
            thr, val = tv
            in_layout = ctx.var2layout[op.x]
            in_shape = flatten(product_each(in_layout.shape_tuple))
            in_stride = compact_col_major(in_shape)
            rc = len(crd)
            valshp = list(val.shape_tuple)
            valstrd = list(val.stride_tuple)
            for i in range(rc):
                if crd[i] is not None:
                    valshp.append(in_shape[i])
                    valstrd.append(in_stride[i])
            val = TensorLayout(valshp, valstrd)
            inptv = make_layout(thr, val)
            inpshp = list(in_shape)
            li = len(inpshp)
            lo = len(shp)
            assert li >= lo
            for i in range(li):
                if i < lo - 1:
                    assert inpshp[i] == shp[i]
                elif i >= lo:
                    inpshp[lo - 1] *= inpshp[i]
            inpshp = tuple(inpshp[:lo])
            return [infer_result(logical_encoding(inpshp, inptv))]

        def infer_output(
            op: SubTensor,
            args: List[LogicalEncoding],
            ctx: InferContext,
            input_vars: Optional[List[Var]] = None,
            output_var: Optional[Var] = None,
        ):
            enc = args[0]
            shp = enc.shape
            tv = enc.layout
            thr, val = tv
            thrd = thr
            crd = op.coord
            in_layout = ctx.var2layout[op.x]
            in_shape = flatten(product_each(in_layout.shape_tuple))
            in_stride = compact_col_major(in_shape)
            rc = len(crd)
            outshp = []
            for i in range(rc):
                if crd[i] is not None:
                    lo = in_stride[i]
                    hi = in_stride[i] * in_shape[i]
                    val = remove_lo_hi(val, lo, hi)
                    thrd = remove_lo_hi(thrd, lo, hi)
                else:
                    outshp.append(in_shape[i])
            # coalesce two layouts first
            assert coalesce(thrd) == coalesce(thr)
            outtv = make_layout(thr, val)
            li = len(shp)
            lo = len(outshp)
            assert lo >= li
            for i in range(lo):
                if i < li - 1:
                    assert outshp[i] == shp[i]
                elif i >= li:
                    outshp[li - 1] *= outshp[i]
            outshp = tuple(outshp[:li])
            return [infer_result(logical_encoding(outshp, outtv))]

        self.update_infer_rules("o2i", infer_input).update_infer_rules("i2o", infer_output)


def forward(
    op: Op,
    args: List[LogicalEncoding],
    ctx: InferContext,
    input_vars: Optional[List[Var]] = None,
    output_var: Optional[Var] = None,
):
    """
    the infer function just forward the logical encoding to the output,
    so we don't actually need to do inference inside this function.
    we just check whether there is conflicts.
    """
    enc = args[0]
    shp = enc.shape
    tv = enc.layout
    in_layout = ctx.var2layout[op.x]
    inpshp = list(in_layout.shape_tuple)
    ls = len(shp)
    li = len(inpshp)
    assert li >= ls
    last_dim = 1
    for i in range(li):
        if i < ls - 1:
            assert inpshp[i] == shp[i]
        elif i >= ls - 1:
            last_dim *= inpshp[i]
    assert shp[ls - 1] == last_dim
    return [infer_result(logical_encoding(shp, tv))]


@register_infer_rules(PartitionA)
class PartitionAInferRules(InferRules):
    def __init__(self):
        super().__init__()

        self.update_infer_rules("i2o", forward).update_infer_rules("o2i", forward)


@register_infer_rules(PartitionSrc)
class PartitionSrcInferRules(InferRules):
    def __init__(self):
        super().__init__()

        self.update_infer_rules("i2o", forward).update_infer_rules("o2i", forward)


@register_infer_rules(PartitionDst)
class PartitionDstInferRules(InferRules):
    def __init__(self):
        super().__init__()

        self.update_infer_rules("i2o", forward).update_infer_rules("o2i", forward)


@register_infer_rules(Reduce)
class ReduceInferRules(InferRules):
    def __init__(self):
        super().__init__()

        def infer_output(
            op: Reduce,
            args: List[LogicalEncoding],
            ctx: InferContext,
            input_vars: Optional[List[Var]] = None,
            output_var: Optional[Var] = None,
        ):
            enc = args[0]
            shp = enc.shape
            tv = enc.layout
            thrd, val = tv
            out_tv = op.infer_layout(shp, thrd, val)
            thrd, val = out_tv.thr_layout(), out_tv.val_layout()
            out_tv = make_layout(thrd, val)
            return [infer_result(logical_encoding(shp, out_tv))]

        self.update_infer_rules("i2o", infer_output)


@register_infer_rules(Arithmetic)
class ArithmeticInferRules(InferRules):
    def __init__(self):
        super().__init__()

        def infer(
            op: Arithmetic,
            args: List[LogicalEncoding],
            ctx: InferContext,
            input_vars: Optional[List[Var]] = None,
            output_var: Optional[Var] = None,
        ):
            enc = args[0]
            shp = enc.shape
            thr, val = enc.layout
            layout = ctx.var2layout[output_var]
            inp_thr = composition(layout, thr)
            inp_val = composition(layout, val)
            inp_enc = logical_encoding(shp, make_layout(inp_thr, inp_val))
            return [infer_result(inp_enc)]

        self.update_infer_rules("i2i", infer)

        def infer2(
            op: Arithmetic,
            args: List[LogicalEncoding],
            ctx: InferContext,
            input_vars: Optional[List[Var]] = None,
            output_var: Optional[Var] = None,
        ):
            from hidet.ir.cute.type import tiled_tensor

            arg_types = []
            for enc, inp in zip(args, op.inputs):
                inp_ty = infer_type(inp)
                shp = enc.shape
                tv = enc.layout
                atom = ThrValAtom("thread_block", shp, tv)
                levels = []
                layout = TiledTensorLayout(atom, levels)
                arg_types.append(tiled_tensor(inp_ty.dtype, layout, inp_ty.scope))
            layout = op.infer_type(arg_types).layout
            shape = layout.shape()
            thr, val = layout.thr_layout(), layout.val_layout()
            layout = make_layout(thr, val)
            return [infer_result(logical_encoding(shape, layout))]

        self.update_infer_rules("i2o", infer2)


@register_infer_rules(Broadcast)
class BroadcastInferRules(InferRules):
    def __init__(self):
        super().__init__()

        def infer_output(
            op: Broadcast,
            args: List[LogicalEncoding],
            ctx: InferContext,
            input_vars: Optional[List[Var]] = None,
            output_var: Optional[Var] = None,
        ):
            enc = args[0]
            x_shape = enc.shape
            x_t, x_v = enc.layout
            enc = args[1]
            trg_shape = enc.shape
            trg_t, trg_v = enc.layout
            tv_lyt = op.infer_layout(x_shape, x_t, x_v, trg_shape, trg_t, trg_v)
            return [infer_result(logical_encoding(trg_shape, tv_lyt))]

        self.update_infer_rules("i2o", infer_output)


class Constraint:
    def __init__(self, inputs: List[Var], output: Var, op: Op, name: str):
        self.inputs = inputs
        self.output = output
        self.op = op
        self.infer_func = get_infer_rules(op).get_infer_func(name)
        self.name: str = name

    def __call__(self, args: List[LogicalEncoding], ctx: InferContext):
        return self.infer_func(self.op, args, ctx, self.inputs, self.output)


def make_constraint(inputs: List[Var], output: Var, op: Op, name: str):
    return Constraint(inputs, output, op, name)


class StackFrame:
    def __init__(
        self,
        constraints: List[Constraint],
        var2logical_encoding: Dict[Var, LogicalEncoding],
        finalize: List[Op] = None,
        solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]] = None,
    ):
        self._constraints: List[Constraint] = constraints.copy()
        self._var2logical_encoding: Dict[Var, LogicalEncoding] = var2logical_encoding.copy()
        if finalize is None:
            self._finalize: List[Op] = []
        else:
            self._finalize: List[Op] = finalize.copy()
        if solution is None:
            self._solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]] = {}
        else:
            self._solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]] = solution.copy()

    def copy(self):
        return StackFrame(self.constraints, self.var2logical_encoding, self.finalize, self.solution)

    @property
    def constraints(self):
        return self._constraints

    @property
    def finalize(self):
        return self._finalize

    @property
    def var2logical_encoding(self):
        return self._var2logical_encoding

    @property
    def solution(self):
        return self._solution


def is_surjective(a: TensorLayout):
    c = coalesce(a)
    return 0 not in c.stride_tuple


class ResolveAuto(IRVisitor):
    def __init__(self, var2layout: Dict[Var, TensorLayout], var2tensor: Dict[Var, TensorInfo]):
        super().__init__()
        self.var2layout: Dict[Var, TensorLayout] = var2layout
        self.var2tensor: Dict[Var, TensorInfo] = var2tensor

        self.ops_resolved: List[Op] = []
        self.ops_unresolved: List[Op] = []
        self.constraints: List[Constraint] = []
        self.ready: List[Constraint] = []

        self.state_stack: List[StackFrame] = []
        self.solutions: List[Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]]] = []

        self.op2vars: Dict[Op, List[Var]] = {}
        self.op2copys: Dict[Union[PartitionSrc, PartitionDst, Mask], List[Copy]] = {}
        self.op2mmas: Dict[Union[PartitionA, PartitionB], List[Mma]] = {}
        self.var2expr: Dict[Var, Union[Var, Expr]] = {}
        self.infer_type = TypeInfer()

        self.threads = None

    def get_partition_op(self, v: Var):
        expr = self.var2expr.get(v, None)
        while isinstance(expr, Var):
            assert expr in self.var2expr
            expr = self.var2expr[expr]
        return expr

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = stmt.var
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            self.visit(call)
            op = call.op
            if op in self.op2vars:
                self.op2vars[op].append(v)
            else:
                self.op2vars[op] = [v]
            if isinstance(op, (Partition, Mask, MBarriers)):
                self.var2expr[v] = op
            elif isinstance(op, SubTensor):
                self.var2expr[v] = op.x
            if op in self.ops_unresolved:
                self.append_constraint(op)

    # TODO
    def visit_AssignStmt(self, stmt: AssignStmt):
        v = stmt.var
        if isinstance(stmt.value, CallOp):
            call = stmt.value
            self.visit(call)
            op = call.op
            if op in self.op2vars:
                self.op2vars[op].append(v)
            else:
                self.op2vars[op] = [v]
            if op in self.ops_unresolved:
                self.append_constraint(op)

    def append_constraint(self, op: Op):
        if isinstance(op, Copy):
            src_ty = self.infer_type(op.src)
            dst_ty = self.infer_type(op.dst)
            is_gmem = src_ty.scope.is_global() or dst_ty.scope.is_global()
            is_smem = src_ty.scope.is_shared() or dst_ty.scope.is_shared()
            if not (is_gmem and is_smem):
                self.constraints.append(make_constraint([op.src], op.dst, op, "i2o"))
                self.constraints.append(make_constraint([op.dst], op.src, op, "o2i"))
        elif isinstance(op, Partition):
            x_ty = self.infer_type(op.x)
            out_var = self.op2vars[op][-1]
            if not x_ty.scope.is_memory():
                self.constraints.append(make_constraint([out_var], op.x, op, "o2i"))
            self.constraints.append(make_constraint([op.x], out_var, op, "i2o"))
        elif isinstance(op, SubTensor):
            out_var = self.op2vars[op][-1]
            self.constraints.append(make_constraint([op.x], out_var, op, "i2o"))
            self.constraints.append(make_constraint([out_var], op.x, op, "o2i"))
        elif isinstance(op, Reduce):
            out_var = self.op2vars[op][-1]
            self.constraints.append(make_constraint([op.x], out_var, op, "i2o"))
        elif isinstance(op, Arithmetic):
            vars = self.op2vars[op]
            narity = len(self.op2vars[op])
            for i in range(narity):
                for j in range(narity):
                    if i != j:
                        var_i = vars[i]
                        var_j = vars[j]
                        layout_i = coalesce(self.var2layout[var_i])
                        layout_j = coalesce(self.var2layout[var_j])
                        if j == narity - 1:
                            self.constraints.append(make_constraint([var_j], var_i, op, "i2i"))
                        elif layout_i == layout_j:
                            self.constraints.append(make_constraint([var_j], var_i, op, "i2i"))
                        elif is_surjective(layout_j):
                            self.constraints.append(make_constraint([var_j], var_i, op, "i2i"))
            self.constraints.append(make_constraint(op.inputs, vars[-1], op, "i2o"))
        elif isinstance(op, Broadcast):
            out_var = self.op2vars[op][-1]
            self.constraints.append(make_constraint([op.x, op.target], out_var, op, "i2o"))
        elif isinstance(op, Mma):
            a_ty = self.infer_type(op.a)
            b_ty = self.infer_type(op.b)
            d, a, b, c = self.op2vars[op]
            if a_ty.scope.is_shared() or b_ty.scope.is_shared():
                self.constraints.append(make_constraint([d, a, b, c], None, op, "shared_ab"))

    def visit_Copy(self, e: Copy):
        self.visit(e.src)
        self.visit(e.dst)
        if e.mask is not None:
            self.visit(e.mask)
        if e.mbarrier is not None:
            self.visit(e.mbarrier)
        src_ty = self.infer_type(e.src)
        dst_ty = self.infer_type(e.dst)
        is_src_auto_layout = src_ty.scope.is_shared() and is_auto_layout(src_ty.layout)
        is_dst_auto_layout = dst_ty.scope.is_shared() and is_auto_layout(dst_ty.layout)
        is_auto_copy_ = is_auto_copy(e.tiled_copy)
        if is_src_auto_layout or is_dst_auto_layout or is_auto_copy_:
            self.ops_unresolved.append(e)
            self.append_constraint(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.src, e.dst]

        def on_var(v: Var):
            assert isinstance(v, Var)
            if v in self.var2expr:
                op = self.get_partition_op(v)
                assert isinstance(op, Op)
                if op not in self.op2copys:
                    self.op2copys[op] = [e]
                else:
                    self.op2copys[op].append(e)

        on_var(e.src)
        on_var(e.dst)
        if e.mask is not None:
            on_var(e.mask)
        if e.mbarrier is not None:
            on_var(e.mbarrier)

    def visit_Mask(self, e: Mask):
        for ex in e.extents:
            self.visit(ex)
        if is_auto_copy(e.tiled_copy):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)

    def visit_TensorBase(self, e: TensorBase):
        if is_auto_layout(e.layout):
            assert e.scope.is_register() or e.scope.is_shared()
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)

    def visit_Tensor(self, e: Tensor):
        self.visit_TensorBase(e)

    def visit_TensorView(self, e: TensorView):
        self.visit_TensorBase(e)

    def visit_PartitionA(self, e: PartitionA):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_mma(e.tiled_mma) or is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_PartitionSrc(self, e: PartitionSrc):
        self.visit(e.x)
        if is_auto_copy(e.tiled_copy):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_PartitionDst(self, e: PartitionDst):
        self.visit(e.x)
        if is_auto_copy(e.tiled_copy):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Mma(self, e: Mma):
        self.visit(e.d)
        self.visit(e.a)
        self.visit(e.b)
        self.visit(e.c)
        self.op2vars[e] = [e.d, e.a, e.b, e.c]
        if is_auto_mma(e.tiled_mma):
            self.ops_unresolved.append(e)
            self.append_constraint(e)
        else:
            a_ty = self.infer_type(e.a)
            b_ty = self.infer_type(e.b)
            if a_ty.scope.is_shared() or b_ty.scope.is_shared():
                self.append_constraint(e)
            self.ops_resolved.append(e)

        def on_var(v: Var):
            assert isinstance(v, Var)
            v_ty = self.infer_type(v)
            if v_ty.scope.is_shared():
                op = self.get_partition_op(v)
                assert isinstance(op, Op)
                if op not in self.op2mmas:
                    self.op2mmas[op] = [e]
                else:
                    self.op2mmas[op].append(e)

        on_var(e.a)
        on_var(e.b)

    def visit_SubTensor(self, e: SubTensor):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Rearrange(self, e: Rearrange):
        self.visit(e.x)
        if is_auto_layout(e.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def visit_Arithmetic(self, e: Arithmetic):
        for i in e.inputs:
            self.visit(i)
        input_tys = [self.infer_type(i) for i in e.inputs]
        if any(is_auto_layout(input_ty.layout) for input_ty in input_tys):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [i for i in e.inputs]

    def visit_Reduce(self, e: Reduce):
        self.visit(e.x)
        x_ty = self.infer_type(e.x)
        if is_auto_layout(x_ty.layout):
            self.ops_unresolved.append(e)
        else:
            self.ops_resolved.append(e)
        self.op2vars[e] = [e.x]

    def _current_state(self):
        assert len(self.state_stack) > 0
        return self.state_stack[-1]

    def _update_ready(self):
        current_state = self._current_state()
        removable = []
        for c in current_state.constraints:
            inferable = True
            for i in c.inputs:
                if i not in current_state.var2logical_encoding:
                    inferable = False
            if inferable:
                if c.output not in current_state.var2logical_encoding:
                    self.ready.append(c)
                else:
                    removable.append(c)
        for c in removable:
            op = c.op
            if op not in current_state.solution:
                self._resolve(op)
            current_state.constraints.remove(c)

    def _get_equivalent_copies(self, op: Op):
        par = self.get_partition_op(op.src)
        return self.op2copys[par]

    def _update_equivalent_ops(self, op: Copy, equivalent_ops: List[Copy], state: StackFrame):
        for copy in equivalent_ops:
            for j, v in enumerate(self.op2vars[copy]):
                if v not in state.var2logical_encoding:
                    ev = self.op2vars[op][j]
                    state.var2logical_encoding[v] = state.var2logical_encoding[ev]

    def _update_state(self, infer_result_: InferResult, constraint_: Constraint, state: StackFrame):
        output = constraint_.output
        if output is not None:
            state.var2logical_encoding[output] = infer_result_[0]
        if infer_result_ is not None and infer_result_.memory_constraints is not None:
            if isinstance(infer_result_.memory_constraints, list):
                for memory_constraint in infer_result_.memory_constraints:
                    tensor = memory_constraint.tensor
                    memory_constraints = memory_constraint.memory_constraints
                    state.solution[tensor] = memory_constraints
            else:
                tensor = infer_result_.memory_constraints.tensor
                memory_constraints = infer_result_.memory_constraints.memory_constraints
                state.solution[tensor] = memory_constraints
        op = constraint_.op
        if isinstance(op, Mma):
            a_ty = self.infer_type(op.a)
            b_ty = self.infer_type(op.b)
            if a_ty.scope.is_shared():
                par = self.get_partition_op(op.a)
            else:
                assert b_ty.scope.is_shared()
                par = self.get_partition_op(op.b)
            equivalents = self.op2mmas[par]
            assert op in equivalents
            for equivalent_constraint in state.constraints:
                if equivalent_constraint.op in equivalents:
                    state.constraints.remove(equivalent_constraint)
            return
        vars = self.op2vars[op]
        is_resolved = all(v in state.var2logical_encoding for v in vars)
        if is_resolved:
            if isinstance(op, Copy):
                ops = self._get_equivalent_copies(op)
                self._update_equivalent_ops(op, ops, state)
            else:
                ops = [op]
            for _op in ops:
                if not isinstance(_op, (PartitionSrc, PartitionDst, Mask)):
                    self._resolve(_op, state)
                else:
                    state.finalize.append(_op)
        state.constraints.remove(constraint_)

    def _resolve_ready(self):
        for c in self.ready:
            current_state = self._current_state()
            args = []
            for i in c.inputs:
                assert i in current_state.var2logical_encoding
                args.append(current_state.var2logical_encoding[i])
            ctx = infer_context(self.var2layout, self.var2tensor, current_state.solution)
            results = c(args, ctx)
            nr_results = len(results)
            if nr_results == 0:
                self.state_stack.pop()
                self.ready = []
                return False
            elif nr_results > 1:
                self.state_stack.pop()
                new_states = [current_state.copy() for _ in range(nr_results)]
                for i, res in enumerate(results):
                    state = new_states[i]
                    self._update_state(res, c, state)
                    self.state_stack.append(state)
            else:
                res = results[0]
                self._update_state(res, c, current_state)
        self.ready = []
        return True

    def _resolve(self, op: Op, state: StackFrame = None):
        current_state = self._current_state() if state is None else state
        if isinstance(op, Copy):
            src = current_state.var2logical_encoding[op.src]
            shp = src.shape
            src_tv = src.layout
            dst_tv = current_state.var2logical_encoding[op.dst].layout
            atom = CopyAtom("thread_block", shp, src_tv, dst_tv)
            current_state.solution[op] = TiledCopy(atom)
        elif isinstance(op, (PartitionSrc, PartitionDst, Mask)):
            assert op in self.op2copys
            copy = self.op2copys[op][0]
            if copy in current_state.solution:
                assert copy in current_state.solution
                current_state.solution[op] = current_state.solution[copy]
        elif isinstance(op, TensorBase):
            ovar = self.op2vars[op][-1]
            enc = current_state.var2logical_encoding[ovar]
            shp = enc.shape
            tv = enc.layout
            atom = ThrValAtom("thread_block", shp, tv)
            levels = []
            current_state.solution[op] = TiledTensorLayout(atom, levels)
        elif isinstance(op, Rearrange):
            ovar = self.op2vars[op][-1]
            enc = current_state.var2logical_encoding[ovar]
            shp = enc.shape
            tv = enc.layout
            atom = ThrValAtom("thread_block", shp, tv)
            levels = []
            current_state.solution[op] = TiledTensorLayout(atom, levels)
        elif isinstance(op, (SubTensor, Arithmetic, Reduce, PartitionA)):
            pass
        else:
            raise NotImplementedError(f"No rule can be used to resolve {op}")

    def _schedule_tiled_copy(
        self,
        dtype: DataType,
        tile_layout: TensorLayout,
        bits_per_memory_inst: int,
        extra_memory_hint: Optional[TensorLayout] = None,
        extra_memory_constraints: Optional[TensorLayout] = None,
        num_threads: Optional[int] = None,
    ):
        if num_threads is None:
            num_threads = self.threads

        tile_shape = product_each(tile_layout.shape)
        shape = flatten(tile_layout.shape_tuple)
        stride = flatten(tile_layout.stride_tuple)
        costride = list(compact_col_major(shape))
        shape = list(shape)
        stride = list(stride)
        index = range(len(shape))
        cosize = tile_layout.cosize()

        def key(x):
            return x[0] if x[0] > 0 else cosize

        sorted_dsi = sorted(zip(stride, shape, index))
        vector_size = 0
        vector_dim_list = []
        non_vector_strides = []
        for d, s, i in sorted_dsi:
            if d == 0:  # stride that equals 0 indicates we need broadcasting this tensor
                costride[i] = 0
            elif d == 1 and vector_size == 0:
                vector_size = s
                vector_dim_list.append(i)
            elif d == vector_size:
                vector_size *= s
                vector_dim_list.append(i)
            else:
                non_vector_strides.append(d)

        alignment = [i * dtype.nbits for i in non_vector_strides]
        bits_per_inst = gcd(dtype.nbits * vector_size, bits_per_memory_inst, *alignment)
        if bits_per_inst < bits_per_memory_inst:
            return None

        elements_per_inst = bits_per_inst // dtype.nbits
        val_shape = []
        val_stride = []
        val_dims = []
        for i in vector_dim_list:
            if elements_per_inst >= shape[i]:
                assert elements_per_inst % shape[i] == 0
                elements_per_inst //= shape[i]
                val_shape.append(shape[i])
                val_stride.append(costride[i])
                val_dims.append(i)
            else:
                val_shape.append(elements_per_inst)
                val_stride.append(costride[i])
                shape[i] //= elements_per_inst
                stride[i] *= elements_per_inst
                costride[i] *= elements_per_inst
                break

        memory_constraints = None
        if extra_memory_hint is not None:
            val = TensorLayout(tuple(val_shape), tuple(val_stride))
            unifier = MemoryConstraintsUnifier()
            memory_constraints = unifier.infer(extra_memory_hint, val)
            if extra_memory_constraints is not None:
                memory_constraints = unifier.unify(extra_memory_constraints, memory_constraints)
            if memory_constraints is None:
                return None

        for i in sorted(val_dims, key=lambda x: -x):
            shape.pop(i)
            stride.pop(i)
            costride.pop(i)

        thr_shape = []
        thr_stride = []
        remaining_threads = num_threads
        sorted_dsi = sorted(zip(stride, shape, costride), key=key)

        for _, s, d in sorted_dsi:
            if remaining_threads > 1:
                s1 = gcd(remaining_threads, s)
                if s1 > 1:
                    thr_shape.append(s1)
                    thr_stride.append(d)
                    remaining_threads //= s1
                    s = s // s1
                    d = d * s1
                if s > 1:
                    val_shape.append(s)
                    val_stride.append(d)
            else:
                val_shape.append(s)
                val_stride.append(d)
        # assert remaining_threads == 1
        if remaining_threads > 1:
            thr_shape.append(remaining_threads)
            thr_stride.append(0)
        thr_layout = TensorLayout(tuple(thr_shape), tuple(thr_stride))
        val_layout = TensorLayout(tuple(val_shape), tuple(val_stride))
        try:
            # check divisibility
            if memory_constraints is not None:
                composition(TensorLayout(memory_constraints.shape), val_layout)
        except AssertionError:
            return None
        copy_atom = CopyAtom("thread_block", tile_shape, make_layout(thr_layout, val_layout))
        return TiledCopy(copy_atom), memory_constraints

    def _schedule_tma_copy(
        self,
        dtype: DataType,
        global_layout: TensorLayout,
        tile_shape: Tuple[int, ...],
        gmem_tile_layout: TensorLayout,
        extra_memory_hint: TensorLayout,
        extra_memory_constraints: TensorLayout,
        num_threads: int,
    ):
        """
        Determine an optimal shared memory layout to minimize the rank of the TMA tensor.

        This function searches for a valid shared memory layout that allows TMA (Tensor Memory Access)
        to be used for a given global tensor and tiling configuration. The goal is to reduce the number
        of dimensions (rank) required by the TMA tensor, while satisfying all hardware and software constraints.

        If a valid layout is found, the function returns both the configured TMA copy operation and
        the corresponding memory constraint layout. Otherwise, it returns `None`.

        ### Returned Memory Constraints
        - **Constrained strides** (fixed by TMA hardware): Actual computed stride values
        - **Unconstrained strides** (flexible): Represented symbolically using variables (e.g., `'v'`)

        ### Parameters:
        - **dtype** (`DataType`):
          The data type of the tensor involved in the TMA operation.

        - **global_layout** (`TensorLayout`):
          The layout of the tensor in global memory.

        - **tile_shape** (`Tuple[int]`):
          The shape of the tile used for the TMA copy.

        - **gmem_tile_layout** (`TensorLayout`):
          The layout of the global memory tile.

        - **extra_memory_hint** (`Tuple`):
          A hint describing the expected shared memory shape or stride layout.

        - **extra_memory_constraints** (`TensorLayout`):
          Constraints imposed by other memory copies (e.g., load/store conflicts).

        - **num_threads** (`int`):
          The number of threads participating in the copy.
          - For non-warp-specialized kernels: threads per block
          - For warp-specialized kernels: threads per warp group (producer/consumer)

        ### Returns:
        - `Optional[Tuple[TiledCopy, TensorLayout]]`:
          A tuple of:
          1. `TiledCopy`: The configured TMA-based copy operation
          2. `TensorLayout`: The shared memory layout with stride constraints
          Returns `None` if no valid layout satisfying all constraints is found.
        """
        # find a shared memory layout such that the rank of the tma tensor is the smallest
        # Step 1. We align the shape of global memory layout and shared memory layout
        last_dim_strides = get_last_dim_strides(tile_shape, global_layout)

        smem_last_dim_strides = [None] * len(tile_shape)
        if extra_memory_constraints is None:
            smem_layout = coalesce_per_dim(extra_memory_hint, tile_shape, smem_last_dim_strides)
        else:
            smem_layout = coalesce_per_dim(extra_memory_constraints, tile_shape, smem_last_dim_strides)
        gmem_layout = coalesce_per_dim(gmem_tile_layout, tile_shape, last_dim_strides)

        divisor = 1
        MAX_ELEMENTS_PER_DIM = 256
        if dtype.is_integer_subbyte():
            divisor = dtype.storage.nbits // dtype.nbits
            MAX_ELEMENTS_PER_DIM = 256 * divisor

        gmem_layout, smem_layout = common_reshape_per_dim(gmem_layout, smem_layout)

        gmem_shape = flatten(gmem_layout.shape_tuple)
        gmem_stride = flatten(gmem_layout.stride_tuple)
        smem_shape = flatten(smem_layout.shape_tuple)
        smem_stride = flatten(smem_layout.stride_tuple)
        index = range(len(gmem_shape))

        # Step 2. Ensure the shared memory (smem) layout is contiguous by sorting dimensions
        # based on the ascending order of smem strides.
        #
        # Goal:
        #   Identify a shared memory layout that minimizes the rank (number of dimensions)
        #   of the resulting TMA tensor.
        #
        # Explanation:
        # - For dimensions with **known (constrained) smem strides** (typically from memory constraints
        #   imposed by other copy operations), we must preserve their order and sort them by increasing stride.
        #
        # - For dimensions with **unknown (unconstrained) smem strides**, we sort them based on
        #   the ascending order of the **global memory (gmem) strides**. This heuristic maximizes
        #   the chance of coalescing gmem dimensions — reducing TMA rank.
        #
        # - Important: Two dimensions can be merged (coalesced) only if **both** the gmem and smem
        #   strides are contiguous. i.e.
        #   smem_stride[i + 1] = smem_shape[i] * smem_stride[i]
        #   gmem_stride[i + 1] = gmem_shape[i] * gmem_stride[i]
        #
        # - When smem strides are unknown, we initialize them using contiguous strides
        #   **in the same order as gmem strides** to maintain mergeability and alignment.
        #
        # Outcome:
        #   This strategy yields a valid shared memory layout with minimal TMA rank.
        #   While multiple valid layouts may exist with the same rank, this method currently can
        #   only selects one.
        smem_stride, smem_shape, gmem_stride, gmem_shape, permute = sort_dims(
            smem_stride, smem_shape, gmem_stride, gmem_shape, index
        )

        sorted_shape = smem_shape

        # Merge the dimensions of the global memory and shared memory if the strides are contiguous.
        gmem_shape, gmem_stride, smem_shape, smem_stride, _ = coalesce_gmem_shape_and_smem_shape(
            gmem_shape, smem_shape, gmem_stride, smem_stride
        )

        innermost_shape = gmem_shape[0] * dtype.nbits // 8
        if gmem_stride[0] != 1 or innermost_shape % 16 != 0:
            return None

        # Determine the contiguous strides for the shared memory layout.
        smem_shape, smem_stride = make_contiguous_stride(smem_shape, smem_stride)

        # check alignment
        for ds, dg in zip(smem_stride[1:], gmem_stride[1:]):
            smem_bytes = ds * dtype.nbits // 8
            gmem_bytes = dg * dtype.nbits // 8
            if (ds != 1 and smem_bytes % 16 != 0) or (dg != 1 and gmem_bytes % 16 != 0):
                return None

        # Merge the dimensions again since the shared memory strides are updated.
        gmem_shape, gmem_stride, smem_shape, smem_stride, _ = coalesce_gmem_shape_and_smem_shape(
            gmem_shape, smem_shape, gmem_stride, smem_stride
        )

        # Step 3. Split the dimension if the number of elements in the dimension is larger than 256
        gmem_shape, gmem_stride, smem_shape, smem_stride, _ = split_shapes(
            gmem_shape, smem_shape, gmem_stride, smem_stride, MAX_ELEMENTS_PER_DIM
        )

        gmem_layout = TensorLayout(tuple(gmem_shape), tuple(gmem_stride))
        smem_layout = TensorLayout(tuple(smem_shape), tuple(smem_stride))
        # check if smem layout is contiguous
        smem_layout_ = coalesce(smem_layout)
        if is_tuple(smem_layout_.stride):
            if len(smem_layout_.stride) != 1:
                return None
            smem_stride = smem_layout_.stride[0]
        else:
            smem_stride = smem_layout_.stride
        if smem_stride != 1:
            return None
        dim = rank(gmem_layout.shape)
        # tma only supports tensor dimensions less than and equal to 5
        if dim > 5:
            return None

        # Step 4. Convert the shared memory layout candidate to a memory constraint layout.
        memory_constraints = construct_memory_constraint(sorted_shape, smem_layout, permute, extra_memory_hint)
        unifier = MemoryConstraintsUnifier()
        memory_constraints = unifier.unify(extra_memory_constraints, memory_constraints)
        if memory_constraints is None:
            return None
        thread_layout = TensorLayout(tuple([num_threads]), tuple([0]))
        value_layout = TensorLayout(tile_shape)
        copy_atom = CopyAtom("thread_block", tuple(tile_shape), make_layout(thread_layout, value_layout))
        return TiledCopy(copy_atom, []), memory_constraints

    def _coalesce_memory_access(self, e: Copy):
        src_ty = self.infer_type(e.src)
        dst_ty = self.infer_type(e.dst)
        if src_ty.scope.is_global():
            src_tensor = self.var2tensor[e.src]
            gmem_layout = src_tensor.tensor.layout
            gmem_tile_layout = src_tensor.layout
            dtype = src_ty.dtype
        else:
            dst_tensor = self.var2tensor[e.dst]
            gmem_layout = dst_tensor.tensor.layout
            gmem_tile_layout = dst_tensor.layout
            dtype = dst_ty.dtype

        current_state = self._current_state()
        shared_tensor_info = None
        if src_ty.scope.is_shared():
            shared_tensor_info = self.var2tensor[e.src]
        elif dst_ty.scope.is_shared():
            shared_tensor_info = self.var2tensor[e.dst]
        shared_tensor = shared_tensor_info.tensor if shared_tensor_info is not None else None
        extra_memory_hint = shared_tensor_info.layout if shared_tensor_info is not None else None
        if shared_tensor is not None and is_auto_layout(extra_memory_hint):
            extra_memory_constraints = (
                current_state.solution[shared_tensor] if shared_tensor in current_state.solution else None
            )
        else:
            extra_memory_constraints = None

        if "group_ids" in e.annotations:
            num_threads = e.annotations["group_threads"]
        else:
            num_threads = self.threads

        tile_shape = e.tiled_copy.shape
        sche = None
        mbarrier = e.mbarrier
        mask = e.mask
        # Step 1. Try to schedule the copy operation using TMA
        if shared_tensor is not None and mask is None and mbarrier is not None:
            sche = self._schedule_tma_copy(
                dtype,
                gmem_layout,
                tile_shape,
                gmem_tile_layout,
                shared_tensor_info.layout,
                extra_memory_constraints,
                num_threads,
            )
            if not is_auto_copy(e.tiled_copy) and sche is None:
                return False

        # Step 2. Try to schedule the copy operation using normal cp_async
        if sche is None:
            tile = TensorLayout(tile_shape)
            tile_layout = composition(gmem_tile_layout, tile)
            candidates = []
            for inst in memory_instructions:
                if inst.src_scope == src_ty.scope and inst.dst_scope == dst_ty.scope:
                    candidates.append(inst)
            candidates = sorted(candidates, key=lambda x: -x.bytes_per_inst)
            for inst in candidates:
                bits_per_memory_inst = inst.alignment * 8
                if extra_memory_constraints is not None:
                    sche = self._schedule_tiled_copy(
                        dtype,
                        tile_layout,
                        bits_per_memory_inst,
                        shared_tensor_info.layout,
                        extra_memory_constraints,
                        num_threads,
                    )
                else:
                    sche = self._schedule_tiled_copy(dtype, tile_layout, bits_per_memory_inst, num_threads=num_threads)
                if sche is not None:
                    break

        if sche is None:
            return False
        tiled_copy, memory_constraints = sche
        current_state.solution[e] = tiled_copy
        # update memory constraints
        if shared_tensor is not None and memory_constraints is not None:
            current_state.solution[shared_tensor] = memory_constraints

        assert e.src not in current_state.var2logical_encoding and e.dst not in current_state.var2logical_encoding
        shp, tv = tiled_copy.src_tv_layout()
        tv = make_layout(tv[0][0], coalesce(make_layout(tv[0][1], tv[1])))
        current_state.var2logical_encoding[e.src] = logical_encoding(shp, tv)
        shp, tv = tiled_copy.dst_tv_layout()
        tv = make_layout(tv[0][0], coalesce(make_layout(tv[0][1], tv[1])))
        current_state.var2logical_encoding[e.dst] = logical_encoding(shp, tv)

        ops = self._get_equivalent_copies(e)
        for op in ops:
            current_state.solution[op] = tiled_copy
        self._update_equivalent_ops(e, ops, current_state)
        return True

    def _materialize_memory_layout(self, solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]]):
        for op, layout in solution.items():
            if isinstance(op, TensorBase):
                if op.scope.is_shared():
                    flat_shape = flatten(layout.shape_tuple)
                    flat_stride = flatten(layout.stride_tuple)
                    profile = [s * d for s, d in zip(flat_shape, flat_stride) if is_constant(d)]
                    profile = max(profile) if len(profile) > 0 else 1
                    lst = []
                    for i, mode in enumerate(layout):
                        min_d = profile
                        for d in flatten(mode.stride_tuple):
                            if is_constant(d):
                                min_d = min(d, min_d)
                        if min_d == 1:
                            lst.insert(0, (i, mode))
                        else:
                            lst.append((i, mode))
                    modes = [TensorLayout(1)] * len(lst)
                    for i, mode in lst:
                        shape = mode.shape_tuple
                        stride = mode.stride_tuple
                        new_stride = []
                        for s, d in zip(shape, stride):
                            if is_constant(d):
                                new_stride.append(d)
                            else:
                                new_stride.append(profile)
                                profile *= s
                        modes[i] = coalesce(TensorLayout(shape, new_stride))
                    solution[op] = make_layout(*modes)
                else:
                    assert op.scope.is_register()
                    assert isinstance(layout, TiledTensorLayout)

    def _select_unresolved_copy(self):
        current_state = self._current_state()
        unresolved_copys = [
            op for op in self.ops_unresolved if op not in current_state.solution and isinstance(op, Copy)
        ]

        def f(op: Copy):
            src_ty = self.infer_type(op.src)
            dst_ty = self.infer_type(op.dst)
            is_gmem = src_ty.scope.is_global() or dst_ty.scope.is_global()
            is_smem = src_ty.scope.is_shared() or dst_ty.scope.is_shared()
            has_auto = is_auto_copy(op.tiled_copy)
            return is_gmem and has_auto and not is_smem

        unresolved_copys = [op for op in unresolved_copys if f(op)]

        def g(op: Copy):
            src_ty = self.infer_type(op.src)
            dst_ty = self.infer_type(op.dst)
            if src_ty.scope.is_global():
                src_tensor = self.var2tensor[op.src]
                gmem = src_tensor.layout
            else:
                assert dst_ty.scope.is_global()
                dst_tensor = self.var2tensor[op.dst]
                gmem = dst_tensor.layout
            if is_surjective(gmem):
                return 0
            else:
                return 1

        unresolved_copys = sorted(unresolved_copys, key=g)
        if len(unresolved_copys) > 0:
            return unresolved_copys[0]

    def _finalize_global_shared_copy(self):
        state = self._current_state()
        unresolved_copys = [op for op in self.ops_unresolved if op not in state.solution and isinstance(op, Copy)]
        visited = set()
        for op in unresolved_copys:
            if op in visited:
                continue
            src_ty = self.infer_type(op.src)
            dst_ty = self.infer_type(op.dst)
            is_gmem = src_ty.scope.is_global() or dst_ty.scope.is_global()
            is_smem = src_ty.scope.is_shared() or dst_ty.scope.is_shared()
            assert is_gmem and is_smem
            top_frame_valid = self._coalesce_memory_access(op)
            if not top_frame_valid:
                return False
            ops = self._get_equivalent_copies(op)
            for copy in ops:
                visited.add(copy)
        return True

    def resolve(self, func: Function):
        if func.kind == "cuda_kernel":
            assert "cuda.block_dim" in func.attrs
            block_dim = func.attrs["cuda.block_dim"]
            self.threads = block_dim
        self.visit(func)

        if len(self.ops_unresolved) == 0:
            return self.solutions

        var2logical_encoding: Dict[Var, LogicalEncoding] = {}
        for op in self.ops_resolved:
            res = op.resolve_logical_encoding()
            if isinstance(op, (Rearrange, PartitionSrc, PartitionDst, Arithmetic, Copy, Mask, SubTensor)):
                pass
            elif isinstance(res, NotImplementedError):
                raise NotImplementedError(
                    f"Missing resolve_logical_encoding method for the following operator: \n{type(op).__name__}"
                )
            else:
                for i, v in enumerate(self.op2vars[op]):
                    if res[i] is not None:
                        var2logical_encoding[v] = res[i]
        stack_frame = StackFrame(self.constraints, var2logical_encoding)
        self.state_stack.append(stack_frame)

        while len(self.state_stack) > 0:
            # Step 1. Iteratively find the constraint that is ready (i.e.,
            #         all TV layouts in the right hand side have been resolved)
            #         and apply the infer rule to resolve the TV layout in the
            #         left hand side.
            self._update_ready()
            while len(self.ready) > 0:
                self._resolve_ready()
                self._update_ready()

            # Step 2. Resolve the partition and mask operations because they
            #         depend on the TiledCopy of their corresponding copy
            #         operations.
            current_state = self._current_state()
            for op in current_state.finalize:
                self._resolve(op)

            # Step 3. If there is no more ready constraints, we use the
            #         memory coalescing heuristic to instantiate the TiledCopy
            #         for the unresolved copy operations. This can make the
            #         inference progress.
            # Note: Each time we could only resolve one copy operation because
            #       copy operations may have dependencies on each other.
            top_frame_valid = True
            op = self._select_unresolved_copy()
            while op is not None:
                top_frame_valid = self._coalesce_memory_access(op)
                if not top_frame_valid:
                    self.state_stack.pop()
                    break

                # Step 4. Repeatedly apply the constraint and infer new TV layouts
                #         until no more constraints can be resolved.
                self._update_ready()
                while len(self.ready) > 0:
                    top_frame_valid = self._resolve_ready()
                    if not top_frame_valid:
                        break
                    self._update_ready()
                if not top_frame_valid:
                    break

                current_state = self._current_state()
                for op in current_state.finalize:
                    self._resolve(op)

                op = self._select_unresolved_copy()

            if not top_frame_valid:
                continue
            top_frame_valid = self._finalize_global_shared_copy()
            if not top_frame_valid:
                self.state_stack.pop()
                continue

            # Step 5. If all constraints are resolved, we find a solution and
            #         pop the current state from the stack.
            state = self.state_stack.pop()
            logger.debug("pop solution ==========================================")
            for k, v in state.var2logical_encoding.items():
                logger.debug(k)
                logger.debug(v.layout)
            logger.debug("pop solution ==========================================")
            self._materialize_memory_layout(state.solution)
            for op in self.ops_unresolved:
                if op not in state.solution:
                    if isinstance(op, (TensorBase, Rearrange, PartitionSrc, PartitionDst, Mask)):
                        self._resolve(op, state)
                    else:
                        assert isinstance(op, (SubTensor, Arithmetic, Reduce, PartitionA)), f"unreachable.(op:{op})"

            self.solutions.append(state.solution)

        return self.solutions


class MaterializeAuto(IRRewriter):
    def __init__(self, solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]]):
        super().__init__()
        self.solution: Dict[Op, Union[TensorLayout, TiledCopy, TiledMma]] = solution
        self.old2new: Dict[Var, Var] = {}
        self.infer_type = TypeInfer()

    def visit_Var(self, v: Var):
        if v in self.old2new:
            return self.old2new[v]
        return super().visit_Var(v)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
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

    def visit_Copy(self, e: Copy):
        if is_auto_copy(e.tiled_copy) and e in self.solution:
            src = self.visit(e.src)
            dst = self.visit(e.dst)
            if e.mask is not None:
                mask = self.visit(e.mask)
            else:
                mask = None
            if e.mbarrier is not None:
                mbarrier = self.visit(e.mbarrier)
            else:
                mbarrier = None
            tiled_copy = self.solution[e]
            assert isinstance(tiled_copy, TiledCopy)
            return e.reforward([src, dst, mask, mbarrier], attrs_update={"tiled_copy": tiled_copy})
        return super().visit_Copy(e)

    def visit_Tensor(self, e: Tensor):
        dtype = self.visit(e.dtype)
        if is_auto_layout(e.layout) and e in self.solution:
            layout = self.solution[e]
        else:
            layout = self.visit_Layout(e.layout)
        if dtype is e.dtype and layout is e.layout:
            return e
        else:
            assert dtype is e.dtype
            return e.reforward([], attrs_update={"layout": layout})

    def visit_TensorView(self, e: TensorView):
        x = self.visit(e.x)
        if is_auto_layout(e.layout) and e in self.solution:
            layout = self.solution[e]
        else:
            layout = self.visit_Layout(e.layout)
        if x is e.x and layout is e.layout:
            return e
        else:
            return e.reforward([x], attrs_update={"layout": layout})

    def visit_PartitionSrc(self, e: PartitionSrc):
        x = self.visit(e.x)
        if is_auto_copy(e.tiled_copy) and e in self.solution:
            tiled_copy = self.solution[e]
            assert isinstance(tiled_copy, TiledCopy)
            return e.reforward([x], attrs_update={"tiled_copy": tiled_copy})
        else:
            if x is e.x:
                return e
            else:
                return e.reforward([x])

    def visit_PartitionDst(self, e: PartitionDst):
        x = self.visit(e.x)
        if is_auto_copy(e.tiled_copy) and e in self.solution:
            tiled_copy = self.solution[e]
            assert isinstance(tiled_copy, TiledCopy)
            return e.reforward([x], attrs_update={"tiled_copy": tiled_copy})
        else:
            if x is e.x:
                return e
            else:
                return e.reforward([x])

    def visit_Mask(self, e: Mask):
        extents = [self.visit(v) for v in e.extents]
        if is_auto_copy(e.tiled_copy) and e in self.solution:
            tiled_copy = self.solution[e]
            assert isinstance(tiled_copy, TiledCopy)
            return e.reforward(extents, attrs_update={"tiled_copy": tiled_copy})
        else:
            if all(x is y for x, y in zip(extents, e.extents)):
                return e
            else:
                return e.reforward(extents)

    def visit_Mma(self, e: Mma):
        args = [self.visit(arg) for arg in e.args]
        if is_auto_mma(e.tiled_mma) and e in self.solution:
            tiled_mma = self.solution[e]
            assert isinstance(tiled_mma, TiledMma)
            return e.reforward(args, attrs_update={"tiled_mma": tiled_mma})
        else:
            if all(x is y for x, y in zip(args, e.args)):
                return e
            else:
                return e.reforward(args)

    def visit_Rearrange(self, e: Rearrange):
        x = self.visit(e.x)
        if is_auto_layout(e.layout) and e in self.solution:
            layout = self.solution[e]
            assert isinstance(layout, TiledTensorLayout)
        else:
            layout = self.visit_Layout(e.layout)
        if x is e.x and layout is e.layout:
            return e
        else:
            return e.reforward([x], attrs_update={"layout": layout})


class InstantiateAutoAnnotationPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        orig_level = None
        orig_handler_level = None
        if verbose:
            orig_level = logger.level
            orig_handler_level = stderr_handler.level
            logger.setLevel(DEBUG)
            setConsoleLevel(DEBUG)

        marker = MarkUnresolved()
        ops_resolved, ops_unresolved, op2vars = marker.mark(func)
        if len(ops_unresolved) == 0:
            return func

        infer_logical_shape = InferLogicalShape(ops_resolved, ops_unresolved, op2vars)
        var2shape = infer_logical_shape.infer()

        infer_layout = InferLogicalLayout(var2shape, ops_resolved, ops_unresolved, op2vars)
        var2layout = infer_layout.infer()

        tensor_alias_analysis = TensorAliasAnalysis()
        var2tensor = tensor_alias_analysis.analyze(func)

        solver = ResolveAuto(var2layout, var2tensor)
        solutions = solver.resolve(func)

        str2func: Dict[str, Function] = {}
        nr_solutions = len(solutions)
        for i in range(nr_solutions):
            rewriter = MaterializeAuto(solutions[i])
            new_func = rewriter.rewrite(func)
            key = str(new_func)
            if key not in str2func:
                str2func[key] = new_func
        nr_solutions = len(str2func.items())
        if nr_solutions == 1:
            return str2func.popitem()[1]

        from .cost_model import LatencyModel
        from .instruction_selection import instruction_selection_pass
        from .resolve_bank_conflict import resolve_bank_conflict_pass

        model = LatencyModel()
        func2lat: Dict[Function, float] = {}
        for _, fn in str2func.items():
            transforms = [instruction_selection_pass(), resolve_bank_conflict_pass()]
            f = None
            for ps in transforms:
                if f is None:
                    f = ps.process_func(fn)
                else:
                    f = ps.process_func(f)
            lat = model.predict(f)
            func2lat[fn] = lat
        funcs = sorted(func2lat.keys(), key=lambda x: func2lat[x])
        func = funcs[0]

        if verbose:
            logger.debug(f"{func}")
            stderr_handler.setLevel(orig_handler_level)
            logger.setLevel(orig_level)

        return func


def instantiate_auto_annotation_pass() -> FunctionPass:
    return InstantiateAutoAnnotationPass()
