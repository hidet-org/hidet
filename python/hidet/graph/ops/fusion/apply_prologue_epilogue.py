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
from typing import Dict, List, Optional, Tuple, Union, Sequence
from collections import defaultdict
from functools import partial
import os

import hidet.option
from hidet.ir.compute import TensorNode, GridCompute, TensorInput, tensor_input
from hidet.ir.type import BaseType, tensor_pointer_type
from hidet.ir.expr import (
    Expr,
    Var,
    TensorElement,
    Call,
    tensor_element,
    var,
    tensor_pointer_var,
    is_constant,
    tensor_var,
    if_then_else,
)
from hidet.ir.stmt import Stmt, DeclareStmt, EvaluateStmt, AssignStmt, SeqStmt, BufferStoreStmt, LaunchKernelStmt
from hidet.ir.dtypes import int64
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives.runtime import request_cuda_workspace, request_cpu_workspace
from hidet.ir.task import Task, InverseMap, Target
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.tools import rewrite, collect, rename_funcs, simplify, infer_type
from hidet.ir.utils.call_graph import CallGraph
from hidet.transforms import Pass
from hidet.graph import FlowGraph, Operator, Tensor
from hidet.utils import strict_zip, prod
from hidet.utils.structure import DirectedGraph
from hidet.graph.graph_utils.functors import GraphVisitor

from hidet.graph.ops.transform import CastOp, ReshapeTask, RearrangeTask
from hidet.graph.ops.arithmetic import (
    UnaryElementwiseOp,
    BinaryElementwiseOp,
    CompositeElementwiseOp,
    WhereOp,
    WhereTensorScalarOp,
    WhereScalarTensorOp,
    WhereScalarScalarOp,
)
from hidet.ir.utils import broadcast_shapes
from hidet.ir.dtypes import i32
from hidet.ir.type import DataType, PointerType

from hidet.ir import cute
from hidet.ir.cute import (
    TensorLayout,
    compact_row_major,
    compact_col_major,
    composition,
    coalesce,
    make_layout,
    complement,
    canonicalize,
    CopyAtom,
    TiledCopy,
    ThrValAtom,
    TiledTensorLayout,
    is_auto_layout,
)
from hidet.ir.cute.collective import CollectiveStore
from hidet.ir.cute.ops import partition_src, partition_dst, copy, mask, tensor_view, rearrange, arithmetic, Rearrange
from hidet.ir.cute.expr import CallOp
from hidet.ir.cute.type import TiledTensorType
from hidet.transforms import lower_with
from hidet.transforms.cute.cuda.lower_cute_dialect import LowerCuteDialectRewriter

from .fused_operator import FusedTask


TENSOR_MANIPULATION = (ReshapeTask, RearrangeTask)
ELEMENTWISE = (
    CastOp,
    UnaryElementwiseOp,
    BinaryElementwiseOp,
    CompositeElementwiseOp,
    WhereOp,
    WhereTensorScalarOp,
    WhereScalarTensorOp,
    WhereScalarScalarOp,
)


class Prologue:
    """
    An input tensor of an operator can have a prologue, which defines how each element of the input tensor is computed.

    For example, consider the following operator:
    Subgraph with two operators a and b:
        a[i, j] = i * j + i - j (0 <= i < 10, 0 <= j < 10)
        b[i, j] = a[j, i] + a[i, j] (0 <= i < 10, 0 <= j < 10)

        If we take the operator b as the anchor operator, the prologue of b is:
            inputs: [a]
            axes: [i, j]
            expr: a[j, i] + a[i, j]
    """

    def __init__(self, inputs: List[TensorInput], axes: List[Var], expr: Expr):
        self.inputs: List[TensorInput] = inputs
        self.axes: List[Var] = axes
        self.expr: Expr = expr


class Epilogue:
    """
    An output tensor of an operator can have an epilogue, which defines how each element of the output tensor can be
    further computed to produce the result of an element of the output tensor of a sub-graph.

    For example, consider the following operator:
    Subgraph with two operators a and b:
        a[i, j] = i * j + i - j (0 <= i < 10, 0 <= j < 10)
        b[i, j] = a[j, i] + i - j (0 <= i < 10, 0 <= j < 10)

        Then, if we take the operator a as the anchor operator, the epilogue of a is:
            inputs: []
            axes: [i, j]
            value: v
            out_tensor: b
            out_indices: [j, i]
            out_expr: v + j - i  (pay attention to the sign of i and j)
    """

    def __init__(
        self,
        inputs: List[TensorInput],
        axes: List[Var],
        value: Var,
        out_tensor: Tensor,
        out_indices: List[Expr],
        out_expr: Expr,
    ):
        self.inputs: List[TensorInput] = inputs
        self.axes: List[Var] = axes
        self.value: Var = value
        self.out_tensor: Tensor = out_tensor
        self.out_indices: List[Expr] = out_indices
        self.out_expr: Expr = out_expr


class PrologueEpilogueExtractor(IRRewriter):
    def __init__(self, fused_task: FusedTask):
        super().__init__()
        self.graph: FlowGraph = fused_task.fused_graph
        self.anchor_operator: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.anchor_task: Task = self.anchor_operator.task

        self.tensor_map: Dict[TensorNode, Tensor] = {}
        for node in self.graph.nodes:
            for task_tensor, tensor in zip(node.task.params, node.inputs + node.outputs):
                self.tensor_map[task_tensor] = tensor

        self.consume: Dict[TensorInput, TensorNode] = {}
        self.consume_by: Dict[TensorNode, List[TensorInput]] = {}
        for node in self.graph.nodes:
            for task_input, tensor in zip(node.task.inputs, node.inputs):
                if tensor.trace is None:
                    continue
                producer: Operator = tensor.op
                producer_output = producer.task.outputs[tensor.trace[1]]
                self.consume[task_input] = producer_output
                if producer_output not in self.consume_by:
                    self.consume_by[producer_output] = []
                self.consume_by[producer_output].append(task_input)

        self.input2task: Dict[TensorInput, Task] = {}
        for node in self.graph.nodes:
            for task_input in node.task.inputs:
                self.input2task[task_input] = node.task

    def extract(self) -> Tuple[Dict[Tensor, Prologue], Dict[Tensor, Epilogue], Dict[TensorNode, Tensor],]:
        """
        Extract prologues and epilogues from the fused graph.
        """
        # extract prologues
        prologues: Dict[Tensor, Prologue] = {}
        for task_input, tensor in zip(self.anchor_task.inputs, self.anchor_operator.inputs):
            if self.tensor_map[task_input] in self.graph.inputs:
                # this input does not have a prologue, skip
                continue
            axes = [var('i') for _ in range(len(task_input.shape))]
            te = tensor_element(task_input, axes)
            expr: Expr = self.visit(te)
            used_tensor_inputs = collect(expr, TensorInput)
            prologues[tensor] = Prologue(inputs=used_tensor_inputs, axes=axes, expr=expr)

        # extract epilogues
        epilogues: Dict[Tensor, Epilogue] = {}
        for task_output, tensor in zip(self.anchor_task.outputs, self.anchor_operator.outputs):
            if self.tensor_map[task_output] in self.graph.outputs:
                # this output does not have a epilogue, skip
                continue
            axes = [var('i') for _ in range(len(task_output.shape))]
            value = var('value', task_output.type.dtype)
            bss = BufferStoreStmt(buf=task_output, indices=axes, value=value)
            updated_bss = self.visit(bss)
            assert isinstance(updated_bss, BufferStoreStmt)
            used_tensor_inputs = collect(updated_bss.value, TensorInput)
            epilogues[tensor] = Epilogue(
                inputs=used_tensor_inputs,
                axes=axes,
                value=value,
                out_tensor=self.tensor_map[updated_bss.buf],
                out_indices=updated_bss.indices,
                out_expr=updated_bss.value,
            )

        return prologues, epilogues, self.tensor_map

    def visit_TensorElement(self, e: TensorElement):
        indices: Tuple[Expr, ...] = self.visit(e.indices)
        if isinstance(e.base, TensorInput):
            if e.base not in self.tensor_map:
                raise ValueError(f'Cannot find tensor for input {e.base}')
            tensor: Tensor = self.tensor_map[e.base]
            if tensor in self.graph.inputs:
                # the graph input, directly use the tensor
                return tensor_element(e.base, indices)
            else:
                # intermediate tensor
                op, output_idx = tensor.trace
                return self.visit(tensor_element(op.task.outputs[output_idx], indices))
        elif isinstance(e.base, GridCompute):
            remap = {a: b for a, b in zip(e.base.axes, indices)}
            return self.visit(rewrite(e.base.value, remap))
        else:
            raise NotImplementedError(f'Unsupported tensor {e}')

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        indices: Tuple[Expr, ...] = self.visit(stmt.indices)
        stmt_value: Expr = self.visit(stmt.value)
        if isinstance(stmt.buf, GridCompute):
            tensor = self.tensor_map[stmt.buf]
            if tensor in self.graph.outputs:
                # the graph output
                return BufferStoreStmt(buf=stmt.buf, indices=indices, value=stmt_value)
            else:
                # get the consumer TensorInput, Task, and its GridCompute output
                assert len(self.consume_by[stmt.buf]) == 1, 'Expect only one consumer for {}.'.format(stmt.buf)
                consumer_input: TensorInput = self.consume_by[stmt.buf][0]

                consumer_task: Task = self.input2task[consumer_input]
                assert len(consumer_task.outputs) == 1, 'Expect consumer task to have exactly one output.'

                consumer_output: TensorNode = consumer_task.outputs[0]
                assert isinstance(consumer_output, GridCompute), 'Only GridCompute is supported in epilogue.'
                gc: GridCompute = consumer_output

                # Example of what we are doing here:
                # original indices:
                # epilogue_out[i, j] = expr(i, j, out[i + 3, i + j])
                # inverse_map: (p, q) -> (p - 3, q - p - 3)
                #
                # original statement:
                # out[e1, e2] = value (e1, e2, and value are all Expr)
                #
                # expected statement:
                # epilogue_out[e1 - 3, e2 - e1 - 3] = expr(e1 - 3, e2 - e1 - 3, value)
                #
                # steps to get the expected statement:
                # 1. get the output index expressions using inverse_map
                #    e.g., e1 - 3, e2 - e1 - 3
                # 2. get the value expression to be stored
                #    e.g. expr(e1 - 3, e2 - e1 - 3, value)
                # 3. create the expected statement. If it still has epilogue, repeat above steps repeatedly.

                # step 1
                inverse_map: InverseMap = consumer_task.inverse_map[consumer_input]
                remap: Dict[Var, Expr] = {a: b for a, b in strict_zip(inverse_map.axes, indices)}
                out_indices: List[Expr] = [rewrite(e, remap) for e in inverse_map.indices]

                # step 2
                # replace index
                remap: Dict[Var, Expr] = {a: b for a, b in strict_zip(gc.axes, out_indices)}
                gc_value: Expr = rewrite(gc.value, remap)
                # replace out[i + 3, i + j] with value (in the example above)
                tensor_elements: List[TensorElement] = collect(gc_value, TensorElement, stop_when_found=False)
                tensor_elements = [te for te in tensor_elements if te.base is consumer_input]
                assert (
                    len(tensor_elements) == 1
                ), 'Epilogue can only index one time of the input tensor with inverse map'
                te: TensorElement = tensor_elements[0]
                # in the context of above example, we replace 'out[i + 3, i + j]' by 'value'
                self.memo[te] = stmt_value

                # step 3
                return self.visit(BufferStoreStmt(consumer_output, out_indices, gc_value))
        else:
            raise NotImplementedError(f'Unsupported buffer {stmt.buf}')


class PrologueEpilogueMarker(IRVisitor):
    def __init__(self, fused_task: FusedTask, prologues: Dict[Tensor, Prologue], epilogues: Dict[Tensor, Epilogue]):
        super().__init__()
        self.anchor: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.prologues: Dict[Tensor, Prologue] = prologues
        self.epilogues: Dict[Tensor, Epilogue] = epilogues
        self.param_match: Dict[str, Dict[Var, Tensor]] = defaultdict(dict)
        self.ir_module: Optional[IRModule] = None
        self.caller_name: Optional[str] = None

    def mark(self, module: IRModule) -> Dict[str, Dict[Var, Tensor]]:
        """
        Get the mapping from the parameters of each function to the corresponding tensors in FlowGraph, if that
        parameter represents a tensor.

        Examples
        --------

        .. code-block:: python

            def launch(a, b, c):
                declare d
                f(a, b, d)
                g(d, c)

        would return
        {'launch': {a: tensor_a, b: tensor_b, c: tensor_c},
          'f': {a: tensor_a, b: tensor_b},
          'g': {c: tensor_c}}

        """
        self.visit_IRModule(module)
        return self.param_match

    def visit_IRModule(self, module: IRModule):
        self.ir_module = module

        if 'launch' not in module.functions:
            raise ValueError('Cannot find launch function in the module.')
        call_graph = CallGraph(module, allow_missing=True)

        # initialize param_match for launch function
        launch_func: Function = module.functions['launch']
        for param_var, tensor in zip(launch_func.params, self.anchor.inputs + self.anchor.outputs):
            self.param_match[launch_func.name][param_var] = tensor

        # fill in param_match for callee functions
        for node in call_graph.order:
            func: Function = node.func
            self.caller_name = func.name
            self.visit(func)

    def process_call(self, callee_name, args):
        if callee_name not in self.ir_module.functions:
            return
        callee: Function = self.ir_module.functions[callee_name]
        for param_var, arg in zip(callee.params, args):
            if arg in self.param_match[self.caller_name]:
                tensor = self.param_match[self.caller_name][arg]
                if param_var not in self.param_match[callee_name]:
                    self.param_match[callee_name][param_var] = tensor
                else:
                    if self.param_match[callee_name][param_var] is not tensor:
                        msg = f'Inconsistent tensor node {tensor} for param {param_var} in function {callee_name}'
                        raise ValueError(msg)

    def visit_Call(self, e: Call):
        self.process_call(e.func_var.name, e.args)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        self.process_call(stmt.func_var.name, stmt.args)


class FuncParamRecord:
    def __init__(self, param_to_tensor, updated_params, tensor_to_updated_param):
        # the original param to the corresponding input/output tensor of the anchor operator
        self.param_to_tensor: Dict[Var, Tensor] = param_to_tensor

        # the updated parameters and mapping from the input/output tensor of the fused operator to updated params
        self.updated_params: List[Var] = updated_params
        self.tensor_to_updated_param: Dict[Tensor, Var] = tensor_to_updated_param


class CanNotFuseError(Exception):
    pass


def is_elementwise(op: Operator):
    return isinstance(op, ELEMENTWISE)


def is_tensor_manipulation(op: Operator):
    return isinstance(op.task, TENSOR_MANIPULATION)


Int = Union[Expr, int]


class CollectiveStoreRecord:
    """
    Represents a record for CollectiveStore operations in the CuTe dialect, used in the epilogue fusion.

    This class encapsulates the behavior of a thread block's CollectiveStore operation,
    which writes results from register files to global memory. The operation is tile-level,
    mapping tasks via `TiledCopy`.

    Attributes:
        tiled_copy (TiledCopy): Specifies the task mapping.
        src (Var): The tensor in registers held by the thread block.
        dst (Var): The output global memory tensor.
        offsets (List[Expr]): Initial offsets in each dimension of the global tensor.
        extents (List[Expr]): Extents in each dimension of the tile.
        tensor_layout (TensorLayout): Layout of the output tensor.
        buf_stub (Var): Placeholder for the global memory tensor.
        base (Var): Placeholder for the linearized offset.

    Example:
        A tensor distributed across a thread block of 32 threads:
        0  0  1  1  2  2  3  3
        4  4  5  5  6  6  7  7
        8  8  9  9 10 10 11 11
        12 12 13 13 14 14 15 15
        ...
        28 28 29 29 30 30 31 31
        0  0  1  1  2  2  3  3
        4  4  5  5  6  6  7  7
        8  8  9  9 10 10 11 11
        12 12 13 13 14 14 15 15
        ...
        28 28 29 29 30 30 31 31

        Tile shape is 16x8, represented with a TV layout:
        tv := ((4, 8), (2, 2)):((32, 1), (16, 8))

        This indicates:
        - (4, 8) means 32 threads are organized as an 8x4 tile.
        - (2, 2) means each thread holds a 2x2 data segment.

        The strides define the mapping from (thread_id, value_id) to coordinates in the logical tile domain (16x8).

        The CollectiveStore operation directly write the register tensor back to the global memory, so the
        TiledCopy can be represented by two TV layouts and the two TV layouts are identical, i.e.

        src_tv := ((4, 8), (2, 2)):((32, 1), (16, 8))
        dst_tv := ((4, 8), (2, 2)):((32, 1), (16, 8))

        With the global tensor:
        g_tensor := f16[16, 32, 16]

        The offsets and extents can be represented as:
        - Offsets: [bid_z, bid_x * 16, bid_y * 8]
        - Extents: [32 - bid_x * 16, 16 - bid_y * 8]
    """

    def __init__(
        self,
        tiled_copy: TiledCopy,
        src: Var,
        dst: Var,
        offsets: List[Expr],
        extents: List[Expr],
        tensor_layout: TensorLayout,
    ):
        self.tiled_copy: TiledCopy = tiled_copy
        self.src: Var = src
        self.dst: Var = dst
        self.offsets: List[Var] = offsets
        self.extents: List[Var] = extents
        self.tensor_layout: TensorLayout = tensor_layout
        self.buf_stub: Var = None
        self.base: Var = None


class CollectiveStoreExtractor(IRVisitor):
    """
    Extracts and records CollectiveStore operations from a schedule template for epilogue fusion.

    This pass identifies CollectiveStore operations within the schedule template, creating
    records for future translation and enabling vectorization optimization for epilogue fusion.

    Attributes:
        marks (Dict[str, Dict[Var, Tensor]]): Marks used for extraction.
        tensor2collective_store (Dict[Tensor, CollectiveStoreRecord]): Maps tensors to their
        respective CollectiveStore records.
        current_var2tensor (Dict[Var, Tensor]): Current variable to tensor mapping.

    Example:
        In a matmul template:
        ```python
        for k in range(0, K, BK):
            copy_g2r(g_A, regs_A)
            copy_g2r(g_B, regs_B)
            mma(regs_C, regs_A, regs_B)
        tensor_c = tensor_view(regs_C, tv_layout, "register")
        collective_store(tiled_copy, tensor_c, g_C, offsets, extents)
        ```
    """

    def __init__(self, marks: Dict[str, Dict[Var, Tensor]]):
        super().__init__()
        self.marks: Dict[str, Dict[Var, Tensor]] = marks
        self.tensor2collective_store: Dict[Tensor, CollectiveStoreRecord] = {}
        self.current_var2tensor: Dict[Var, Tensor] = None

    def extract(self, module: IRModule) -> Tuple[Dict[Var, Tensor], Tuple[Int]]:
        self.visit_IRModule(module)
        tshp = None
        for _, collective_store in self.tensor2collective_store.items():
            tiled_copy = collective_store.tiled_copy
            tile_shape, src_tv_layout = tiled_copy.src_tv_layout()
            if complement(src_tv_layout) == TensorLayout(1) and tshp is None:
                tshp = tile_shape
                break
        return self.tensor2collective_store, (tshp[1], tshp[0]) if tshp else None

    def visit_Function(self, func: Function):
        if func.name in self.marks:
            self.current_var2tensor = self.marks[func.name]
        else:
            self.current_var2tensor = None
        super().visit_Function(func)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallOp):
            call = stmt.expr
            op = call.op
            self.visit(op)
            if isinstance(op, CollectiveStore):
                tiled_copy = op.tiled_copy
                src = op.src
                dst = op.dst
                offsets = op.offsets
                extents = op.extents
                src_ty = infer_type(src)
                dst_ty = infer_type(dst)
                offsets = [var("v") for v in offsets]
                extents = [var("x") for x in extents] if extents else None
                assert self.current_var2tensor is not None and dst in self.current_var2tensor
                tensor = self.current_var2tensor[dst]
                tensor_shape = tensor.shape
                tensor_layout = TensorLayout(tuple(reversed(tensor_shape)))
                self.tensor2collective_store[tensor] = CollectiveStoreRecord(
                    tiled_copy, var("t_regs", src_ty), var("t_gmem", dst_ty), offsets, extents, tensor_layout
                )
        else:
            super().visit_EvaluateStmt(stmt)


class TileDerivation(GraphVisitor):
    """
    Infers and represents tiles as Layouts for all tensors in a fused graph.

    This pass traverses the fused graph to create a DAG of operators, inferring tiles
    through forward and backward passes.

    Attributes:
        anchor (Operator): The anchor operator.
        anchor_output_tile (Tuple[int, ...]): The output tile of the anchor.
        tensor2tile (Dict[Tensor, Tuple[TensorLayout, TensorLayout, Tuple[Int, ...]]]): Maps tensors
        to their tile information.
        dag (DirectedGraph): Directed acyclic graph of operators.

    Example:
        Consider the following computing graph:
        ```python
        c = matmul(a, b)  # 512x1024
        c1 = c + bias  # 512x1024
        c2 = c1.reshape(32, 32, 512)  # 32x32x512
        c3 = c2.permute(1, 2, 0)  # 32x512x32
        c4 = c3.reshape(256, 128, 16)  # 256x128x16
        ```

        Assuming a tile size of 64x128, the tile for output tensor `c` can be represented as:
        tile_c = (128, 64):(1, 1024)

        The memory layout of tensor `c` is:
        c_layout = (1024, 512):(1, 1024)

        To derive efficient task mappings, tiles for all leaf nodes must be inferred.

        Assume the tile size of the schedule template is 64x128. The original tile of the output tensor C
        can be represented as a layout: (128, 64):(1, 1024).

        The memory layout of tensor C is: (1024, 512):(1, 1024).

        Note that the memory layout function takes the coordinates of C and returns the address.

        To derive the efficient task mapping for each load and store operation in the epilogue graph,
        we need to infer the tile for all leaf nodes.

        To infer the tile, we must first rewrite the layout of each tensor as a function that takes
        the coordinates of tensor C and returns the target tensor's address. To achieve this,
        we register the tile mapping for each operator in their task:

        tile_mapping_c1 = 1:1 # identity
        tile_mapping_c2 = 1:1 # identity
        tile_mapping_c3 = (512, 32, 32):(32, 32 * 512, 1)
        tile_mapping_c4 = 1:1 # identity

        In the forward pass, we can infer the layouts and tiles for tensors c1, c2, c3, and c4.
        For example, to rewrite the layout for the output tensor c4, we compose five layouts:

        c4_layout = tile_mapping_c4 * tile_mapping_c3 * tile_mapping_c2 * tile_mapping_c1
                    * c_layout
                  = (1024, (16, 32)):(32, (32768, 1))

        Then, by performing tile division between c4_layout and the tile size (128, 64), we get the tile for c4:

        tile_c4 = (128, (16, 4)):(32, (32768, 1))

        With tile_c4, we can derive the efficient task mapping for the store task of tensor c4.

        In the backward pass, we can infer the remaining leaf node, bias, because the layout of c1 is known.
        Although elementwise operations won't change the layout function, the broadcasting relation will affect it.
        Therefore, we set the stride of the broadcasted dimension to 0:

        bias_layout = (1024, 512):(1, 0)
        tile_bias = (128, 64):(1, 0)

        With this method, we ensure that the layout functions that represents the tiles are correctly derived
        for later IR translation and efficient task mapping generation.
    """

    def __init__(self, anchor: Operator, anchor_output_tile: Tuple[int, ...]):
        super().__init__()
        self.anchor: Operator = anchor
        self.anchor_output_tile: Tuple[int, ...] = anchor_output_tile
        self.tensor2tile: Dict[Tensor, Tuple[TensorLayout, TensorLayout, Tuple[Int, ...]]] = {}
        self.dag = DirectedGraph()

    def _assert(self, expr: Union[Expr, bool], msg: Optional[str] = None):
        simplified = hidet.ir.tools.simplify(expr)
        if is_constant(simplified):
            assert simplified, msg
        # assertion at runtime

    def is_anchor(self, op: Operator):
        return op is self.anchor

    def visit_Operator(self, op: Operator):
        for inp in op.inputs:
            self.visit(inp)
            inp_op = inp.op
            if inp_op is None:
                continue
            self.dag.add_edge(inp_op, op)
        self.dag.add_node(op)

    def _tile_divide(self, layout: TensorLayout):
        result_shape = []
        result_stride = []
        for i, extent in enumerate(self.anchor_output_tile):
            shape = layout[i].shape_tuple
            stride = layout[i].stride_tuple
            if any(not is_constant(s) for s in shape[:-1]):
                return None
            cur_idx = 0
            cur_shape = []
            cur_stride = []
            while extent > 1:
                s = shape[cur_idx]
                d = stride[cur_idx]
                if cur_idx == len(shape) - 1 or s > extent:
                    cur_shape.append(extent)
                    cur_stride.append(d)
                    extent //= extent
                else:
                    if extent % s != 0:
                        return None
                    extent //= s
                    cur_shape.append(s)
                    cur_stride.append(d)
                cur_idx += 1
            result_shape.append(tuple(cur_shape) if len(cur_shape) > 1 else cur_shape[0])
            result_stride.append(tuple(cur_stride) if len(cur_stride) > 1 else cur_stride[0])
        return TensorLayout(tuple(result_shape), tuple(result_stride))

    def _forward(self, ops: Sequence[Operator]):
        for op in ops:
            if self.is_anchor(op):
                for output in op.outputs:
                    tile_rank = len(self.anchor_output_tile)
                    tensor_shape = output.shape
                    rev_tshp = tuple(reversed(tensor_shape))
                    tile = TensorLayout(self.anchor_output_tile, compact_col_major(rev_tshp)[:tile_rank])
                    self.tensor2tile[output] = (tile, TensorLayout(rev_tshp), tensor_shape)
            elif is_elementwise(op):
                if any(ti in self.tensor2tile for ti in op.inputs):
                    broadcast_shape = broadcast_shapes([ti.shape for ti in op.inputs])
                    for ti in op.inputs:
                        if ti in self.tensor2tile:
                            (input_tile, input_layout, _) = self.tensor2tile[ti]
                    self.tensor2tile[op.outputs[0]] = (input_tile, input_layout, tuple(broadcast_shape))
            else:
                inp = op.inputs[0]
                task_input = op.task.inputs[0]
                if inp in self.tensor2tile:
                    (input_tile, input_layout, broadcast_shape) = self.tensor2tile[inp]
                    tile_mapping = op.task.inverse_map[task_input].tile_mapping
                    self._assert(prod(inp.shape) == prod(broadcast_shape))
                    layout = composition(tile_mapping, input_layout)
                    self.tensor2tile[op.outputs[0]] = (self._tile_divide(layout), layout, op.outputs[0].shape)

    def _backward(self, ops: Sequence[Operator]):
        for op in reversed(ops):
            if self.is_anchor(op):
                pass
            elif is_elementwise(op):
                if op.outputs[0] in self.tensor2tile:
                    _, output_layout, _ = self.tensor2tile[op.outputs[0]]
                    output_shape = broadcast_shapes([ti.shape for ti in op.inputs])
                    for ti in op.inputs:
                        if ti not in self.tensor2tile:
                            shape = list(ti.shape)
                            while len(shape) < len(output_shape):
                                shape = [i32(1)] + shape
                            result_shape = tuple(int(s) if is_constant(s) else s for s in output_shape)
                            result_stride = list(compact_row_major(tuple(shape)))
                            for i, e in enumerate(shape):
                                if is_constant(e) and e == 1:
                                    result_stride[i] = 0
                            tile_mapping = coalesce(
                                TensorLayout(tuple(reversed(result_shape)), tuple(reversed(result_stride)))
                            )
                            layout = composition(tile_mapping, output_layout)
                            layout = canonicalize(layout)
                            # assert all(is_constant(s) for s in output_shape)
                            self.tensor2tile[ti] = (self._tile_divide(layout), layout, tuple(output_shape))
            else:
                if op.outputs[0] in self.tensor2tile:
                    (_, output_layout, broadcast_shape) = self.tensor2tile[op.outputs[0]]
                    output_shape = op.outputs[0].shape
                    self._assert(prod(broadcast_shape) == prod(output_shape))
                    inp = op.inputs[0]
                    if inp not in self.tensor2tile:
                        task_input = op.task.inputs[0]
                        tile_mapping = op.task.inverse_map[task_input].tile_mapping
                        layout = composition(tile_mapping, output_layout)
                        self.tensor2tile[inp] = (self._tile_divide(layout), layout, inp.shape)

    def visit_FlowGraph(self, graph: FlowGraph):
        for output in graph.outputs:
            self.visit(output)

        ops = self.dag.topological_order()

        self._forward(ops)
        self._backward(ops)


class EpilogueVisitorRewriter(GraphVisitor):
    """
    Translates the graph-level IR to kernel-level IR.

    Note: Currently, only one output is supported in the fused graph.

    This pass traverses the fused graph and translates the operators into CuTe operations.

    Example:
        Consider the following computing graph:

        a = tensor f16[512, 128]
        b = tensor f16[128, 1024]
        c = tensor f16[512, 1024]
        bias = tensor f16[1024]
        c = matmul(a, b)  # 512x1024
        c1 = c + bias  # 512x1024
        c2 = c1.reshape(32, 32, 512)  # 32x32x512
        c3 = c2.permute(1, 2, 0)  # 32x512x32
        c4 = c3.reshape(256, 128, 16)  # 256x128x16

        In the previous pass, we inferred the tile of tensor c4 as:
        tile_c4 = (128, (16, 4)):(32, (32768, 1))

        With tile_c4, we can derive the task mapping that improves locality and
        coalesces accesses for the final write-back task. In this example, we assume
        we have 128 threads in total:

        Step 1:
        - Sort the pair of shapes and strides according to the strides' ascending order.
        - Sorted shapes (S): 4, 128, 16
        - Sorted strides (D): 1, 32, 32768

        Observations:
        1. The smallest stride is 1, and the corresponding shape is 4.
        2. All remaining strides are multiples of 4.

        So the maximum vector size is 4, and the TV layout for tensor c4 becomes:
        ((t1, t2, ...), (4, v1, v2, ...)):((d1, d2, ...), (16, s1, s2, ...))

        The next dimension 128 should be assigned to the 128 threads, improving global memory access locality:
        ((128, ), (4, v1, v2, ...)):((1, ), (16, s1, s2, ...))

        The remaining dimension is assigned to each thread, resulting in the final layout:
        ((128, ), (4, 16)):((64, ), (16, 1))

        This layout maps thread ID (tid) and value ID (vid) to the coordinates in the tile 64x128.
        Composing this layout and tile_c4 gives the function from tid and vid to the actual memory address.

        Step 2:
        - Traverse the epilogue and generate kernel-level operators for each graph-level operator.
        Assume all computation nodes use the same layout as the TV layout of the output layout,
        and computation happens on registers. Tensor manipulation operators (e.g., reshape, permute)
        can be treated as NoOps.

        Visit c + bias and generate:
        - c' = rearrange(c, c4_tv_layout)
        - bias = load(bias_ptr)
        - c1 = c' + bias

        Skip c2, c3, and c4 as they only forward the tensor without changes:
        - c2 = c1
        - c3 = c2
        - c4 = c3

        Finally, generate a store operator for c4:
        - store(c4, c4_ptr)

    Args:
        fused_task (FusedTask): The fused task.
        tensor2tile (Dict[Tensor, Tuple[TensorLayout, TensorLayout, Tuple[Int]]]): Mapping from tensors
        to their tile information.
        tensor2collective_store (Dict[Tensor, CollectiveStoreRecord]): Mapping from tensors to
        their collective store records.
    """

    def __init__(
        self,
        fused_task: FusedTask,
        tensor2tile: Dict[Tensor, Tuple[TensorLayout, TensorLayout, Tuple[Int]]],
        tensor2collective_store: Dict[Tensor, CollectiveStoreRecord],
    ):
        super().__init__()
        self.anchor: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.anchor_outputs: List[Tensor] = self.anchor.outputs
        self.inputs = fused_task.fused_graph.inputs
        self.outputs = fused_task.fused_graph.outputs
        self.tensor2tile = tensor2tile
        self.stmts: List[Stmt] = []
        self.tensor2collective_store: Dict[Tensor, CollectiveStoreRecord] = tensor2collective_store
        self.tensor2var: Dict[Tensor, Var] = {}
        self.current_output: Tensor = None
        self.current_collective_store = None
        self.current_output_thrval: TensorLayout = None
        self.tile_shape = None

        self.tensor2stmts: Dict[Tensor, List[Stmt]] = {}
        self.inputs2stub: Dict[Tensor, Var] = {}

        for tensor, collective_store in self.tensor2collective_store.items():
            _, _, broadcast_shape = self.tensor2tile[tensor]
            tensor_shape = tensor.shape
            if broadcast_shape == tensor_shape or prod(broadcast_shape) == prod(tensor_shape):
                tiled_copy = collective_store.tiled_copy
                shape, src_tv_layout = tiled_copy.src_tv_layout()
                self.tile_shape = shape
                self.threads = src_tv_layout[0][0].size()

    def declare(self, v: Var = None, hint: str = None, e: Expr = None):
        '''Declare a variable or expression.'''
        if e is None:
            return self.stmts.append(DeclareStmt(v))

        v_ty = infer_type(e)
        v = var(hint, v_ty)
        self.stmts.append(DeclareStmt(v, e))
        return v

    def append(self, e: Expr):
        '''Append an expression to the statement list.'''
        self.stmts.append(EvaluateStmt(e))

    def flush_stmts(self):
        '''Flush and return the current list of statements.'''
        stmts = self.stmts
        self.stmts = []
        return stmts

    def is_anchor(self, op: Operator):
        '''Check if an operator is the anchor.'''
        return op is self.anchor

    def _schedule_tiled_copy(self, dtype: DataType, tile_layout: TensorLayout):
        """
        Schedule tiled copy operations based on data type and tile layout.
        """
        from hidet.utils import gcd

        tile_layout = coalesce(tile_layout)
        shape = cute.flatten(tile_layout.shape_tuple)
        stride = cute.flatten(tile_layout.stride_tuple)
        costride = list(compact_col_major(shape))
        shape = list(shape)
        stride = list(stride)
        index = range(len(shape))

        need_broadcast = False
        sorted_dsi = sorted(zip(stride, shape, index))
        vector_size = 0
        vector_dim_list = []
        # maximum elements that each thread can process
        non_vector_strides = [prod(shape) // self.threads]
        for d, s, i in sorted_dsi:
            if d == 0:  # stride that equals 0 indicates we need broadcasting this tensor
                costride[i] = 0
                need_broadcast = True
            elif d == 1 and vector_size == 0:
                vector_size = s
                vector_dim_list.append(i)
            elif d == vector_size:
                vector_size *= s
                vector_dim_list.append(i)
            else:
                non_vector_strides.append(d)
        # Here, we handle the tensor that needs to be broadcasted seperately because we
        # want to eliminate the register rearrange through shared memory.
        if need_broadcast:
            codomain = coalesce(TensorLayout(tuple(shape), tuple(costride)))
            thrval_layout = composition(codomain, self.current_output_thrval)
            thr_layout = coalesce(thrval_layout[0])
            val_layout = coalesce(thrval_layout[1])
            copy_atom = CopyAtom("thread_block", self.tile_shape, make_layout(thr_layout, val_layout))
            return TiledCopy(copy_atom)

        BITS_PER_MEMORY_INST = 128
        alignment = [i * dtype.nbits for i in non_vector_strides]
        bits_per_inst = gcd(dtype.nbits * vector_size, BITS_PER_MEMORY_INST, *alignment)
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

        for i in sorted(val_dims, key=lambda x: -x):
            shape.pop(i)
            stride.pop(i)
            costride.pop(i)

        thr_shape = []
        thr_stride = []
        remaining_threads = self.threads
        sorted_dsi = sorted(zip(stride, shape, costride))
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
        if remaining_threads > 1:
            thr_shape.append(remaining_threads)
            thr_stride.append(0)
        thr_layout = TensorLayout(tuple(thr_shape), tuple(thr_stride))
        val_layout = TensorLayout(tuple(val_shape), tuple(val_stride))
        copy_atom = CopyAtom("thread_block", self.tile_shape, make_layout(thr_layout, val_layout))
        return TiledCopy(copy_atom)

    def emit_LoadImpl(self, tensor: Tensor):
        '''Implement the loading of tensors from global memory.'''
        # Step 1. Use the tile information to derive the efficient task
        # mapping, then load the tensor from global memory.
        tile, addr_functor, _ = self.tensor2tile[tensor]
        tile_layout = make_layout(tile[1], tile[0])
        tiled_copy = self._schedule_tiled_copy(tensor.dtype, tile_layout)
        tv_atom = ThrValAtom.from_copy_atom_src(tiled_copy.copy_atom)
        tiled_tensor_layout = TiledTensorLayout(tv_atom, tiled_copy.levels)
        nr_regs = tiled_tensor_layout.val_count()
        regs = tensor_var("regs", shape=[nr_regs], dtype=tensor.dtype)
        self.declare(v=regs)
        if tensor in self.inputs2stub:
            tensor_stub = self.inputs2stub[tensor]
        else:
            tensor_stub = var("stub", PointerType(tensor.dtype))
            self.inputs2stub[tensor] = tensor_stub
        # tensor_layout = self.current_collective_store.tensor_layout
        # tile_rank = rank(tile.shape)
        # shape = tuple(mode.size() for mode in tile)
        # stride = tensor_layout.stride_tuple[:tile_rank]
        # tensor_layout = TensorLayout(shape, stride).reversed()
        if self.current_collective_store.base is None:
            self.current_collective_store.base = var("base")
        # composed_layout = ComposedTensorLayout(tensor_layout, self.current_collective_store.base, addr_functor)
        t_regs = self.declare(hint="t_regs", e=tensor_view(regs, tiled_tensor_layout, "register"))
        base = addr_functor(self.current_collective_store.base)
        t_gmem = self.declare(hint="t_gmem", e=tensor_view(tensor_stub + base, tile_layout, "global"))
        txgx = self.declare(hint="txgx", e=partition_src(t_gmem, tiled_copy))
        txrx = self.declare(hint="txrx", e=partition_dst(t_regs, tiled_copy))
        masks = self.declare(hint="masks", e=mask(tiled_copy, self.current_collective_store.extents))
        self.append(copy(tiled_copy, txgx, txrx, masks))
        t_load = self.declare(hint="t_load", e=tensor_view(txrx, tiled_tensor_layout, "register"))
        # all the layout of the input tensor should be aligned with the output
        # tensor. If the layout is not aligned with the output, we insert a
        # rearrange operator to re-distribute the data across the thread block.
        output_aligned_layout = self._output_aligned_layout(tiled_tensor_layout)
        if tiled_tensor_layout != output_aligned_layout:
            t_cvt = self.declare(hint="t_cvt", e=rearrange(t_load, output_aligned_layout, "register"))
            self.tensor2var[tensor] = t_cvt
        else:
            self.tensor2var[tensor] = t_load

    def emit_StoreImpl(self, tensor: Tensor):
        '''Implement the storing of tensors to global memory.'''
        tile, addr_functor, _ = self.tensor2tile[tensor]
        tile_layout = make_layout(tile[1], tile[0])
        tiled_copy = self._schedule_tiled_copy(tensor.dtype, tile_layout)
        assert tensor in self.tensor2var
        t_regs = self.tensor2var[tensor]
        self.current_collective_store.buf_stub = var("t_gmem", PointerType(tensor.dtype))
        if self.current_collective_store.base is None:
            self.current_collective_store.base = var("base")
        # tensor_layout = self.current_collective_store.tensor_layout
        # tile_rank = rank(tile.shape)
        # shape = tuple(mode.size() for mode in tile)
        # stride = tensor_layout.stride_tuple[:tile_rank]
        # tensor_layout = TensorLayout(shape, stride).reversed()
        base = addr_functor(self.current_collective_store.base)
        # composed_layout = ComposedTensorLayout(tensor_layout, self.current_collective_store.base, addr_functor)
        t_gmem = self.declare(
            hint="t_gmem", e=tensor_view(self.current_collective_store.buf_stub + base, tile_layout, "global")
        )
        txrx = self.declare(hint="txrx", e=partition_src(t_regs, tiled_copy))
        txgx = self.declare(hint="txgx", e=partition_dst(t_gmem, tiled_copy))
        masks = self.declare(hint="masks", e=mask(tiled_copy, self.current_collective_store.extents))
        self.append(copy(tiled_copy, txrx, txgx, masks))

    # we are trying to make pickle happy because pickle cannot serialize local
    # functions.
    # TODO: maybe we can use dill instead of pickle in the future so that we
    # don't need this trick to bypass the serialization issue.
    @staticmethod
    def cast(dtype: DataType, x: Expr):
        from hidet.ir.expr import cast as ir_cast

        return ir_cast(x, dtype)

    @staticmethod
    def where(cond: Expr, x: Expr, y: Expr):
        return if_then_else(cond, x, y)

    @staticmethod
    def where_tensor_scalar(y: Expr, cond: Expr, x: Expr):
        return if_then_else(cond, x, y)

    @staticmethod
    def where_scalar_tensor(y: Expr, cond: Expr, x: Expr):
        return if_then_else(cond, y, x)

    @staticmethod
    def where_scalar_scalar(x: Expr, y: Expr, cond: Expr):
        return if_then_else(cond, x, y)

    def _get_elementwise_op(self, op: Operator):
        '''Retrieve the elementwise operation.'''
        if isinstance(op, CastOp):
            dtype = op.attrs['dtype']
            return partial(EpilogueVisitorRewriter.cast, dtype)
        elif isinstance(op, (UnaryElementwiseOp, BinaryElementwiseOp)):
            return op.op
        elif isinstance(op, CompositeElementwiseOp):
            left_unary_op = op.attrs["left_unary_op"]
            right_unary_op = op.attrs["right_unary_op"]
            binary_op = op.attrs["binary_op"]
            return partial(EpilogueVisitorRewriter.composite_elementwise, binary_op, left_unary_op, right_unary_op)
        elif isinstance(op, WhereOp):
            return EpilogueVisitorRewriter.where
        elif isinstance(op, WhereTensorScalarOp):
            y = op.attrs["y"]
            return partial(EpilogueVisitorRewriter.where_tensor_scalar, y)
        elif isinstance(op, WhereScalarTensorOp):
            x = op.attrs["x"]
            return partial(EpilogueVisitorRewriter.where_scalar_tensor, x)
        elif isinstance(op, WhereScalarScalarOp):
            x = op.attrs["x"]
            y = op.attrs["y"]
            return partial(EpilogueVisitorRewriter.where_scalar_scalar, x, y)
        else:
            raise NotImplementedError(f"Operator{op.__class__.__name__} is not supported yet")

    def emit_Elementwise(self, op: Operator):
        '''Emit elementwise operations.'''
        t_inps = []
        for inp in op.inputs:
            assert inp in self.tensor2var
            t_inps.append(self.tensor2var[inp])
        t_out = self.declare(hint=op.name, e=arithmetic(*t_inps, op=self._get_elementwise_op(op)))
        output = op.outputs[0]
        self.tensor2var[output] = t_out

    def emit_ForwardNoOp(self, op: Operator):
        '''Emit NoOp for tensor manipulation operations.'''
        inp = op.inputs[0]
        assert inp in self.tensor2var
        v = self.tensor2var[inp]
        output = op.outputs[0]
        self.tensor2var[output] = v

    def _output_aligned_layout(self, tiled_tensor_layout: TiledTensorLayout):
        """
        Align the tiled tensor layout with the output layout so that all the
        tile layouts for all the intermediate nodes are aligned, and elementwise
        computation can perform on these tensors.
        """
        tile_shape = tiled_tensor_layout.shape()
        tensor_thrval = tiled_tensor_layout.thrval_layout()
        shape = list(cute.flatten(tensor_thrval.shape_tuple))
        stride = list(cute.flatten(tensor_thrval.stride_tuple))
        sorted_ds = sorted(zip(stride, shape))
        result_shape = []
        result_stride = []
        current_idx = 1
        from hidet.ir.cute import shape_div, size

        for d, s in sorted_ds:
            if d == 0:
                continue
            if d != current_idx:
                result_shape.append(shape_div(d, current_idx))
                result_stride.append(0)
                current_idx = d

            result_shape.append(s)
            result_stride.append(d)
            current_idx *= s
        total_idx = size(tile_shape)
        if current_idx != total_idx:
            result_shape.append(shape_div(total_idx, current_idx))
            result_stride.append(0)
        codomain = coalesce(TensorLayout(tuple(result_shape), tuple(result_stride)))
        new_tensor_thrval = composition(codomain, self.current_output_thrval)
        new_tensor_thrval = make_layout(coalesce(new_tensor_thrval[0]), coalesce(new_tensor_thrval[1]))
        tv_atom = ThrValAtom("thread_block", tile_shape, new_tensor_thrval)
        return TiledTensorLayout(tv_atom)

    def visit_Tensor(self, tensor: Tensor):
        '''Visit a tensor and generate its load or store implementation.'''
        if tensor in self.inputs:
            self.emit_LoadImpl(tensor)
        elif tensor in self.outputs:
            tile, _, _ = self.tensor2tile[tensor]
            tile_layout = make_layout(tile[1], tile[0])
            tiled_copy = self._schedule_tiled_copy(tensor.dtype, tile_layout)
            _, tv = tiled_copy.src_tv_layout()
            t, v = tv[0][0], coalesce(make_layout(tv[0][1], tv[1]))
            self.current_output_thrval = make_layout(t, v)
            self.visit(tensor.op)
            self.emit_StoreImpl(tensor)
        elif self.is_anchor(tensor.op):
            collective_store = self.current_collective_store
            src = collective_store.src
            tshp, tv_layout = collective_store.tiled_copy.src_tv_layout()
            thr_layout, val_layout = tv_layout[0][0], coalesce(make_layout(tv_layout[0][1], tv_layout[1]))
            tv = make_layout(thr_layout, val_layout)
            tv_atom = ThrValAtom("thread_block", tshp, tv)
            tiled_tensor_layout = TiledTensorLayout(tv_atom)
            if not is_auto_layout(src.type.layout) and not isinstance(src.type.layout, TiledTensorLayout):
                src = self.declare(hint="view", e=tensor_view(src, tiled_tensor_layout, "register"))
            tiled_tensor_layout = self._output_aligned_layout(tiled_tensor_layout)
            cvt = self.declare(hint="cvt", e=rearrange(src, tiled_tensor_layout, "register"))
            self.tensor2var[tensor] = cvt
        else:
            self.visit(tensor.op)

    def visit_Operator(self, op: Operator):
        '''Visit an operator, handling elementwise or tensor manipulation operations.'''
        for inp in op.inputs:
            self.visit(inp)
        if self.is_anchor(op):
            pass
        elif is_elementwise(op):
            self.emit_Elementwise(op)
        elif is_tensor_manipulation(op):
            self.emit_ForwardNoOp(op)

    def visit_FlowGraph(self, graph: FlowGraph):
        """
        Translate graph operators for each output in the fused graph.
        Handle the anchor operator and collective store records.
        Flush statements after processing each output.
        """
        # Translate the graph operator for each output in the fused graph.
        # Note: we can only have one output in the fused graph currently.
        for output in graph.outputs:
            self.current_output = output
            # Note: anchor operator only has one output
            assert len(self.anchor_outputs) == 1
            anchor_output = self.anchor_outputs[0]
            assert anchor_output in self.tensor2collective_store
            self.current_collective_store = self.tensor2collective_store[anchor_output]
            self.visit(output)
            self.tensor2stmts[output] = self.flush_stmts()


class RearrangeEliminateRewriter(IRRewriter):
    def visit_CallOp(self, call: CallOp):
        op = call.op
        if isinstance(op, Rearrange):
            x = op.x
            if isinstance(x, CallOp) and isinstance(x.op, Rearrange):
                new_x = self.visit(x.op.x)
                return rearrange(new_x, layout=op.layout, scope=op.scope)
        return super().visit_CallOp(call)


class LowerCuteDialectToBufferStoreStmtRewriter(LowerCuteDialectRewriter):
    """
    Lowers the CollectiveStore operator to BufferStoreStmt if an error occurs when inferring the tile for each tensor.

    This pass handles cases where the tile cannot be represented as a Layout.
    In such scenarios, vectorizing memory access is not possible, so this fallback mechanism is used.

    Methods:
        __init__(): Initializes the rewriter.
        declare(v=None, hint=None, e=None): Declares a variable or expression.
        visit_EvaluateStmt(stmt): Visits and processes EvaluateStmt nodes.

    Example:
        If an error occurs during tile inference, the CollectiveStore operation will be lowered to BufferStoreStmt,
        ensuring proper memory handling without vectorization.
    """

    def __init__(self):
        '''Initializes the rewriter.'''
        super().__init__()

    def declare(self, v: Var = None, hint: str = None, e: Expr = None):
        """
        Declares a variable or expression.

        Args:
            v (Var, optional): Variable to declare. Defaults to None.
            hint (str, optional): Hint for variable name. Defaults to None.
            e (Expr, optional): Expression to assign to the variable. Defaults to None.

        Returns:
            Var: The declared variable.
        """
        if e is None:
            return self.append_stmt(DeclareStmt(v))

        v_ty = infer_type(e)
        v = var(hint, v_ty)
        self.append_stmt(DeclareStmt(v, e))
        return v

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        """
        Visits and processes EvaluateStmt nodes.

        Args:
            stmt (EvaluateStmt): The statement to visit.

        Returns:
            Stmt: The processed statement.
        """
        from hidet.lang.cuda import threadIdx
        from hidet.ir.expr import logical_and
        from hidet.ir.stmt import IfStmt
        from hidet.ir.cute import idx2crd

        if isinstance(stmt.expr, CallOp):
            call = stmt.expr
            op = call.op
            if isinstance(op, CollectiveStore):
                src_ty = infer_type(op.src)
                assert (
                    isinstance(src_ty, TiledTensorType)
                    and src_ty.scope.is_register()
                    and isinstance(src_ty.layout, TiledTensorLayout)
                )
                if isinstance(op.src, Var):
                    src_buf = self.var2buffer[op.src]
                else:
                    src_buf = self.visit(op.src)
                dst = self.visit(op.dst)
                offsets = [self.visit(o) for o in op.offsets]
                extents = [self.visit(e) for e in op.extents]
                src_buf = src_buf.buffer
                shape = src_ty.layout.shape()
                tile_rank = len(shape)
                thr_layout = src_ty.layout.thr_layout()
                val_layout = src_ty.layout.val_layout()
                crd = self.declare(hint="crd", e=thr_layout(threadIdx.x))
                for i in range(val_layout.size()):
                    v = self.declare(hint="v", e=crd + val_layout(i))
                    coords = idx2crd(v, shape)
                    cond = [e < v for v, e in zip(extents, coords)]
                    tile_coords = [o + c for o, c in zip(offsets[-tile_rank:], coords)]
                    offsets_ = offsets[:-tile_rank] + tile_coords
                    self.append_stmt(IfStmt(logical_and(*cond), dst.write(offsets_, src_buf[i], protected=False)))
                stmts = self.flush_stmts()
                new_stmt = SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
                return new_stmt
        return super().visit_EvaluateStmt(stmt)


class PrologueEpilogueRewriter(IRRewriter):
    def __init__(
        self,
        fused_task: FusedTask,
        prologues: Dict[Tensor, Prologue],
        epilogues: Dict[Tensor, Epilogue],
        tensor_map: Dict[TensorNode, Tensor],
        marks: Dict[str, Dict[Var, Tensor]],
        epilogue_visitor_tree: EpilogueVisitorRewriter = None,
    ):
        super().__init__()
        self.fused_graph: FlowGraph = fused_task.fused_graph
        self.anchor: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.prologues: Dict[Tensor, Prologue] = prologues
        self.epilogues: Dict[Tensor, Epilogue] = epilogues
        self.marks: Dict[str, Dict[Var, Tensor]] = marks
        self.tensor_map: Dict[TensorNode, Tensor] = tensor_map

        self.func_records: Dict[str, FuncParamRecord] = {}
        self.current_record: Optional[FuncParamRecord] = None
        self.ir_module: Optional[IRModule] = None

        self.epilogue_visitor_tree: EpilogueVisitorRewriter = epilogue_visitor_tree

    def visit_IRModule(self, module: IRModule):
        call_graph = CallGraph(module, allow_missing=True)
        self.ir_module = module

        for func_name, var2tensor in self.marks.items():
            param_to_tensor: Dict[Var, Tensor] = {}
            # the original param to the corresponding input/output tensor of the anchor operator
            for param_var, anchor_tensor in var2tensor.items():
                param_to_tensor[param_var] = anchor_tensor

            # add the parameters that does not correspond to any tensor, or has no prologue/epilogue
            updated_params = []
            tensor_to_updated_param: Dict[Tensor, Var] = {}
            for param in self.ir_module.functions[func_name].params:
                if param in param_to_tensor:
                    tensor = param_to_tensor[param]
                    if tensor not in self.prologues and tensor not in self.epilogues:
                        updated_params.append(param)
                        tensor_to_updated_param[tensor] = param
                    else:
                        # skip the parameters that have prologue/epilogue, since they will not exist
                        pass
                else:
                    # the parameter does not correspond to any tensor, keep it
                    updated_params.append(param)

            # add the required input/output tensors for prologue/epilogue
            for param_var, tensor in param_to_tensor.items():
                if tensor in self.prologues:
                    prologue: Prologue = self.prologues[tensor]
                    for ti in prologue.inputs:
                        t = self.tensor_map[ti]
                        if t not in tensor_to_updated_param:
                            updated_params.append(tensor_pointer_var(ti.name, t.shape, t.dtype))
                            tensor_to_updated_param[t] = updated_params[-1]
                elif tensor in self.epilogues:
                    epilogue: Epilogue = self.epilogues[tensor]
                    for ti in epilogue.inputs:
                        t = self.tensor_map[ti]
                        if t not in tensor_to_updated_param:
                            updated_params.append(tensor_pointer_var(ti.name, t.shape, t.dtype))
                            tensor_to_updated_param[t] = updated_params[-1]
                    out_tensor = epilogue.out_tensor
                    updated_params.append(tensor_pointer_var(param_var.hint, out_tensor.shape, out_tensor.dtype))
                    tensor_to_updated_param[out_tensor] = updated_params[-1]
                else:
                    # the parameter does not have prologue/epilogue, skip it
                    pass

            if func_name == 'launch':
                # reorder the updated params to match the order of the inputs and outputs of the graph
                assert len(updated_params) == len(self.fused_graph.inputs) + len(self.fused_graph.outputs)
                updated_params = [
                    tensor_to_updated_param[t] for t in self.fused_graph.inputs + self.fused_graph.outputs
                ]

            self.func_records[func_name] = FuncParamRecord(param_to_tensor, updated_params, tensor_to_updated_param)

        for node in call_graph.order:
            func: Function = node.func
            self.visit(func)

        return module.copy().reset_funcs(
            functions=self.visit(module.functions), global_vars=self.visit(module.global_vars)
        )

    def visit_Function(self, func: Function):
        if func.name not in self.func_records:
            return func

        record: FuncParamRecord = self.func_records[func.name]

        self.current_record = record

        # update the parameters of the function
        params = record.updated_params
        body = self.visit(func.body)
        return Function(
            name=func.name, params=params, body=body, ret_type=func.ret_type, kind=func.kind, attrs=func.attrs
        )

    def visit_Var(self, e: Var):
        if e in self.current_record.param_to_tensor:
            tensor = self.current_record.param_to_tensor[e]
            if tensor in self.prologues:
                # we encounter a usage of an input tensor of the task other than TensorElement and BufferStoreStmt
                raise CanNotFuseError(
                    'Did you used a tensor in expression other than pure tensor indexing (e.g., tensor[...])'
                    ' while marking the task as allowing prologue?'
                )
            elif tensor in self.epilogues:
                # we encounter a usage of an output tensor of the task other than TensorElement and BufferStoreStmt
                raise CanNotFuseError(
                    'Did you used a tensor in expression other than tensor storing (e.g., tensor[...] = ...)'
                    ' while marking the task as allowing epilogue?'
                )
        return super().visit_Var(e)

    def process_call(self, func_name: str, args: List[Expr]) -> List[Expr]:
        caller_record: FuncParamRecord = self.current_record
        callee_record: FuncParamRecord = self.func_records[func_name]
        new_args = []
        origin_param_to_arg: Dict[Var, Expr] = {a: b for a, b in zip(self.ir_module.functions[func_name].params, args)}
        updated_var_to_tensor: Dict[Var, Tensor] = {v: k for k, v in callee_record.tensor_to_updated_param.items()}
        for updated_param in callee_record.updated_params:
            if updated_param in updated_var_to_tensor:
                new_args.append(caller_record.tensor_to_updated_param[updated_var_to_tensor[updated_param]])
            else:
                assert updated_param in origin_param_to_arg
                new_args.append(self.visit(origin_param_to_arg[updated_param]))
        return new_args

    def visit_Call(self, e: Call):
        if isinstance(e.func_var, Var):
            func_name = e.func_var.name
            if func_name in self.func_records:
                args = self.process_call(func_name, list(e.args))
                return Call(e.func_var, args)
        return super().visit_Call(e)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        if isinstance(stmt.func_var, Var):
            func_name = stmt.func_var.name
            if func_name in self.func_records:
                args = self.process_call(func_name, list(stmt.args))
                return LaunchKernelStmt(
                    stmt.func_var,
                    args,
                    stmt.grid_dim,
                    stmt.cluster_dim,
                    stmt.block_dim,
                    stmt.shared_mem_bytes,
                    stmt.target,
                )
        return super().visit_LaunchKernelStmt(stmt)

    def visit_TensorElement(self, e: TensorElement):
        if e.base in self.current_record.param_to_tensor:
            assert isinstance(e.base, Var)
            tensor: Tensor = self.current_record.param_to_tensor[e.base]
            if tensor in self.prologues:
                prologue: Prologue = self.prologues[tensor]
                remap = {ti: self.current_record.tensor_to_updated_param[self.tensor_map[ti]] for ti in prologue.inputs}
                remap.update({a: b for a, b in zip(prologue.axes, self.visit(e.indices))})
                return rewrite(prologue.expr, remap)
        return super().visit_TensorElement(e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        if stmt.buf in self.current_record.param_to_tensor:
            tensor: Tensor = self.current_record.param_to_tensor[stmt.buf]
            if tensor in self.epilogues:
                epilogue: Epilogue = self.epilogues[tensor]
                indices = self.visit(stmt.indices)
                value = self.visit(stmt.value)

                remap = {ti: self.current_record.tensor_to_updated_param[self.tensor_map[ti]] for ti in epilogue.inputs}
                remap.update({a: b for a, b in zip(epilogue.axes, indices)})
                remap.update({epilogue.value: value})
                out_indices = rewrite(epilogue.out_indices, remap)
                out_expr = rewrite(epilogue.out_expr, remap)
                out_buf = self.current_record.tensor_to_updated_param[epilogue.out_tensor]
                return BufferStoreStmt(out_buf, out_indices, out_expr)

        return super().visit_BufferStoreStmt(stmt)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallOp):
            call = stmt.expr
            op = call.op
            if isinstance(op, CollectiveStore):
                assert self.epilogue_visitor_tree is not None
                inputs2stub: Dict[Tensor, Var] = self.epilogue_visitor_tree.inputs2stub
                tensor2collective_store: Dict[
                    Tensor, CollectiveStoreRecord
                ] = self.epilogue_visitor_tree.tensor2collective_store
                tensor2stmts: Dict[Tensor, List[Stmt]] = self.epilogue_visitor_tree.tensor2stmts
                src = self.visit(op.src)
                dst = op.dst
                assert dst in self.current_record.param_to_tensor
                tensor: Tensor = self.current_record.param_to_tensor[dst]
                collective_store = tensor2collective_store[tensor]
                output = self.fused_graph.outputs[0]
                stmts = tensor2stmts[output]
                offsets = [self.visit(e) for e in op.offsets]
                tensor_shape = tensor.shape
                tensor_layout = TensorLayout(tensor_shape, compact_row_major(tensor_shape))
                base = simplify(tensor_layout(offsets))
                buf = self.current_record.tensor_to_updated_param[output]
                remap = {
                    collective_store.src: src,
                    collective_store.buf_stub: ~buf[[0] * len(output.shape)],
                    collective_store.base: base,
                }
                for i, v in inputs2stub.items():
                    buf = self.current_record.tensor_to_updated_param[i]
                    remap[v] = ~buf[[0] * len(i.shape)]
                extents = [self.visit(e) for e in op.extents] if op.extents else None
                if extents:
                    remap.update({a: b for a, b in zip(collective_store.extents, extents)})
                new_stmt = SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
                new_stmt = rewrite(new_stmt, remap)
                return new_stmt
        return super().visit_EvaluateStmt(stmt)


class PrologueEpilogueNotFuseRewriter(IRRewriter):
    """
    Given a fused graph with anchor operator, we generate separate kernels for each prologue and epilogue around
    the anchor operator, and generate a new launch function that calls these kernels in correct order.

    For example, consider the following fused graph:

    ```
    x1: input
    x2: input
    x3 = matmul(x1 + 1, relu(x2) + x1)   # signature: c = matmul(a, b)
    x4 = gelu(x3)
    x5 = x4 + x1
    ```

    where `matmul` is the anchor operator.

    The given ir module would be like:

    ```
    def launch(a, b, c):
        ...
    ```

    We will generate three new kernels for the prologue for a and b, as well as the epilogue for c.

    ```
    original_launch = launch    # rename the original launch function

    def prologue_a(x1, a):
        a = x1 + 1

    def prologue_b(x1, x2, b):
        b = relu(x2) + x1

    def epilogue_c(c, x1, x5):
        x5 = gelu(c) + x1

    def launch(x1, x2, x5):
        buf = allocate work space for all intermediate tensors
        a = &buf[offset of a]
        b = &buf[offset of b]
        prologue_a(x1, a)
        prologue_b(x1, x2, b)
        original_launch(a, b, c)
        epilogue_c(c, x1, x5)
    ```

    We will implement the above algorithm in the following steps:
    1. Generate the Task for each prologue and epilogue.
    2. Use auto-scheduler to generate the kernel for each prologue and epilogue.
    3. Generate the new launch function.
    """

    class SubGraph:
        def __init__(
            self,
            inputs: List[Tensor],
            outputs: List[Tensor],
            nodes: List[Operator],
            ir_module: Optional[IRModule] = None,
        ):
            self.inputs: List[Tensor] = inputs
            self.outputs: List[Tensor] = outputs
            self.nodes: List[Operator] = nodes
            self.ir_module: Optional[IRModule] = ir_module

        @staticmethod
        def from_trace(output: Tensor, stop_nodes: List[Tensor]):
            # get the inputs by traversing the sub-graph from the output until we reach the stop nodes
            inputs: List[Tensor] = []
            visited: List[Tensor] = [output]
            queue: List[Tensor] = [output]
            operators: List[Operator] = []
            while len(queue) > 0:
                tensor: Tensor = queue.pop()
                if tensor.trace is None or tensor in stop_nodes:
                    inputs.append(tensor)
                else:
                    if tensor.op not in operators:
                        operators.append(tensor.op)
                    for x in tensor.op.inputs:
                        if x not in visited:
                            visited.append(x)
                            queue.append(x)

            # get a topological order of the nodes in the sub-graph
            directed_graph = DirectedGraph()
            for u in operators:
                directed_graph.add_node(u)
            for u in operators:
                for x in u.inputs:
                    if x.trace and x.op in operators:
                        v = x.op
                        directed_graph.add_edge(v, u)
            nodes: List[Operator] = directed_graph.topological_order()

            return PrologueEpilogueNotFuseRewriter.SubGraph(inputs, [output], nodes)

        def generate_ir_module(self, target, working_dir) -> IRModule:
            input_nodes: List[TensorNode] = [tensor_input('x', tensor.dtype, tensor.shape) for tensor in self.inputs]
            tensor2compute: Dict[Tensor, TensorNode] = {a: b for a, b in zip(self.inputs, input_nodes)}

            for op in self.nodes:
                task: Task = op.task
                remap: Dict[TensorNode, TensorNode] = {
                    task.inputs[i]: tensor2compute[op.inputs[i]] for i in range(len(task.inputs))
                }
                for tensor_output, compute_output in zip(op.outputs, task.outputs):
                    tensor2compute[tensor_output] = rewrite(compute_output, remap)
            fused_outputs: List[TensorNode] = [tensor2compute[tensor] for tensor in self.outputs]
            task = Task(name='fused_sub_graph', inputs=input_nodes, outputs=fused_outputs)
            ir_modules = task.implement(target=target, working_dir=working_dir)
            assert len(ir_modules) == 1
            return ir_modules[0]

    def __init__(self, fused_task: FusedTask, target: Target, working_dir: str):
        super().__init__()
        self.fused_task: FusedTask = fused_task
        self.fused_graph: FlowGraph = fused_task.fused_graph
        self.anchor_operator: Operator = self.fused_graph.nodes[fused_task.anchor]
        self.target: Target = target
        self.working_dir: str = working_dir

    def generate_sub_graphs(self, anchor_module: IRModule) -> List[SubGraph]:
        SubGraph = self.SubGraph
        sub_graphs: List[SubGraph] = []
        for output in self.fused_graph.outputs:
            if output in self.anchor_operator.outputs:
                # the graph output is one of the output of the anchor operator, no epilogue, skip
                continue
            sub_graphs.append(SubGraph.from_trace(output, self.anchor_operator.outputs + self.fused_graph.inputs))
        sub_graphs.append(
            SubGraph(self.anchor_operator.inputs, self.anchor_operator.outputs, [self.anchor_operator], anchor_module)
        )
        for anchor_input in self.anchor_operator.inputs:
            if anchor_input in self.fused_graph.inputs:
                # the graph input is one of the input of the anchor operator, no prologue, skip
                continue
            sub_graphs.append(SubGraph.from_trace(anchor_input, self.fused_graph.inputs))

        sub_graphs = list(reversed(sub_graphs))  # sort the sub-graphs in topological order
        return sub_graphs

    def schedule_sub_graphs(self, sub_graphs: List[SubGraph]):
        for sub_graph in sub_graphs:
            if sub_graph.ir_module is not None:
                # this is the sub-graph corresponding to the anchor operator, we already have the kernel
                continue
            sub_graph.ir_module = sub_graph.generate_ir_module(self.target, self.working_dir)
        # rename the functions in each sub-graph to avoid name conflict
        for idx, sub_graph in enumerate(sub_graphs):
            sub_graph.ir_module = rename_funcs(
                ir_module=sub_graph.ir_module,
                rmap={name: 'module_{}_{}'.format(idx, name) for name in sub_graph.ir_module.functions.keys()},
            )

    def generate_wrapper_module(self, sub_graphs: List[SubGraph]) -> IRModule:
        from hidet.lang import attrs, meta

        device = sub_graphs[0].outputs[0].device
        with hidet.script_module() as script_module:
            param_types: List[BaseType] = []
            for t in self.fused_graph.inputs + self.fused_graph.outputs:
                param_types.append(tensor_pointer_type(t.dtype, t.shape))

            def generate(params: Sequence[Var]):
                sb = StmtBuilder()

                param_tensors = self.fused_graph.inputs + self.fused_graph.outputs
                tensor2var: Dict[Tensor, Var] = {tensor: param for tensor, param in zip(param_tensors, params)}

                # allocate work space for all intermediate tensors
                intermediate_tensors: List[Tensor] = []
                tensor2offset: Dict[Tensor, Expr] = {}
                workspace_size: Expr = int64.zero
                alignment: int = 128  # 128 bytes alignment
                for sub_graph in sub_graphs:
                    for tensor in sub_graph.outputs:
                        if tensor not in tensor2var:
                            intermediate_tensors.append(tensor)
                            tensor2var[tensor] = tensor_pointer_var('buf', tensor.shape, tensor.dtype)
                            tensor2offset[tensor] = workspace_size
                            tensor_size = tensor.dtype.nbytes * prod([int64(v) for v in tensor.shape])
                            workspace_size = workspace_size + (tensor_size + alignment - 1) // alignment * alignment
                            sb += DeclareStmt(tensor2var[tensor])

                buffer_var: Var = tensor_pointer_var('workspace', [workspace_size], 'uint8')
                sb += DeclareStmt(buffer_var)
                if device.is_cpu():
                    sb += AssignStmt(buffer_var, request_cpu_workspace(workspace_size))
                elif device.is_cuda():
                    sb += AssignStmt(buffer_var, request_cuda_workspace(workspace_size))
                else:
                    raise NotImplementedError(f'Unsupported device {device}')

                for tensor in intermediate_tensors:
                    sb += AssignStmt(tensor2var[tensor], ~buffer_var[tensor2offset[tensor]])

                # launch the sub-graphs launch function in order
                for idx, sub_graph in enumerate(sub_graphs):
                    launch_func_name = 'module_{}_{}'.format(idx, 'launch')
                    func_var = sub_graph.ir_module.lookup_var(launch_func_name)
                    args = [tensor2var[tensor] for tensor in sub_graph.inputs + sub_graph.outputs]
                    sb += Call(func_var, tuple(args))

                return sb.finish()

            @hidet.script
            def launch(p: meta.types(param_types)):
                attrs.func_kind = 'public'
                generate(p)

        # include all the functions and variables in the ir module of sub-graphs into the new module
        ir_module = script_module.ir_module()
        for sub_graph in sub_graphs:
            ir_module.functions.update(sub_graph.ir_module.functions)
            ir_module.global_vars.update(sub_graph.ir_module.global_vars)
            ir_module.extern_functions.update(sub_graph.ir_module.extern_functions)
            ir_module.include_headers.extend(sub_graph.ir_module.include_headers)
            ir_module.include_dirs.extend(sub_graph.ir_module.include_dirs)
            ir_module.linking_dirs.extend(sub_graph.ir_module.linking_dirs)
            ir_module.linking_libs.extend(sub_graph.ir_module.linking_libs)
            ir_module.object_files.extend(sub_graph.ir_module.object_files)

        # unique the include headers and dirs
        ir_module.include_headers = list(set(ir_module.include_headers))
        ir_module.include_dirs = list(set(ir_module.include_dirs))
        ir_module.linking_dirs = list(set(ir_module.linking_dirs))
        ir_module.linking_libs = list(set(ir_module.linking_libs))
        ir_module.object_files = list(set(ir_module.object_files))

        return ir_module

    def visit_IRModule(self, module: IRModule) -> IRModule:
        # 1. Generate the Task for each prologue and epilogue.
        SubGraph = self.SubGraph
        sub_graphs: List[SubGraph] = self.generate_sub_graphs(module)

        # 2. Use auto-scheduler to generate the kernel for each prologue and epilogue.
        self.schedule_sub_graphs(sub_graphs)

        # 3. Generate the new launch function.
        ir_module = self.generate_wrapper_module(sub_graphs)

        return ir_module


class FusePrologueEpiloguePass(Pass):
    def __init__(self, fused_task: FusedTask, target: Target, working_dir: str):
        super().__init__()
        self.fused_task: FusedTask = fused_task
        self.target: Target = target
        self.working_dir: str = working_dir

    def process_module(self, ir_module: IRModule) -> IRModule:
        extractor = PrologueEpilogueExtractor(self.fused_task)
        prologues, epilogues, tensor_map = extractor.extract()

        marker = PrologueEpilogueMarker(self.fused_task, prologues, epilogues)
        marks: Dict[str, Dict[Var, Tensor]] = marker.mark(ir_module)

        collective_store_extractor = CollectiveStoreExtractor(marks)
        tensor2collective_store, tile_shape = collective_store_extractor.extract(ir_module)

        if len(tensor2collective_store) > 0:
            # pylint: disable=broad-except
            try:
                tile_derivation = TileDerivation(self.fused_task.fused_graph.nodes[self.fused_task.anchor], tile_shape)
                tile_derivation(self.fused_task.fused_graph)

                epilogue_visitor_tree = EpilogueVisitorRewriter(
                    self.fused_task, tile_derivation.tensor2tile, tensor2collective_store
                )
                epilogue_visitor_tree(self.fused_task.fused_graph)
            except Exception:
                # This is a fallback mechanism for epilogue fusion. As long as we
                # have a generic tile derivation algorithm to handle dynamic
                # shapes, we can remove the fallback mechanism. Because currently, we
                # only support limited situations in dynamic shape scenarios.
                # We can extend the tile derivation algorithm when
                # `torch.compile(...)` for dynamic shapes is fixed.
                from hidet.transforms.cute.generic.canonicalize import canonicalize_pass
                from hidet.transforms.cute.generic.canonicalize_arithmetic_expression import (
                    canonicalize_arithmetic_expression_pass,
                )
                from hidet.transforms.cute.generic.deadcode_elimination import deadcode_elimination_pass

                from hidet.transforms.cute.cuda.resolve_bank_conflict import resolve_bank_conflict_pass
                from hidet.transforms.cute.cuda.vectorize_elementwise import vectorize_elementwise_pass
                from hidet.transforms.cute.cuda.shared_memory_allocation import shared_memory_allocation_pass
                from hidet.transforms.cute.cuda.instruction_selection import instruction_selection_pass
                from hidet.transforms.cute.cuda.instantiate_auto_annotation import instantiate_auto_annotation_pass

                from hidet.logging import logger

                logger.warning(
                    "Epilogue fusion goes fallback, which is unusual. "
                    "This happens when enabling dynamic shapes, otherwise "
                    "this may indicate a bug. Please report to the Hidet team."
                )
                transforms = [
                    canonicalize_arithmetic_expression_pass(),
                    canonicalize_pass(),
                    deadcode_elimination_pass(),
                    instantiate_auto_annotation_pass(),
                    vectorize_elementwise_pass(),
                    instruction_selection_pass(),
                    resolve_bank_conflict_pass(),
                    instruction_selection_pass(),
                    shared_memory_allocation_pass(),
                ]
                ir_module = lower_with(ir_module, transforms)
                rewriter = LowerCuteDialectToBufferStoreStmtRewriter()
                ir_module = rewriter.rewrite(ir_module)
                epilogue_visitor_tree = None
            # pylint: enable=broad-except
        else:
            epilogue_visitor_tree = None

        try:
            rewriter = PrologueEpilogueRewriter(
                self.fused_task, prologues, epilogues, tensor_map, marks, epilogue_visitor_tree
            )
            ir_module = rewriter.rewrite(ir_module)

            rewriter = RearrangeEliminateRewriter()
            return rewriter.rewrite(ir_module)
        except CanNotFuseError:
            pass
        # there are some invalid usages of tensors with prologue/epilogue, we can not fuse them
        # fallback to generate separate kernels for all prologue/epilogue
        rewriter = PrologueEpilogueNotFuseRewriter(self.fused_task, self.target, self.working_dir)
        return rewriter.rewrite(ir_module)


def fuse_prologue_epilogue_pass(fused_task: FusedTask, target: Target, working_dir: str):
    return FusePrologueEpiloguePass(fused_task, target, working_dir)


def apply_prologue_epilogue(ir_module: IRModule, fused_task: FusedTask, target: Target, working_dir: str) -> IRModule:
    from hidet.transforms import inline_function_pass, declare_to_let_pass, inline_let_stmt_pass
    from hidet.transforms import flatten_tensor_slice_pass, PassContext, SaveIRInstrument, ProfileInstrument
    from hidet.transforms import generate_launch_func_pass

    transforms = [
        generate_launch_func_pass(),
        flatten_tensor_slice_pass(),
        inline_function_pass(),
        declare_to_let_pass(),
        inline_let_stmt_pass(inline_all=False),
        fuse_prologue_epilogue_pass(fused_task, target, working_dir),
    ]
    instruments = []
    if hidet.option.get_save_lower_ir():
        fused_candidate_dir = os.path.join(working_dir, './fuse_ir', ir_module.namespace)
        instruments.append(SaveIRInstrument(out_dir=fused_candidate_dir))
        instruments.append(ProfileInstrument(log_file=os.path.join(fused_candidate_dir, 'profile.txt')))

    with PassContext(instruments=instruments):
        ir_module = lower_with(ir_module, transforms)

    return ir_module


def apply_prologue_epilogue_batch(
    anchor_modules: List[IRModule], fused_task: FusedTask, target: Target, working_dir: str
) -> List[IRModule]:
    from hidet.utils.multiprocess import parallel_imap
    from tqdm import tqdm

    def _apply_prologue_epilogue_batch(args):
        return apply_prologue_epilogue(*args)

    if len(anchor_modules) > 1:
        for i, module in enumerate(anchor_modules):
            module.namespace = f'candidate_{i}'

    jobs = [(m, fused_task, target, working_dir) for m in anchor_modules]
    fused_modules: List[IRModule] = list(
        tqdm(parallel_imap(_apply_prologue_epilogue_batch, jobs), desc='Appling fusing', total=len(jobs), ncols=80)
    )
    return fused_modules
