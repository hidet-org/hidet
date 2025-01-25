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
We referenced the shared memory allocation algorithm in Triton
https://github.com/triton-lang/triton/blob/main/lib/Analysis/Allocation.cpp

The algorithm is originally proposed in the following paper:
Jordan Gergov. 1999. Algorithms for Compile-time Memory Optimization.
In Proceedings of the Tenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA '99).

The algorithm first analyzes the liveness of each buffer. In the design of the CuTe dialect, a tensor will be sliced or
partitioned after its creation, and then be consumed by copy or computation operators. Therefore, we first do the tensor
alias analysis for each tensor, and then analyze the liveness of each buffer. The liveness of a buffer is the range of
operations that use the buffer. In this way, each buffer corresponds to a triple (start, end, size). The proposed
algorithm employs the following process to determine the offset of each buffer:

1. Set: J is the triple set of all buffers, H is an auxiliary set of triples and initialized to {(0, minimum start of J,
maximum end of J)}.
2. While J is not empty:
    a. Select a triple (w, x, y) from H with the smallest w. Remove (w, x, y) from H.
    b. If there exists a buffer z in J such that (x, y) intersects with the liveness of z, and all other triples in H do
       not intersect with the liveness of z:
        i. Set the offset of z to w. Remove z from J.
        ii. Add (w + size of z, max(x, start of z), min(y, end of z)) to H.
        if x < start of z:
            Add (w, x, start of z) to H.
        if end of z < y:
            Add (w, end of z, y) to H.
3. Now, a function, alpha: buf -> offset, maps each buffer to an offset has been determined. Then, we should resolve the
   interference between buffers.
4. First, we build the interference graph of buffers. If two buffers, vi and vj, intersect with the following
   conditions:
    a. The liveness of the two buffers intersect.
    b. The (alpha(vi), alpha(vi) + si) and (alpha(vj), alpha(vj) + sj) intersect.
5. We color the interference graph with the First-Fit coloring algorithm.
6. Finally, we update the offset of each buffer based on the coloring result.
   beta(vi) = alpha(vi) + color(vi) * max(beta(vj) + size of vj), where vj is the neighbor of vi in the interference
   graph.

The paper claims the algorithm has O(nlogn) time complexity. Please refer to the paper for more details.

Classes:
    - Interval: Represents a range with a start and end.
    - SharedMemoryAllocationAnalysis: Analyzes the shared memory allocation for CUDA kernels.
    - SharedMemoryOffsetAnnotation: Annotates the IR with shared memory offsets.
    - ApplySharedMemoryUsageUpdate: Updates the shared memory usage in the IR based on the analysis.
    - SharedMemoryAllocationPass: A pass that processes an IR module for shared memory allocation.
"""
from typing import List, Dict, Union
import sys

from hidet.ir.tools import infer_type
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.module import IRModule
from hidet.ir.func import Function
from hidet.transforms.base import Pass
from hidet.ir.stmt import DeclareStmt, ForStmt, LaunchKernelStmt, BlackBoxStmt
from hidet.ir.expr import Var, is_constant, if_then_else
from hidet.utils import same_list

from hidet.ir.cute.expr import Op, CallOp, CConst
from hidet.ir.cute.ops import Tensor, Copy, Rearrange, Reduce
from hidet.transforms.cute.analysis import TensorAliasAnalysis, TensorInfo

from hidet.logging import logger, setConsoleLevel, DEBUG

from .lower_ops import request_smem_nbytes


verbose = False


class Interval:
    """
    Represents a range with a start and end.

    Attributes:
        start (int): The start of the interval.
        end (int): The end of the interval.
        size (int): The size of the interval.

    Methods:
        contains(addr): Checks if an address is within the interval.
        intersects(other): Checks if the interval intersects with another interval.
        __eq__(other): Checks if two intervals are equal.
        __lt__(other): Compares two intervals.
        __str__(): Returns a string representation of the interval.
    """

    def __init__(self, start: int = 0, end: int = sys.maxsize):
        self._start = start
        self._end = end

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def size(self):
        return self._end - self._start

    def contains(self, addr: int) -> bool:
        """
        Checks if an address is within the interval.

        Args:
            addr (int): The address to check.

        Returns:
            bool: True if the address is within the interval, False otherwise.
        """
        return self._start <= addr < self._end

    def intersects(self, other) -> bool:
        """
        Checks if the interval intersects with another interval.

        Args:
            other (Interval): The other interval to check.

        Returns:
            bool: True if the intervals intersect, False otherwise.
        """
        return self.start < other.end and other.start < self.end

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.end == other.end

    def __lt__(self, other) -> bool:
        if self.start < other.start:
            return True
        if other.start < self.start:
            return False
        return self.end < other.end

    def __str__(self):
        return f"[{self.start}, {self.end})"


class SharedMemoryAllocationAnalysis(IRVisitor):
    """
    Analyzes the shared memory allocation for CUDA kernels.

    Attributes:
        var2tensor (Dict[Var, TensorInfo]): Mapping from variables to tensor information.
        tensor2users (Dict[Op, List[Op]]): Mapping from tensors to the operations that use them.
        operation_id (Dict[Op, int]): Mapping from operations to their IDs.
        buffers (List[Op]): List of buffer operations.
        buffer2liveness (Dict[Op, Interval]): Mapping from buffers to their liveness intervals.
        buffer2size (Dict[Op, int]): Mapping from buffers to their sizes.
        forstmt2local_tensors (Dict[ForStmt, List[Tensor]]): Mapping from For statements to local tensors
        (declared inside the loop).
        forstmt2non_local_tensors (Dict[ForStmt, Dict[Tensor, bool]]): Mapping from For statements to
        non-local tensors(declared outside the loop).
        for_stmt_stack (List[ForStmt]): Stack of For statements.
        shared_memory_size (int): Total size of shared memory required.

    Methods:
        current_for_stmt(): Returns the current For statement.
        visit_*: Visits various types of operations and statements.
        analyze(func): Analyzes the given function for shared memory allocation.
    """

    def __init__(self, var2tensor: Dict[Var, TensorInfo]):
        super().__init__()
        self.var2tensor: Dict[Var, TensorInfo] = var2tensor
        self.tensor2users: Dict[Op, List[Op]] = {}
        self.operation_id: Dict[Op, int] = {}
        self.buffers: List[Op] = []
        self.buffer2liveness: Dict[Op, Interval] = {}
        self.buffer2size: Dict[Op, int] = {}

        self.forstmt2local_tensors: Dict[ForStmt, List[Tensor]] = {}
        self.forstmt2non_local_tensors: Dict[ForStmt, Dict[Tensor, bool]] = {}
        self.for_stmt_stack: List[ForStmt] = []
        self.shared_memory_size: int = 0

    def current_for_stmt(self):
        if len(self.for_stmt_stack) > 0:
            return self.for_stmt_stack[-1]
        else:
            return None

    def _add_tensor(self, tensor: Tensor, is_reader: bool):
        for stmt in self.for_stmt_stack:
            if stmt not in self.forstmt2local_tensors or tensor not in self.forstmt2local_tensors[stmt]:
                if stmt in self.forstmt2non_local_tensors:
                    if tensor not in self.forstmt2non_local_tensors[stmt]:
                        self.forstmt2non_local_tensors[stmt][tensor] = is_reader
                else:
                    self.forstmt2non_local_tensors[stmt] = {tensor: is_reader}

    def _insert_tensor_and_user(self, tensor: Tensor, op: Op):
        if tensor in self.tensor2users:
            self.tensor2users[tensor].append(op)
        else:
            self.tensor2users[tensor] = [op]

    def visit_Tensor(self, op: Tensor):
        # tensor = op
        # self._insert_tensor_and_user(tensor, op)
        if op.scope.is_shared():
            self.buffers.append(op)

    # partition and subtensor are not the actual users of the tensor
    # so we could skip them to have better overlap between the tensors
    #    def visit_PartitionSrc(self, op: PartitionSrc):
    #        x_ty = infer_type(op.x)
    #        if x_ty.scope.is_shared():
    #            tensor = self.var2tensor[op.x].tensor
    #            self._add_tensor(tensor)
    #            self._insert_tensor_and_user(tensor, op)
    #
    #    def visit_PartitionDst(self, op: PartitionDst):
    #        x_ty = infer_type(op.x)
    #        if x_ty.scope.is_shared():
    #            tensor = self.var2tensor[op.x].tensor
    #            self._add_tensor(tensor)
    #            self._insert_tensor_and_user(tensor, op)
    #
    #    def visit_SubTensor(self, op: SubTensor):
    #        x_ty = infer_type(op.x)
    #        if x_ty.scope.is_shared():
    #            tensor = self.var2tensor[op.x].tensor
    #            self._add_tensor(tensor)
    #            self._insert_tensor_and_user(tensor, op)

    def visit_Rearrange(self, op: Rearrange):
        self.buffers.append(op)

    def visit_Reduce(self, op: Reduce):
        self.buffers.append(op)

    def visit_Copy(self, op: Copy):
        src_ty = infer_type(op.src)
        if src_ty.scope.is_shared():
            src_tensor = self.var2tensor[op.src].tensor
            self._add_tensor(src_tensor, True)
            self._insert_tensor_and_user(src_tensor, op)
        dst_ty = infer_type(op.dst)
        if dst_ty.scope.is_shared():
            dst_tensor = self.var2tensor[op.dst].tensor
            self._add_tensor(dst_tensor, False)
            self._insert_tensor_and_user(dst_tensor, op)

    def visit_CallOp(self, call: CallOp):
        op = call.op
        self.visit(op)
        size = len(self.operation_id.items())
        assert op not in self.operation_id
        self.operation_id[op] = size

    def _resolve_liveness_for_operator(self, op: Union[Reduce, Rearrange]):
        min_id = self.operation_id[op]
        max_id = min_id + 1
        self.buffer2liveness[op] = Interval(min_id, max_id)
        self.buffer2size[op] = request_smem_nbytes(op)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        self.visit(stmt.var)
        self.visit(stmt.init)
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = call.op
            if isinstance(op, Tensor):
                current_for_stmt = self.current_for_stmt()
                if current_for_stmt is not None:
                    if current_for_stmt not in self.forstmt2local_tensors:
                        self.forstmt2local_tensors[current_for_stmt] = [op]
                    else:
                        self.forstmt2local_tensors[current_for_stmt].append(op)

    def visit_ForStmt(self, stmt: ForStmt):
        self.for_stmt_stack.append(stmt)
        self.visit(stmt.extent)
        self.visit(stmt.body)
        size = len(self.operation_id.items())
        self.operation_id[stmt] = size
        if stmt in self.forstmt2non_local_tensors:
            for tensor, is_reader in self.forstmt2non_local_tensors[stmt].items():
                if is_reader:
                    self._insert_tensor_and_user(tensor, stmt)
        self.for_stmt_stack.pop()

    def _resolve_liveness_for_tensor(self, tensor: Tensor):
        from hidet.ir.cute import filter

        users = self.tensor2users[tensor]
        min_id = sys.maxsize
        max_id = -sys.maxsize - 1
        for user in users:
            op_id = self.operation_id[user]
            min_id = min(op_id, min_id)
            max_id = max(op_id + 1, max_id)
        self.buffer2liveness[tensor] = Interval(min_id, max_id)
        tensor_type = infer_type(tensor.make_call())
        self.buffer2size[tensor] = filter(tensor.layout).size() * tensor_type.dtype.nbits // 8

    def compute_offset(self):
        buffer_start = self.calculate_starts()

        interference_graph = self.build_interference_graph(buffer_start)
        self.allocate(buffer_start, interference_graph)
        while len(interference_graph) > 0:
            self.allocate(buffer_start, interference_graph)
            interference_graph = self.build_interference_graph(buffer_start)

        return buffer_start

    def calculate_starts(self):
        buffer_start: Dict[Op, int] = {}

        from collections import defaultdict

        triple_map = defaultdict(list)
        triple_map[0].append(Interval())
        x_buffers = [buffer for buffer in self.buffers]
        while len(x_buffers) > 0:
            size = next(iter(triple_map))
            val = triple_map[size]
            rng = val.pop()
            if len(val) == 0:
                triple_map.pop(size)
            buffer_it = None
            for buffer in x_buffers:
                x_range = self.buffer2liveness[buffer]
                res = x_range.intersects(rng)
                if not res:
                    continue
                for _, v in triple_map.items():
                    for w in v:
                        res = res and (not w.intersects(x_range))
                        if not res:
                            break
                    if not res:
                        break
                if not res:
                    continue
                if res:
                    buffer_it = buffer
            if buffer_it is not None:
                x_size = self.buffer2size[buffer_it]
                x_range = self.buffer2liveness[buffer_it]
                # TODO: alignment in bytes should be a parameter of buffer
                alignment = 16
                aligned_size = ((size + alignment - 1) // alignment) * alignment
                buffer_start[buffer_it] = aligned_size
                triple_map[aligned_size + x_size].append(
                    Interval(max(rng.start, x_range.start), min(rng.end, x_range.end))
                )
                if rng.start < x_range.start:
                    triple_map[size].append(Interval(rng.start, x_range.start))
                if x_range.end < rng.end:
                    triple_map[size].append(Interval(x_range.end, rng.end))
                triple_map = defaultdict(list, sorted(triple_map.items(), key=lambda item: item[0]))
                x_buffers.remove(buffer_it)

        for buffer, start in buffer_start.items():
            logger.debug(f"buffer: {buffer}, start: {start}")
        for buffer, size in self.buffer2size.items():
            logger.debug(f"buffer: {buffer}, size: {size}")
        for buffer, liveness in self.buffer2liveness.items():
            logger.debug(f"buffer: {buffer}, liveness: {liveness}")
        return buffer_start

    def build_interference_graph(self, buffer_start: Dict[Op, int]):
        from collections import defaultdict

        interference_graph = defaultdict(set)

        for x in self.buffers:
            for y in self.buffers:
                if x is y:
                    continue
                x_start = buffer_start[x]
                y_start = buffer_start[y]
                x_size = self.buffer2size[x]
                y_size = self.buffer2size[y]
                x_size_range = Interval(x_start, x_start + x_size)
                y_size_range = Interval(y_start, y_start + y_size)
                x_op_range = self.buffer2liveness[x]
                y_op_range = self.buffer2liveness[y]
                if x_op_range.intersects(y_op_range) and x_size_range.intersects(y_size_range):
                    interference_graph[x].add(y)

        return interference_graph

    def allocate(self, buffer_start: Dict[Op, int], interference_graph):
        self.shared_memory_size = 0

        colors: Dict[Op, int] = {}
        for i, buffer in enumerate(self.buffers):
            colors[buffer] = 0 if i == 0 else -1

        for buffer in self.buffers:
            available: List[bool] = [True for _ in self.buffers]
            for y in interference_graph[buffer]:
                color = colors[y]
                if color >= 0:
                    available[color] = False
            it = next(i for i, c in enumerate(available) if c)
            colors[buffer] = it

        for buffer in self.buffers:
            adj = 0
            for y in interference_graph[buffer]:
                adj = max(adj, buffer_start[y] + self.buffer2size[y])
            buffer_offset = buffer_start[buffer] + colors[buffer] * adj
            buffer_start[buffer] = buffer_offset
            self.shared_memory_size = max(self.shared_memory_size, buffer_offset + self.buffer2size[buffer])

    def analyze(self, func: Function):
        self.visit(func)
        if "cuda.dynamic_smem_bytes" in func.attrs:
            shared_memory_size = func.attrs["cuda.dynamic_smem_bytes"]
        else:
            shared_memory_size = 0

        for buffer in self.buffers:
            if isinstance(buffer, (Reduce, Rearrange)):
                self._resolve_liveness_for_operator(buffer)
            elif isinstance(buffer, Tensor):
                self._resolve_liveness_for_tensor(buffer)

        buffer_start = self.compute_offset()
        shared_memory_size = if_then_else(
            shared_memory_size > self.shared_memory_size, shared_memory_size, self.shared_memory_size
        )

        logger.debug("============ finalize ===============")
        for buffer, start in buffer_start.items():
            logger.debug(f"buffer: {buffer}, start: {start}")
        logger.debug(f"shared_memory_size: {self.shared_memory_size}")
        logger.debug("=====================================")
        return shared_memory_size, buffer_start


class SharedMemoryOffsetAnnotation(IRRewriter):
    def __init__(self, buffer_start: Dict[Op, int]):
        super().__init__()
        self.buffer_start: Dict[Op, int] = buffer_start

    def visit_Tensor(self, e: Tensor):
        if e.scope.is_shared():
            buffer_start = self.buffer_start[e]
            annotations: Dict[str, CConst] = {}
            annotations["smem_offset"] = buffer_start
            return e.reforward([], annotations_update=annotations)
        return super().visit_Tensor(e)

    def visit_Reduce(self, e: Reduce):
        x = self.visit(e.x)
        buffer_start = self.buffer_start[e]
        annotations: Dict[str, CConst] = {}
        annotations["smem_offset"] = buffer_start
        return e.reforward([x], annotations_update=annotations)

    def visit_Rearrange(self, e: Rearrange):
        x = self.visit(e.x)
        buffer_start = self.buffer_start[e]
        annotations: Dict[str, CConst] = {}
        annotations["smem_offset"] = buffer_start
        return e.reforward([x], annotations_update=annotations)


class ApplySharedMemoryUsageUpdate(IRRewriter):
    def __init__(self, func2dyn_smem: Dict[str, int]):
        super().__init__()
        self.func2dyn_smem = func2dyn_smem

    def visit_Function(self, func: Function):
        func = super().visit_Function(func)
        if func.kind == "cuda_kernel":
            updated_dynamic_smem_bytes = self.func2dyn_smem.get(func.name, None)
            if "cuda.dynamic_smem_bytes" in func.attrs and updated_dynamic_smem_bytes is not None:
                original_dynamic_smem_bytes = func.attrs["cuda.dynamic_smem_bytes"]
                if is_constant(original_dynamic_smem_bytes):
                    if int(original_dynamic_smem_bytes) == 0:
                        func.attrs["cuda.dynamic_smem_bytes"] = updated_dynamic_smem_bytes
                    elif int(original_dynamic_smem_bytes) < updated_dynamic_smem_bytes:
                        raise RuntimeError(
                            f"User has set the shared memory size(got:{original_dynamic_smem_bytes}), but"
                            f"it less than the calculated size({updated_dynamic_smem_bytes})."
                            "Please double check the program to see if this is expected."
                        )
                else:
                    func.attrs["cuda.dynamic_smem_bytes"] = updated_dynamic_smem_bytes
        return func

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        template_string = r'cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});'
        if stmt.template_string == template_string:
            assert len(stmt.exprs) == 2
            func, smem = [self.visit(x) for x in stmt.exprs]
            func_name = func.name
            new_smem = self.func2dyn_smem.get(func_name, None)
            orig_smem = stmt.exprs[1]
            if new_smem is not None and (
                is_constant(orig_smem) and (int(orig_smem) < new_smem and int(orig_smem) != 0)
            ):
                raise RuntimeError(
                    f"User has set the shared memory size(got:{orig_smem}), so it is "
                    "dangerous to let the compiler allocate shared memory for tensors."
                    "Please double check the program to see if this is expected."
                )
            if all(x is y for x, y in zip([func, new_smem], stmt.exprs)):
                return stmt
            else:
                return BlackBoxStmt(template_string, func, smem)
        else:
            return super().visit_BlackBoxStmt(stmt)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        func_var = self.visit(stmt.func_var)
        args = [self.visit(e) for e in stmt.args]
        grid_dim = tuple(self.visit(d) for d in stmt.grid_dim)
        block_dim = tuple(self.visit(d) for d in stmt.block_dim)
        cluster_dim = tuple(self.visit(d) for d in stmt.cluster_dim)

        if func_var.name in self.func2dyn_smem:
            shared_mem_bytes = self.func2dyn_smem[func_var.name]
        else:
            shared_mem_bytes = self.visit(stmt.shared_mem_bytes)
        if same_list(
            [func_var, *args, *grid_dim, *cluster_dim, *block_dim, shared_mem_bytes],
            [stmt.func_var, *stmt.args, *stmt.cluster_dim, *stmt.grid_dim, *stmt.block_dim, stmt.shared_mem_bytes],
        ):
            return stmt
        else:
            return LaunchKernelStmt(func_var, args, grid_dim, cluster_dim, block_dim, shared_mem_bytes, stmt.target)


class SharedMemoryAllocationPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        orig_level = None
        if verbose:
            orig_level = logger.level
            setConsoleLevel(DEBUG)

        new_funcs = {}
        func2dyn_smem = {}
        for name, func in ir_module.functions.items():
            if func.kind == "cuda_kernel":
                tensor_alias_analysis = TensorAliasAnalysis()
                var2tensor = tensor_alias_analysis.analyze(func)
                shared_memory_allocation = SharedMemoryAllocationAnalysis(var2tensor)
                shared_memory_size, buffer_start = shared_memory_allocation.analyze(func)
                shared_memory_annotation = SharedMemoryOffsetAnnotation(buffer_start)
                func = shared_memory_annotation(func)
                func2dyn_smem[name] = shared_memory_size
                new_funcs[name] = func
            else:
                new_funcs[name] = func
        if any(new_funcs[name] is not ir_module.functions[name] for name in new_funcs):
            ir_module = ir_module.copy().reset_funcs(new_funcs, ir_module.global_vars)

        rewriter = ApplySharedMemoryUsageUpdate(func2dyn_smem)

        if verbose:
            setConsoleLevel(orig_level)
        return rewriter(ir_module)


def shared_memory_allocation_pass() -> Pass:
    return SharedMemoryAllocationPass()
