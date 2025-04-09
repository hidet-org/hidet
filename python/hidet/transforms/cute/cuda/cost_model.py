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
Cost Model for Tile-Level Primitives in CUDA

This module implements a cost model for estimating the execution time of tile-level primitives in CUDA.
Tile-level primitives are high-level operations that are lowered into PTX instructions during compilation.

Key Concepts:
1. Tile-Level Primitives: High-level operations that operate on data tiles (e.g., matrix multiplication,
   reduction, rearrangement) that are eventually compiled into PTX instructions.

2. Instruction Selection: During compilation, each tile-level primitive is transformed into a sequence
   of PTX instructions. The specific instructions chosen depend on the hardware capabilities and
   optimization goals.

3. Cost Calculation: The cost of a tile-level primitive is calculated as:
   cost = number_of_instructions * instruction_latency
   where instruction_latency is determined through micro-benchmarking on the target hardware.

4. Latency Tables: The module maintains two sets of latency tables:
   - independent_cpi_lut: Latency for independent instructions (no data dependencies)
   - dependent_cpi_lut: Latency for dependent instructions (with data dependencies)

5. Operation Overlap: The cost model assumes perfect overlap between copy operations and MMA operations.

   For example, in a loop with multiple operations:
    ```
    for ko in range(cdiv(k, BK)):
        copy(ga[:, :, ko + stages - 1], sa[:, :, smem_write_pipe])
        copy(gb[:, :, ko + stages - 1], sb[:, :, smem_write_pipe])
        mma(sa[:, :, smem_read_pipe], sb[:, :, smem_read_pipe, rc])
        syncthreads()
    ```
   The overall latency of one iteration is calculated as:
    ```
    cost_issue_copy_g2s_a = num_instructions(copy_g2s_a) * latency_of_issue_g2s
    cost_issue_copy_g2s_b = num_instructions(copy_g2s_b) * latency_of_issue_g2s
    cost_issue_mma = num_instructions(mma) * latency_of_issue_mma

    cost_complete_copy_g2s_a = max(latency_of_complete_g2s - cost_issue_copy_g2s_b - cost_issue_mma, 0)
    cost_complete_copy_g2s_b = max(latency_of_complete_g2s - cost_issue_mma, 0)
    cost_complete_mma = latency_of_complete_mma

    cost_per_iteration = (
        cost_issue_copy_g2s_a + cost_issue_copy_g2s_b + cost_issue_mma +
        max(cost_complete_copy_g2s_a, cost_complete_copy_g2s_b, cost_complete_mma)
    )
    ```

    latency_of_issue_g2s = independent_cpi_lut[Opcode.Ldgsts]
    latency_of_complete_g2s = dependent_cpi_lut[Opcode.Ldgsts]

    We assume tensors in this example is all float16 tensors.
    latency_of_issue_mma = independent_cpi_lut[Opcode.Hmma]
    latency_of_complete_mma = dependent_cpi_lut[Opcode.Hmma]

Note: The latency values are based on micro-benchmarking results from research papers:
1. "Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis"
2. "Demystifying GPU Microarchitecture through Microbenchmarking"

Note: Currently, the cost model can capture the performance difference of different shared memory layouts
because the shared memory layout will affect the instruction selected to execute the copy operation.
Then, the instruction will affect the number of issued instructions and the completion latency.

Note: The impact of bank conflicts will be added in the future.

Note: Currently, the cost model assumes the pipelines are fully overlapped, and does not consider the
`cp_async_wait_group()` and `mbarrier` instructions. The support should be added in the future.

Note: Tensor manipulation operations (broadcast, transpose, partition, subtensor)
are lowered into arithmetic operations that calculate addresses. These
operations are considered to have negligible cost and are ignored in the
cost model.

6. How to use this cost model?
Say we have a function in Hidet IR:
```
with hidet.script_module() as module:
    @hidet.script
    def matmul(a: f16[m, k], b: f16[k, n]) -> f16[m, n]:
        # matmul kernel with tile-level operations
        ...

# compile the module to Hidet IR
# get the function from the module
func = ...

# create a cost model
cost_model = LatencyModel()

# predict the execution time of the function
cost = cost_model.predict(func)
```

Note: The cost model only captures the latency of the tile-level operations. It cannot apply to normal
Hidet IR functions.

Note: The cost model is intended to be used in layout synthesis pass. It can differentiate the performance
of different layout choices. The accuracy of the cost model is not guaranteed for normal Hidet IR functions.
"""
from typing import Dict, Union, Tuple
from abc import abstractmethod

import enum
from enum import auto as enum_auto

from hidet.ir.tools import infer_type
from hidet.ir.functors import IRVisitor
from hidet.ir.func import Function
from hidet.ir.stmt import DeclareStmt, AssignStmt, EvaluateStmt, ForStmt, IfStmt, SeqStmt
from hidet.ir.expr import is_constant, Var, Call, var
from hidet.ir import expr as ir

from hidet.ir.cute import flatten, TiledTensorLayout
from hidet.ir.cute.expr import Op, CallOp
from hidet.ir.cute.ops import (
    Rearrange,
    Reduce,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    ReduceMean,
    Arithmetic,
    Copy,
    Mma,
    Tensor,
    TensorView,
    PartitionSrc,
    PartitionDst,
    PartitionA,
    PartitionB,
    Transpose,
    Broadcast,
    SubTensor,
)
from hidet.ir.cute.ops.arithmetic import (
    Fill,
    Cast,
    Add,
    Sub,
    Multiply,
    Div,
    MultiplyAdd,
    Exp,
    Relu,
    Silu,
    RSqrt,
    ElementwiseMin,
    ElementwiseMax,
)


verbose = False


class CostModel(IRVisitor):
    """Base class for cost models that estimate execution time of functions.

    This is an abstract base class that defines the interface for cost models.
    Subclasses must implement the predict() method to provide specific cost estimation logic.
    """

    @abstractmethod
    def predict(self, func: Function) -> float:
        """Predict the execution time of a function.

        Args:
            func: The function to estimate execution time for.

        Returns:
            float: Estimated execution time in cycles.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError


class Opcode(enum.Enum):
    """Enumeration of PTX instruction opcodes used in the cost model.

    Each opcode represents a specific PTX instruction type with its associated latency.
    The opcodes are used to look up instruction latencies in the CPI (Cycles Per Instruction) tables.
    """

    # Arithmetic operations
    IAdd = enum_auto()  # Integer addition
    IMul = enum_auto()  # Integer multiplication
    FAdd = enum_auto()  # Float addition
    FMul = enum_auto()  # Float multiplication
    FFma = enum_auto()  # Float fused multiply-add
    HFma = enum_auto()  # Half-precision fused multiply-add
    FDiv = enum_auto()  # Float division
    FMin = enum_auto()  # Float minimum
    FMax = enum_auto()  # Float maximum
    FSqrt = enum_auto()  # Float square root
    FRsqrt = enum_auto()  # Float reciprocal square root
    FExp = enum_auto()  # Float exponential
    FSilu = enum_auto()  # SiLU activation function

    # Type conversion operations
    I2F = enum_auto()  # Integer to float conversion
    F2I = enum_auto()  # Float to integer conversion
    I2I = enum_auto()  # Integer to integer conversion
    F2F = enum_auto()  # Float to float conversion

    # Memory operations
    Ldg = enum_auto()  # Load from global memory
    Lds = enum_auto()  # Load from shared memory
    Ldgsts = enum_auto()  # Load from global to shared memory
    Sts = enum_auto()  # Store to shared memory
    Stg = enum_auto()  # Store to global memory

    # Matrix multiplication operations
    Hmma = enum_auto()  # Half-precision matrix multiply-accumulate
    Imma = enum_auto()  # Integer matrix multiply-accumulate
    Wgmma = enum_auto()  # Warpgroup-level matrix multiply-accumulate

    # Default opcode for unknown instructions
    Default = enum_auto()


# We don't differentiate the compute capability in the cost model because I
# found the latency of arithmetic instructions are almost the same for Ampere and Hopper
# architectures based on the micro-benchmarking results.
#
# CPI means Cycles Per Instruction.
# CPI for independent is extracted from the following paper
# Demystifying the Nvidia Ampere Architecture through Microbenchmarking and
# Instruction-level Analysis
# Authors: Hamdy A., Yehia A., Nandakishore S., Abdel-Hameed B.,
independent_cpi_lut = {
    Opcode.IAdd: 2,
    # assume mul.lo.u32
    Opcode.IMul: 2,
    Opcode.FAdd: 2,
    Opcode.FMul: 2,
    Opcode.FFma: 2,
    Opcode.HFma: 2,
    Opcode.FDiv: 253,
    Opcode.FMin: 2,
    Opcode.FMax: 2,
    Opcode.FSqrt: 40,
    Opcode.FRsqrt: 18,
    Opcode.FExp: 43,
    # div + add + exp
    Opcode.FSilu: 300,
    # cast
    Opcode.I2F: 2,
    Opcode.F2I: 2,
    Opcode.F2F: 2,
    # memory
    Opcode.Ldg: 2,
    Opcode.Lds: 2,
    Opcode.Ldgsts: 2,
    Opcode.Sts: 2,
    Opcode.Stg: 2,
    Opcode.Hmma: 8,
    Opcode.Imma: 8,
    Opcode.Wgmma: 8,
    # workaround to bypass unknown arithmetic instructions
    Opcode.Default: 2,
}


# CPI for dependent is measured via the method in the following paper
# Demystifying GPU Microarchitecture through Microbenchmarking
# Authors: Henry W., Misel-Myrto P., Maryam S., Andreas M.,
dependent_cpi_lut = {
    # Seems like the latency of arithmetic instructions have the same
    # clocks after Ampere architecture. So, we keep one table without
    # differentiate the compute capability.
    Opcode.IAdd: 5,
    # assume mul.lo.u32
    Opcode.IMul: 4,
    Opcode.FAdd: 4,
    Opcode.FMul: 4,
    Opcode.FFma: 4,
    Opcode.HFma: 4,
    Opcode.FDiv: 253,
    Opcode.FMin: 10,
    Opcode.FMax: 10,
    Opcode.FSqrt: 40,
    Opcode.FRsqrt: 18,
    Opcode.FExp: 43,
    Opcode.FSilu: 300,
    # cast
    Opcode.I2F: 23,
    Opcode.F2I: 23,
    Opcode.F2F: 6,
    # DRAM ~500 clocks: 541.5(RTX4090), 466.3(A100), 478.8(H800)
    # L2 Cache hit: 273.0(RTX4090), 261.5(A100), 263.0(H800)
    Opcode.Ldg: 280,
    Opcode.Lds: 30,
    Opcode.Ldgsts: 280,
    Opcode.Sts: 19,
    Opcode.Stg: 280,
    Opcode.Hmma: 16,
    Opcode.Imma: 16,
    # workaround to bypass unknown arithmetic instructions
    Opcode.Default: 4,
}


def get_op_code_for_reduce(op: Reduce) -> Opcode:
    """Determine the appropriate opcode for a reduce operation.

    Args:
        op: The reduce operation to analyze.

    Returns:
        Opcode: The corresponding PTX instruction opcode.

    Raises:
        NotImplementedError: If the input type is not float or the reduce operation is not supported.
    """
    inp_ty = infer_type(op.x)
    if not inp_ty.dtype.is_float():
        raise NotImplementedError(f"input of reduce operation should be float type.(input:{inp_ty})")
    if isinstance(op, (ReduceSum, ReduceMean)):
        return Opcode.FAdd
    elif isinstance(op, ReduceMax):
        return Opcode.FMax
    elif isinstance(op, ReduceMin):
        return Opcode.FMin
    else:
        raise NotImplementedError(f"Unsupported reduce operation.({op})")


def get_latency_for_reduce(op: Reduce) -> Tuple[int, bool]:
    """Calculate the latency for a reduce operation.

    This function computes the total latency including:
    1. Intra-thread reduction
    2. Intra-warp reduction using shuffle instructions
    3. Inter-warp reduction using shared memory

    Args:
        op: The reduce operation to analyze.

    Returns:
        Tuple[int, bool]: A tuple containing:
            - The total latency in cycles
            - Whether inter-warp reduction is needed
    """
    src_ty = infer_type(op.x)
    dst_ty = op.infer_type([src_ty])
    assert isinstance(src_ty.layout, TiledTensorLayout) and isinstance(dst_ty.layout, TiledTensorLayout)
    src_val = src_ty.layout.val_layout()
    dst_val = dst_ty.layout.val_layout()
    from hidet.ir.cute import compact_col_major
    from hidet.ir.cute.layout import common_reshape, group
    from hidet.utils.py import prod

    src_val, dst_val = common_reshape(src_val, dst_val)
    from hidet.transforms.cute.cuda.lower_ops.registry import get_op_emitter

    emitter_cls = get_op_emitter(op)
    reduce_emitter = emitter_cls()
    par, red, _, _ = reduce_emitter.canonicalize_val(src_val, dst_val)
    opcode = get_op_code_for_reduce(op)
    dependent_insts = prod(red)
    independent_insts = prod(par)
    # intra-thread reduce
    cycles = dependent_cpi_lut[opcode] * dependent_insts * independent_insts
    # intra-warp reduce
    dst_thr = dst_ty.layout.thr_layout()
    # one warp has 32 threads.
    WARP_SIZE = 32
    lane, warp = group(dst_thr, WARP_SIZE)
    flat_shape = flatten(lane.shape_tuple)
    flat_stride = flatten(lane.stride_tuple)
    costride = compact_col_major(flat_shape)
    out_loops = 0
    for s, d, d1 in zip(reversed(flat_shape), reversed(flat_stride), reversed(costride)):
        if d == 0:
            end = d1 * s // 2
            start = d1
            while end >= start:
                out_loops += 1
                end //= 2
    nr_regs = dst_val.count()
    nr_bits = nr_regs * src_ty.dtype.nbits
    from hidet.ir.dtypes import f64, u32, f16

    if nr_bits % 64 == 0:
        shfl_dtype = f64
    elif nr_bits % 32 == 0:
        shfl_dtype = u32
    elif nr_bits % 16 == 0:
        shfl_dtype = f16
    else:
        raise NotImplementedError(f"unable to shuffle data within a warp.(reduce_per_thread:{dst_val})")
    # latency of __shfl_xor_sync
    shfl_latency = 30
    iters = nr_bits // shfl_dtype.nbits
    cycles += shfl_latency * out_loops * iters
    # inter-warp reduce
    need_sync = reduce_emitter.require_inter_warp_reduce(warp)
    if need_sync:
        _, _, bits_per_inst = reduce_emitter.get_lds_sts(nr_bits)
        iters = nr_bits // bits_per_inst
        red_shape_warp = []
        current = WARP_SIZE
        for s, d in zip(warp.shape_tuple, warp.stride_tuple):
            if d == 0:
                red_shape_warp.append(s)
            current = current * s
        sts_latency = (
            independent_cpi_lut[Opcode.Sts] * iters + dependent_cpi_lut[Opcode.Sts] - independent_cpi_lut[Opcode.Sts]
        )

        sts_lds_latency = prod(red_shape_warp) * (
            iters * independent_cpi_lut[Opcode.Lds]
            + dependent_cpi_lut[Opcode.Lds]
            - independent_cpi_lut[Opcode.Lds]
            + iters * independent_cpi_lut[Opcode.Sts]
            + dependent_cpi_lut[Opcode.Sts]
            - independent_cpi_lut[Opcode.Sts]
        )
        lds_latecny = independent_cpi_lut[Opcode.Lds] * iters
        cycles += sts_latency + sts_lds_latency + lds_latecny
    return cycles, need_sync


def get_mma_latency_to_complete(opcode: Opcode, inst) -> int:
    """Calculate the latency for a matrix multiply-accumulate operation to complete.

    Args:
        opcode: The opcode of the MMA operation.
        inst: The MMA instruction instance.

    Returns:
        int: The latency in cycles for the operation to complete.
    """
    if opcode == Opcode.Wgmma:
        _, n, _ = inst.shape_mnk
        return n * 128 // 256
    else:
        lat = dependent_cpi_lut.get(opcode, None)
        assert lat is not None, f"Unknown opcode({opcode})"
        return lat


class LatencyModel(CostModel):
    """A cost model that estimates the latency of a function based on instruction-level cycle count.

    This model emulates the execution of the function by visiting the IR and counting cycles.
    It tracks the remaining cycles for tile-level instructions and maintains a mapping of variables
    to their last writing operation.
    """

    def __init__(self):
        """Initialize the latency model."""
        super().__init__()
        # Operators that generate spatial loops can be on-the-fly. For example,
        # a copy operation, copy(a, b), will be lowered into a loop:
        #   for i in range(num_insts):
        #       copy_inst(a[i], b[i])
        # The copy_insts inside the loop are all independent instructions, so we only
        # record the cycles that we need to issue these instructions. The cycles of
        # waiting this operation to complete is:
        #   dependent_cpi_lut[copy_inst] - independent_cpi_lut[copy_inst]
        #
        # Operators that can be on-the-fly:
        # - mma: Matrix multiply-accumulate operations
        # - copy: Memory copy operations
        # - elementwise: Elementwise arithmetic operations
        # - rearrange: Rearrange operations
        #
        # Operators that cannot be on-the-fly:
        # - reduce: Contains inter-thread synchronization
        #
        # If an operator consumes the result of an on-the-fly operator, the producer
        # operator in on-the-fly table will be synced.
        #
        # RAW (Read After Write) latency means the cycles that we need to wait until
        # the operands are ready to read (i.e., the cycles of waiting the instruction
        # that produces the operand to finish).
        self.ops_on_the_fly: Dict[Op, int] = {}
        self.cycles = 0
        # last op that writes to a variable
        self.var2op: Dict[Var, Op] = {}

    def visit_Reduce(self, red: Reduce) -> int:
        """Visit a reduce operation and calculate its latency.

        This method handles three types of reduction:
        1. Intra-thread reduction
        2. Intra-warp reduction using shuffle instructions
        3. Inter-warp reduction using shared memory

        Args:
            red: The reduce operation to analyze.

        Returns:
            int: The total latency in cycles.
        """
        # Check Read After Write (RAW) dependency - wait for the input operand to be ready
        cycles = 0
        raw = red.x  # The input operand that needs to be ready before reduction
        op = self.var2op[raw]
        if op in self.ops_on_the_fly:
            cycles = self.ops_on_the_fly[op]
            self.ops_on_the_fly.pop(op)
        # op latency
        red_cycles, need_sync = get_latency_for_reduce(red)
        cycles += red_cycles
        # pop instructions that are finished
        pops = []
        if need_sync:
            sync_cycles = 0
            for op, remain in self.ops_on_the_fly.items():
                sync_cycles = max(sync_cycles, remain)
                pops.append(op)
            for op in pops:
                self.ops_on_the_fly.pop(op)
            cycles += sync_cycles
        else:
            for op, remain in self.ops_on_the_fly.items():
                if remain > cycles:
                    self.ops_on_the_fly[op] -= cycles
                else:
                    pops.append(op)
            for op in pops:
                self.ops_on_the_fly.pop(op)
        return cycles

    def _is_cast(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a cast operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a cast operation.
        """
        if isinstance(op, Cast):
            return True
        else:
            args = [var('v', infer_type(v).dtype) for v in op.inputs]
            expr = op.op(*args)
            return isinstance(expr, ir.Cast)

    def _is_add(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is an addition operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is an addition operation.
        """
        if isinstance(op, Add):
            return True
        else:
            args = [var('v', infer_type(v).dtype) for v in op.inputs]
            expr = op.op(*args)
            return isinstance(expr, ir.Add)

    def _is_sub(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a subtraction operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a subtraction operation.
        """
        if isinstance(op, Sub):
            return True
        else:
            args = [var('v', infer_type(v).dtype) for v in op.inputs]
            expr = op.op(*args)
            return isinstance(expr, ir.Sub)

    def _is_mul(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a multiplication operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a multiplication operation.
        """
        if isinstance(op, Multiply):
            return True
        else:
            args = [var('v', infer_type(v).dtype) for v in op.inputs]
            expr = op.op(*args)
            return isinstance(expr, ir.Multiply)

    def _is_div(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a division operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a division operation.
        """
        if isinstance(op, Div):
            return True
        else:
            args = [var('v', infer_type(v).dtype) for v in op.inputs]
            expr = op.op(*args)
            return isinstance(expr, ir.Div)

    def _is_mad(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a multiply-add operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a multiply-add operation.
        """
        return isinstance(op, MultiplyAdd)

    def _is_exp(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is an exponential operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is an exponential operation.
        """
        return isinstance(op, Exp)

    def _is_relu(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a ReLU operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a ReLU operation.
        """
        return isinstance(op, Relu)

    def _is_silu(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a SiLU operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a SiLU operation.
        """
        return isinstance(op, Silu)

    def _is_rsqrt(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is a reciprocal square root operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is a reciprocal square root operation.
        """
        return isinstance(op, RSqrt)

    def _is_min(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is an elementwise minimum operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is an elementwise minimum operation.
        """
        return isinstance(op, ElementwiseMin)

    def _is_max(self, op: Arithmetic) -> bool:
        """Check if an arithmetic operation is an elementwise maximum operation.

        Args:
            op: The arithmetic operation to check.

        Returns:
            bool: True if the operation is an elementwise maximum operation.
        """
        return isinstance(op, ElementwiseMax)

    def _get_op_code_for_arith(self, op: Arithmetic) -> Opcode:
        """Determine the appropriate opcode for an arithmetic operation.

        Args:
            op: The arithmetic operation to analyze.

        Returns:
            Opcode: The corresponding PTX instruction opcode.

        Raises:
            NotImplementedError: If the operation type is not supported.
        """
        if isinstance(op, Fill):
            inp_ty = infer_type(op.inputs[0])
            if inp_ty.dtype.is_float():
                return Opcode.FAdd
            elif inp_ty.dtype.is_integer():
                return Opcode.IAdd
            else:
                raise NotImplementedError(f"Unsupported fill operation.({op})")
        call = op.make_call()
        out_ty = infer_type(call)
        if self._is_cast(op):
            inp_ty = infer_type(op.inputs[0])
            if inp_ty.dtype.is_integer() and out_ty.dtype.is_float():
                return Opcode.I2F
            elif inp_ty.dtype.is_integer() and out_ty.dtype.is_integer():
                return Opcode.I2F
            elif inp_ty.dtype.is_float() and out_ty.dtype.is_integer():
                return Opcode.F2I
            elif inp_ty.dtype.is_float() and out_ty.dtype.is_float():
                return Opcode.F2F
            else:
                raise NotImplementedError(f"Unsupported cast operation.(src:{inp_ty.dtype},dst:{out_ty.dtype})")
        elif out_ty.dtype.is_float():
            if self._is_add(op):
                return Opcode.FAdd
            elif self._is_sub(op):
                return Opcode.FAdd
            elif self._is_mul(op):
                return Opcode.FMul
            elif self._is_div(op):
                return Opcode.FDiv
            elif self._is_mad(op):
                return Opcode.FFma
            elif self._is_exp(op):
                return Opcode.FExp
            elif self._is_relu(op):
                return Opcode.FMax
            elif self._is_silu(op):
                return Opcode.FSilu
            elif self._is_rsqrt(op):
                return Opcode.FRsqrt
            elif self._is_min(op):
                return Opcode.FMin
            elif self._is_max(op):
                return Opcode.FMax
            else:
                return Opcode.Default
        elif out_ty.dtype.is_integer():
            if self._is_add(op):
                return Opcode.IAdd
            elif self._is_sub(op):
                return Opcode.IAdd
            elif self._is_mul(op):
                return Opcode.IMul
            else:
                return Opcode.Default
        else:
            raise NotImplementedError(f"Unsupported arithmetic operation.({op})")

    def visit_Arithmetic(self, arith: Arithmetic) -> int:
        """Visit an arithmetic operation and calculate its latency.

        Args:
            arith: The arithmetic operation to analyze.

        Returns:
            int: The total latency in cycles.
        """
        # Check Read After Write (RAW) dependencies - wait for all input operands to be ready
        cycles = 0
        for raw in arith.inputs:  # Each input operand that needs to be ready
            op = self.var2op[raw]
            if op in self.ops_on_the_fly:
                raw_cycles = self.ops_on_the_fly[op]
                cycles = max(cycles, raw_cycles)  # Take the maximum latency among all inputs
        # op latency
        if isinstance(arith, Fill):
            inp_ty = infer_type(arith.inputs[0])
            val = inp_ty.layout.val_layout()
            nr_regs = val.count()
        else:
            call = arith.make_call()
            out_ty = infer_type(call)
            if isinstance(out_ty.layout, TiledTensorLayout):
                val = out_ty.layout.val_layout()
            else:
                val = out_ty.layout
            nr_regs = val.count()
        # one instruction per output element
        issued_insts = nr_regs
        opcode = self._get_op_code_for_arith(arith)
        issue_latency = independent_cpi_lut[opcode]
        issue_cycles = issue_latency * issued_insts
        cycles += issue_cycles
        # pop instructions that are finished
        pops = []
        for op, remain in self.ops_on_the_fly.items():
            if remain > cycles:
                self.ops_on_the_fly[op] -= cycles
            else:
                pops.append(op)
        for op in pops:
            self.ops_on_the_fly.pop(op)
        if dependent_cpi_lut[opcode] > issue_latency:
            self.ops_on_the_fly[arith] = dependent_cpi_lut[opcode] - issue_latency
        return cycles

    def _get_element_per_access(
        self, flat_shape: Tuple[int, ...], flat_stride: Tuple[int, ...], element_per_access: int, m: int
    ) -> int:
        """Calculate the number of elements that can be accessed in one memory operation.

        Args:
            flat_shape: The flattened shape of the tensor.
            flat_stride: The flattened stride of the tensor.
            element_per_access: The maximum number of elements that can be accessed in one operation.
            m: The stride value to check against.

        Returns:
            int: The number of elements that can be accessed in one operation.
        """
        for s, d in zip(flat_shape, flat_stride):
            if d == m:
                element_per_access = min(element_per_access, s)
        return element_per_access

    def visit_Rearrange(self, rng: Rearrange) -> int:
        """Visit a rearrange operation and calculate its latency.

        The rearrange operation exchanges data through shared memory, requiring synchronization.

        Args:
            rng: The rearrange operation to analyze.

        Returns:
            int: The total latency in cycles.
        """
        # currently, we always exchange the data through shared memory. So
        # there would be a synchronization inside the rearrange operation,
        # which indicates that we should pop all the operator on the fly.
        cycles = 0
        pops = []
        for op, remain in self.ops_on_the_fly.items():
            cycles = max(cycles, remain)
            pops.append(op)
        for op in pops:
            self.ops_on_the_fly.pop(op)
        src_ty = infer_type(rng.x)
        src_layout = src_ty.layout
        m, _ = src_layout.shape()
        src_val_layout = src_layout.val_layout()
        flat_shape = flatten(src_val_layout.shape_tuple)
        flat_stride = flatten(src_val_layout.stride_tuple)
        element_per_access = 128 // src_ty.dtype.nbits
        element_per_access = self._get_element_per_access(flat_shape, flat_stride, element_per_access, m)
        num_sts = src_val_layout.size() // element_per_access
        sts_latency = independent_cpi_lut[Opcode.Sts]
        cycles += num_sts * sts_latency + dependent_cpi_lut[Opcode.Sts] - sts_latency
        dst_val_layout = rng.layout.val_layout()
        flat_shape = flatten(dst_val_layout.shape_tuple)
        flat_stride = flatten(dst_val_layout.stride_tuple)
        element_per_access = 128 // src_ty.dtype.nbits
        element_per_access = self._get_element_per_access(flat_shape, flat_stride, element_per_access, m)
        num_lds = dst_val_layout.size() // element_per_access
        lds_latency = independent_cpi_lut[Opcode.Lds]
        cycles += num_lds * lds_latency
        self.ops_on_the_fly[rng] = dependent_cpi_lut[Opcode.Lds] - lds_latency
        return cycles

    def _get_op_code_for_copy(self, op: Copy) -> Opcode:
        """Determine the appropriate opcode for a copy operation.

        Args:
            op: The copy operation to analyze.

        Returns:
            Opcode: The corresponding PTX instruction opcode.

        Raises:
            NotImplementedError: If the source and destination scopes are not supported.
        """
        src_ty = infer_type(op.src)
        dst_ty = infer_type(op.dst)
        if src_ty.scope.is_global() and dst_ty.scope.is_shared():
            return Opcode.Ldgsts
        elif src_ty.scope.is_global() and dst_ty.scope.is_register():
            return Opcode.Ldg
        elif src_ty.scope.is_shared() and dst_ty.scope.is_register():
            return Opcode.Lds
        elif src_ty.scope.is_register() and dst_ty.scope.is_shared():
            return Opcode.Sts
        elif src_ty.scope.is_register() and dst_ty.scope.is_global():
            return Opcode.Stg
        else:
            raise NotImplementedError(f"Unsupported opcode for copy({src_ty.scope}, {dst_ty.scope}) operation")

    def visit_Copy(self, copy: Copy) -> int:
        """Visit a copy operation and calculate its latency.

        The cost model assumes perfect overlap between copy operations and MMA operations.
        The latency calculation takes into account:
        1. Issue latency: Time to issue all copy instructions
        2. Completion latency: Time for the last instruction to complete, adjusted for overlap

        Args:
            copy: The copy operation to analyze.

        Returns:
            int: The total latency in cycles.
        """
        # Check Read After Write (RAW) dependency - wait for the source operand to be ready
        op = self.var2op[copy.src]  # The source operand that needs to be ready before copying
        cycles = 0
        if op in self.ops_on_the_fly:
            cycles = self.ops_on_the_fly[op]
        annotations = copy.annotations
        assert len(annotations) > 0
        src_layout = annotations["src_layout"]
        dst_layout = annotations["dst_layout"]
        bank_conflicts = annotations.get("bank_conflicts", None)
        src_elements = src_layout[1].size()
        dst_elements = dst_layout[1].size()
        assert src_elements == dst_elements, f"elements mismatch.(src:{src_elements},dst:{dst_elements})"
        opcode = self._get_op_code_for_copy(copy)
        issue_latency = independent_cpi_lut[opcode]
        issued_insts = src_elements
        # the code generation will issue independent copy instructions in a loop
        # so we need to multiply the latency by the number of issued instructions
        issue_cycles = issue_latency * issued_insts
        cycles += issue_cycles
        pops = []
        for op, remain in self.ops_on_the_fly.items():
            if remain > cycles:
                self.ops_on_the_fly[op] -= cycles
            else:
                pops.append(op)
        for op in pops:
            self.ops_on_the_fly.pop(op)
        # the cycles to wait for the last instruction to complete
        # Note: The completion latency is adjusted for overlap with subsequent operations
        bank_conflicts_penalty = 1 if bank_conflicts is None else bank_conflicts
        self.ops_on_the_fly[copy] = dependent_cpi_lut[opcode] * bank_conflicts_penalty - issue_latency
        self.var2op[copy.dst] = copy
        return cycles

    def _get_op_code_for_mma(self, inst) -> Opcode:
        """Determine the appropriate opcode for a matrix multiply-accumulate operation.

        Args:
            inst: The MMA instruction instance.

        Returns:
            Opcode: The corresponding PTX instruction opcode.

        Raises:
            NotImplementedError: If the input types are not supported.
        """
        from .instruction_selection import WgmmaAsyncInstruction

        if isinstance(inst, WgmmaAsyncInstruction):
            return Opcode.Wgmma
        elif inst.a_dtype.is_float() and inst.b_dtype.is_float():
            return Opcode.Hmma
        elif inst.a_dtype.is_integer() and inst.b_dtype.is_integer():
            return Opcode.Imma
        else:
            raise NotImplementedError(f"Unsupported opcode for mma({inst.a_dtype}, {inst.b_dtype}) operation")

    def visit_Mma(self, mma: Mma) -> int:
        """Visit a matrix multiply-accumulate operation and calculate its latency.

        The cost model assumes perfect overlap between copy operations and MMA operations.
        The latency calculation takes into account:
        1. Issue latency: Time to issue all MMA instructions
        2. Completion latency: Time for the last instruction to complete, adjusted for overlap

        Args:
            mma: The MMA operation to analyze.

        Returns:
            int: The total latency in cycles.
        """
        # Check Read After Write (RAW) dependencies - wait for all input operands to be ready
        cycles = 0
        raws = mma.a, mma.b, mma.c  # The input operands that need to be ready before MMA
        for raw in raws:
            op = self.var2op[raw]
            if op in self.ops_on_the_fly:
                raw_cycles = self.ops_on_the_fly[op]
                cycles = max(cycles, raw_cycles)  # Take the maximum latency among all inputs
        annotations = mma.annotations
        assert len(annotations) > 0
        a_rest = annotations["a_rest"]
        d_rest = annotations["d_rest"]
        inst = annotations["inst"]
        issued_insts = d_rest[:2].size() * a_rest[1].size()
        opcode = self._get_op_code_for_mma(inst)
        issue_latency = independent_cpi_lut[opcode]
        issue_cycles = issue_latency * issued_insts
        cycles += issue_cycles
        pops = []
        for op, remain in self.ops_on_the_fly.items():
            if remain > cycles:
                self.ops_on_the_fly[op] -= cycles
            else:
                pops.append(op)
        for op in pops:
            self.ops_on_the_fly.pop(op)
        # the cycles to wait for the last instruction to complete
        lat_to_complete = get_mma_latency_to_complete(opcode, inst)
        if lat_to_complete > issue_latency:
            self.ops_on_the_fly[mma] = lat_to_complete - issue_latency
        self.var2op[mma.d] = mma
        return cycles

    def visit_Broadcast(self, op: Broadcast) -> int:
        """Visit a broadcast operation.

        Broadcast operations only compute memory addresses, so they have no latency.

        Args:
            op: The broadcast operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_Transpose(self, op: Transpose) -> int:
        """Visit a transpose operation.

        Transpose operations only compute memory addresses, so they have no latency.

        Args:
            op: The transpose operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_Tensor(self, tensor: Tensor) -> int:
        """Visit a tensor operation.

        Tensor operations only compute memory addresses, so they have no latency.

        Args:
            tensor: The tensor operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_TensorView(self, tensor: TensorView) -> int:
        """Visit a tensor view operation.

        Tensor view operations only compute memory addresses, so they have no latency.

        Args:
            tensor: The tensor view operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_PartitionSrc(self, op: PartitionSrc) -> int:
        """Visit a source partition operation.

        Partition operations only compute memory addresses, so they have no latency.

        Args:
            op: The source partition operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_PartitionDst(self, op: PartitionDst) -> int:
        """Visit a destination partition operation.

        Partition operations only compute memory addresses, so they have no latency.

        Args:
            op: The destination partition operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_PartitionA(self, op: PartitionA) -> int:
        """Visit a partition A operation.

        Partition operations only compute memory addresses, so they have no latency.

        Args:
            op: The partition A operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_PartitionB(self, op: PartitionB) -> int:
        """Visit a partition B operation.

        Partition operations only compute memory addresses, so they have no latency.

        Args:
            op: The partition B operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def visit_SubTensor(self, op: SubTensor) -> int:
        """Visit a sub-tensor operation.

        Sub-tensor operations only compute memory addresses, so they have no latency.

        Args:
            op: The sub-tensor operation to analyze.

        Returns:
            int: 0 cycles (no latency).
        """
        return 0

    def zerofy(self, latency: Union[float, None]) -> float:
        """Convert None latency to 0.

        Args:
            latency: The latency value to process.

        Returns:
            float: The processed latency value (0 if input is None).
        """
        if latency is None:
            return 0
        else:
            return latency

    def visit_ForStmt(self, stmt: ForStmt) -> int:
        """Visit a for loop statement and calculate its latency.

        Args:
            stmt: The for loop statement to analyze.

        Returns:
            int: The total latency in cycles.
        """
        num_iters = stmt.extent
        cycles = 0
        lat = self.visit(stmt.body)
        lat = self.zerofy(lat)
        cycles += lat
        if is_constant(num_iters):
            if num_iters > 1:
                lat = self.visit(stmt.body)
                lat = self.zerofy(lat)
                cycles += (num_iters - 1) * lat
        else:
            lat = self.visit(stmt.body)
            if lat is not None:
                # TODO: if we have a dynamic loop in the code, how to estimate the number of iterations and the latency?
                estimated_iters = 8
                cycles += (estimated_iters - 1) * lat
        return cycles

    def visit_SeqStmt(self, stmt: SeqStmt) -> int:
        """Visit a sequence of statements and calculate their total latency.

        Args:
            stmt: The sequence of statements to analyze.

        Returns:
            int: The total latency in cycles.
        """
        cycles = 0
        for s in stmt.seq:
            lat = self.visit(s)
            cycles += self.zerofy(lat)
        return cycles

    def visit_IfStmt(self, stmt: IfStmt) -> int:
        """Visit an if statement and calculate its latency.

        Currently uses a simple model that averages the latency of both branches.

        Args:
            stmt: The if statement to analyze.

        Returns:
            int: The estimated latency in cycles.
        """
        # TODO: find a better way to determine the latency of the if-else statement
        lat_then = self.visit(stmt.then_body)
        lat_else = self.visit(stmt.else_body)
        lat_then = self.zerofy(lat_then)
        lat_else = self.zerofy(lat_else)
        return 0.5 * lat_then + 0.5 * lat_else

    def visit_EvaluateStmt(self, stmt: EvaluateStmt) -> int:
        """Visit an evaluate statement and calculate its latency.

        Args:
            stmt: The evaluate statement to analyze.

        Returns:
            int: The latency in cycles.
        """
        if isinstance(stmt.expr, CallOp):
            call = stmt.expr
            op = call.op
            return self.visit(op)
        elif isinstance(stmt.expr, Call):
            func_var = stmt.expr.func_var
            func_name = func_var.name
            # synchronization of the entire thread block
            # we pop all the instructions on the fly
            if func_name == "cuda_syncthreads":
                cycles = 0
                pops = []
                for op, remain in self.ops_on_the_fly.items():
                    cycles = max(cycles, remain)
                    pops.append(op)
                for op in pops:
                    self.ops_on_the_fly.pop(op)
                return cycles
        return 0

    def visit_AssignStmt(self, stmt: AssignStmt) -> int:
        """Visit an assignment statement and calculate its latency.

        Args:
            stmt: The assignment statement to analyze.

        Returns:
            int: The latency in cycles.
        """
        if isinstance(stmt.value, CallOp):
            call = stmt.value
            op = call.op
            # last op that writes to the variable
            self.var2op[stmt.var] = op
            return self.visit(op)
        return 0

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> int:
        """Visit a declaration statement and calculate its latency.

        Args:
            stmt: The declaration statement to analyze.

        Returns:
            int: The latency in cycles.
        """
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = call.op
            # last op that writes to the variable
            self.var2op[stmt.var] = op
            return self.visit(op)
        return 0

    def visit_Function(self, func: Function) -> int:
        """Visit a function and calculate its total latency.

        Args:
            func: The function to analyze.

        Returns:
            int: The total latency in cycles.
        """
        return self.visit(func.body)

    def predict(self, func: Function) -> float:
        """Predict the execution time of a function.

        Args:
            func: The function to estimate execution time for.

        Returns:
            float: Estimated execution time in cycles.
        """
        return self.visit(func)
