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

# Support for instructions covered in the PTX document section 9.7.16.7,
# "Tensor Memory Allocation and Management Instructions"
# reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-alloc-manage-instructions

from typing import Union

from hidet.ir import PointerType, VoidType
from hidet.ir.stmt import asm
from hidet.utils import initialize
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda


@initialize()
def register_tmem_allocation():
    from hidet.lang import script, i32, u32, attrs
    from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared

    # tcgen05.alloc - Single CTA version
    func_name = 'cuda_tcgen05_alloc'
    template_string = 'tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;'

    @script
    def cuda_tcgen05_alloc(dst_addr: PointerType(VoidType()), num_cols: i32):  # type: ignore
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        dst_smem_addr = cvta_generic_to_shared(dst_addr)
        asm(template=template_string, inputs=[dst_smem_addr, num_cols])

    assert isinstance(cuda_tcgen05_alloc, Function)
    register_primitive_function(name=cuda_tcgen05_alloc.name, func_or_type=cuda_tcgen05_alloc)

    # tcgen05.alloc - CTA Pair version
    funct_name = 'cuda_tcgen05_alloc_cta_pair'
    template_string = 'tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;'

    @script
    def cuda_tcgen05_alloc_cta_pair(dst_addr: PointerType(VoidType()), num_cols: i32):  # type: ignore
        attrs.func_name = funct_name
        attrs.func_kind = 'cuda_internal'
        dst_smem_addr = cvta_generic_to_shared(dst_addr)
        asm(template=template_string, inputs=[dst_smem_addr, num_cols])

    assert isinstance(cuda_tcgen05_alloc_cta_pair, Function)
    register_primitive_function(name=cuda_tcgen05_alloc_cta_pair.name, func_or_type=cuda_tcgen05_alloc_cta_pair)

    # tcgen05.dealloc - Single CTA version
    func_name = 'cuda_tcgen05_dealloc'
    template_string = 'tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;'

    @script
    def cuda_tcgen05_dealloc(tmem_addr: u32, num_cols: i32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string, inputs=[tmem_addr, num_cols])

    assert isinstance(cuda_tcgen05_dealloc, Function)
    register_primitive_function(name=cuda_tcgen05_dealloc.name, func_or_type=cuda_tcgen05_dealloc)

    # tcgen05.dealloc - CTA Pair version
    func_name = 'cuda_tcgen05_dealloc_cta_pair'
    template_string = 'tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;'

    @script
    def cuda_tcgen05_dealloc_cta_pair(tmem_addr: u32, num_cols: i32):
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string, inputs=[tmem_addr, num_cols])

    assert isinstance(cuda_tcgen05_dealloc_cta_pair, Function)
    register_primitive_function(name=cuda_tcgen05_dealloc_cta_pair.name, func_or_type=cuda_tcgen05_dealloc_cta_pair)

    # tcgen05.relinquish_alloc_permit - Single CTA version
    func_name = 'cuda_tcgen05_relinquish_alloc_permit'
    template_string = 'tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;'

    @script
    def cuda_tcgen05_relinquish_alloc_permit():
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string)

    assert isinstance(cuda_tcgen05_relinquish_alloc_permit, Function)
    register_primitive_function(
        name=cuda_tcgen05_relinquish_alloc_permit.name, func_or_type=cuda_tcgen05_relinquish_alloc_permit
    )

    # tcgen05.relinquish_alloc_permit - CTA Pair version
    func_name = 'cuda_tcgen05_relinquish_alloc_permit_cta_pair'
    template_string = 'tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;'

    @script
    def cuda_tcgen05_relinquish_alloc_permit_cta_pair():
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string)

    assert isinstance(cuda_tcgen05_relinquish_alloc_permit_cta_pair, Function)
    register_primitive_function(
        name=cuda_tcgen05_relinquish_alloc_permit_cta_pair.name,
        func_or_type=cuda_tcgen05_relinquish_alloc_permit_cta_pair,
    )


def tcgen05_alloc(dst_addr: Expr, num_cols: Union[Expr, int], use_cta_pair: bool = False):
    """
    Allocate Tensor Memory.

    See Also:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit

    Parameters
    ----------
    dst_addr: Expr
        The address in the shared memory, used to store the returned TMEM address.
        For more details about the tensor memory address:
        https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing
    num_cols: Union[Expr, int]
        The number of columns to allocate.
        Must be a power of 2, at least 32 and at most 512.
    use_cta_pair: bool
        Whether to use a CTA pair.

    """
    func_name = 'tcgen05_alloc_cta_pair' if use_cta_pair else 'tcgen05_alloc'
    return call_cuda(func_name, [dst_addr, num_cols])


def tcgen05_dealloc(tmem_addr: Expr, num_cols: Union[Expr, int], use_cta_pair: bool = False):
    """
    Deallocate the Tensor Memory.

    Parameters
    ----------
    tmem_addr: Expr
        The address of the Tensor Memory to deallocate.
    num_cols: Union[Expr, int]
        The number of columns to deallocate.
        Must be a power of 2, at least 32 and at most 512.
    """
    func_name = 'tcgen05_dealloc_cta_pair' if use_cta_pair else 'tcgen05_dealloc'
    return call_cuda(func_name, [tmem_addr, num_cols])


def tcgen05_relinquish_alloc_permit(use_cta_pair: bool = False):
    func_name = 'tcgen05_relinquish_alloc_permit_cta_pair' if use_cta_pair else 'tcgen05_relinquish_alloc_permit'
    return call_cuda(func_name, [])


# Helper functions for tensor memory address computation
def compute_tmem_address(lane: int, column: int) -> int:
    """
    Compute the tensor memory address encoding.

    The address is a 32-bit value where:
    - Bits 31-16: lane index (0-127)
    - Bits 15-0: column index (0-511)
    """
    assert 0 <= lane <= 127, f"Lane must be 0-127, got {lane}"
    assert 0 <= column <= 511, f"Column must be 0-511, got {column}"
    return (lane << 16) | column


def compute_tmem_offset_address(
    base_addr: int, lane_offset: int, column_offset: int, allocated_columns: int = 512
) -> int:
    """
    Compute a tensor memory address as an offset from a base address.

    TMEM uses a 32-bit addressing scheme where addresses encode a 2D position:
    - Bits 31-16: Lane ID (row) ranging from 0-127
    - Bits 15-0: Column ID ranging from 0-511

    Parameters:
    -----------
    base_addr : int
        The base TMEM address returned by tcgen05.alloc
        Format: 0xLLLLCCCC where LLLL is lane (16 bits) and CCCC is column (16 bits)
    lane_offset : int
        Additional lane offset to add to base lane
        Must result in a lane within [0, 127]
    column_offset : int
        Additional column offset to add to base column
        Must not exceed the allocated column range
    allocated_columns : int, default=512
        Number of columns allocated for this TMEM block
        Must be power of 2: 32, 64, 128, 256, or 512

    Returns:
    --------
    int : The computed TMEM address

    Raises:
    -------
    ValueError : If the computed address would be out of bounds

    Example:
    --------
    >>> # Assume we allocated 256 columns starting at column 0
    >>> base = 0x00000000  # Lane 0, Column 0
    >>> # Access element at lane 5, column 10
    >>> addr = compute_tmem_offset_address(base, 5, 10, allocated_columns=256)
    >>> # addr = 0x0005000A (lane 5 << 16 | column 10)
    """

    # Extract base lane from bits 22-16 (only 7 bits needed for 0-127)
    # Using 0x7F mask even though full lane field is 16 bits, since max lane is 127
    base_lane = (base_addr >> 16) & 0x7F

    # Extract base column from bits 8-0 (9 bits for 0-511)
    # 0x1FF = 0b111111111 = 511 (max column index)
    base_column = base_addr & 0x1FF

    # Calculate new position by adding offsets
    # NOTE: It is the responsibility of the caller to ensure the new lane and column are within bounds
    # NOTE: This function will be mainly used in Hidet Script codes, where
    # NOTE: I cannot add `if` checks without causing "internal exception occurred during transpiling this ast node"
    new_lane = base_lane + lane_offset
    new_column = base_column + column_offset

    # Reconstruct the 32-bit TMEM address
    # Lane goes in upper 16 bits, column in lower 16 bits
    # This matches the hardware's addressing scheme where:
    #   - addr[31:16] = lane_id
    #   - addr[15:0] = column_id
    return (new_lane << 16) | new_column


def get_register_count(shape: str, num: str) -> int:
    """Get the number of registers required for a given shape and num combination."""
    register_counts = {
        # .16x64b, .32x32b, and .16x32bx2 have same register requirements
        ("16x64b", "x1"): 1,
        ("16x64b", "x2"): 2,
        ("16x64b", "x4"): 4,
        ("16x64b", "x8"): 8,
        ("16x64b", "x16"): 16,
        ("16x64b", "x32"): 32,
        ("16x64b", "x64"): 64,
        ("16x64b", "x128"): 128,
        ("32x32b", "x1"): 1,
        ("32x32b", "x2"): 2,
        ("32x32b", "x4"): 4,
        ("32x32b", "x8"): 8,
        ("32x32b", "x16"): 16,
        ("32x32b", "x32"): 32,
        ("32x32b", "x64"): 64,
        ("32x32b", "x128"): 128,
        ("16x32bx2", "x1"): 1,
        ("16x32bx2", "x2"): 2,
        ("16x32bx2", "x4"): 4,
        ("16x32bx2", "x8"): 8,
        ("16x32bx2", "x16"): 16,
        ("16x32bx2", "x32"): 32,
        ("16x32bx2", "x64"): 64,
        ("16x32bx2", "x128"): 128,
        # .16x128b has different register requirements
        ("16x128b", "x1"): 2,
        ("16x128b", "x2"): 4,
        ("16x128b", "x4"): 8,
        ("16x128b", "x8"): 16,
        ("16x128b", "x16"): 32,
        ("16x128b", "x32"): 64,
        ("16x128b", "x64"): 128,
        # .16x256b has different register requirements
        ("16x256b", "x1"): 4,
        ("16x256b", "x2"): 8,
        ("16x256b", "x4"): 16,
        ("16x256b", "x8"): 32,
        ("16x256b", "x16"): 64,
        ("16x256b", "x32"): 128,
    }
    return register_counts[(shape, num)]
