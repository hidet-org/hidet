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
from typing import Optional, Tuple, Union

from hidet.ir.stmt import asm
from hidet.utils import initialize
from hidet.ir.expr import Expr
from hidet.ir.primitives import is_primitive_function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.tools.simplifier import simplify_to_int


def _get_register_count(shape: str, num: str) -> Optional[int]:
    """
    Get the number of registers required for a given shape and num combination.

    Based on Table 48 from PTX documentation section 9.7.16.8.3.

    Note: The .16x32bx2 shape uses the same register count as .16x64b since
    it performs two 16x32b accesses, which equals 16x64b in total data volume.
    """
    # Register count mapping: (shape, num) -> registers per thread
    register_counts = {
        # .16x64b and .32x32b have same register requirements
        # .16x32bx2 also uses these same counts (documented in PTX manual)
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
        # .16x128b has different register requirements
        ("16x128b", "x1"): 2,
        ("16x128b", "x2"): 4,
        ("16x128b", "x4"): 8,
        ("16x128b", "x8"): 16,
        ("16x128b", "x16"): 32,
        ("16x128b", "x32"): 64,
        ("16x128b", "x64"): 128,
        ("16x128b", "x128"): None,
        # .16x256b has different register requirements
        ("16x256b", "x1"): 4,
        ("16x256b", "x2"): 8,
        ("16x256b", "x4"): 16,
        ("16x256b", "x8"): 32,
        ("16x256b", "x16"): 64,
        ("16x256b", "x32"): 128,
        ("16x256b", "x64"): None,
        ("16x256b", "x128"): None,
    }
    return register_counts.get((shape, num))


def _register_tcgen05_ld_variant(shape: str, num: str, pack: bool):
    """
    Dynamically register a tcgen05.ld instruction variant.

    Supports the basic format:
    tcgen05.ld.sync.aligned.shape1.num{.pack}.b32 r, [taddr];

    This function uses the same pattern as wgmma.py: instead of having variable
    numbers of register parameters, we pass a single pointer that we cast and
    index into. This allows us to have a fixed function signature while still
    supporting different register counts through array indexing.
    """
    from hidet.lang import script, attrs
    from hidet.ir.dtypes import u32
    from hidet.ir.type import PointerType
    from hidet.ir.expr import cast

    reg_count = _get_register_count(shape, num)
    if reg_count is None:
        raise ValueError(f"Unsupported tcgen05.ld combination: {shape}.{num}")

    # Build function name
    func_name = f'cuda_tcgen05_ld_{shape}_{num}'
    if pack:
        func_name += '_pack_16b'

    # Build template string
    template_parts = ['tcgen05.ld.sync.aligned', shape, num]
    if pack:
        template_parts.append('pack::16b')
    template_parts.append('b32')

    # Create register list part of template
    reg_list = ', '.join([f'%{i}' for i in range(reg_count)])
    template_string = f'{".".join(template_parts)} {{{reg_list}}}, [%{reg_count}];'

    @script
    def cuda_tcgen05_ld(regs: ~u32, taddr: u32):  # type: ignore
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'

        # Cast the pointer to access as array - this pattern allows us to
        # support different register counts with the same function signature
        regs_ptr = cast(regs, PointerType("uint32"))

        # Ensure taddr is in a register by creating a local variable
        taddr_reg: u32 = taddr  # type: ignore

        asm(
            template=template_string,
            outputs=[regs_ptr[i] for i in range(reg_count)],
            inputs=[taddr_reg],
            is_volatile=True,
        )

    register_primitive_function(name=func_name, func_or_type=cuda_tcgen05_ld)


def _register_tcgen05_st_variant(shape: str, num: str, unpack: bool):
    """
    Dynamically register a tcgen05.st instruction variant.

    Supports the basic format:
    tcgen05.st.sync.aligned.shape1.num{.unpack}.b32 [taddr], r;
    """
    from hidet.lang import script, attrs
    from hidet.ir.dtypes import u32
    from hidet.ir.type import PointerType
    from hidet.ir.expr import cast

    reg_count = _get_register_count(shape, num)
    if reg_count is None:
        raise ValueError(f"Unsupported tcgen05.st combination: {shape}.{num}")

    # Build function name
    func_name = f'cuda_tcgen05_st_{shape}_{num}'
    if unpack:
        func_name += '_unpack_16b'

    # Build template string
    template_parts = ['tcgen05.st.sync.aligned', shape, num]
    if unpack:
        template_parts.append('unpack::16b')
    template_parts.append('b32')

    # Create register list part of template
    reg_list = ', '.join([f'%{i + 1}' for i in range(reg_count)])  # +1 because taddr is %0
    template_string = f'{".".join(template_parts)} [%0], {{{reg_list}}};'

    @script
    def cuda_tcgen05_st(taddr: u32, regs: ~u32):  # type: ignore
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'

        # Cast the pointer to access as array
        regs_ptr = cast(regs, PointerType("uint32"))

        # Ensure taddr is in a register by creating a local variable
        taddr_reg: u32 = taddr

        asm(template=template_string, inputs=[taddr_reg] + [regs_ptr[i] for i in range(reg_count)], is_volatile=True)

    register_primitive_function(name=func_name, func_or_type=cuda_tcgen05_st)


@initialize()
def register_tcgen05_wait():
    """Register tcgen05.wait instruction variants."""
    from hidet.lang import script, attrs

    # Register tcgen05.wait::ld
    func_name = 'cuda_tcgen05_wait_ld'
    template_string = 'tcgen05.wait::ld.sync.aligned;'

    @script
    def tcgen05_wait_ld():
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string, is_volatile=True)

    register_primitive_function(name=tcgen05_wait_ld.name, func_or_type=tcgen05_wait_ld)

    # Register tcgen05.wait::st
    func_name = 'cuda_tcgen05_wait_st'
    template_string = 'tcgen05.wait::st.sync.aligned;'

    @script
    def tcgen05_wait_st():
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'
        asm(template=template_string, is_volatile=True)

    register_primitive_function(name=tcgen05_wait_st.name, func_or_type=tcgen05_wait_st)


# Helper function to change the 'shape' argument to tcgen05_ld/st from a tuple to a string
def _shape_to_string(shape: Union[Tuple[int, int], Tuple[int, int, int]]) -> str:
    allowed_shapes = {
        (16, 64): "16x64b",
        (16, 128): "16x128b",
        (16, 256): "16x256b",
        (32, 32): "32x32b",
        (16, 32, 2): "16x32bx2",
    }
    if shape not in allowed_shapes:
        raise ValueError(f"Invalid shape: {shape}")
    return allowed_shapes[shape]


# tcgen05_ld function implementation below
def tcgen05_ld(
    regs: Expr,
    tmem_addr: Expr,
    shape: Union[str, Tuple[int, int], Tuple[int, int, int]] = "32x32b",
    num: Union[str, int] = "x1",
    pack: bool = False,
    imm_half_splitoff: Optional[Expr] = None,
):
    """
    Load data from Tensor Memory to registers.

    This implements the tcgen05.ld instruction formats:
    - Basic: tcgen05.ld.sync.aligned.shape1.num{.pack}.b32 r, [taddr];
    - 16x32bx2: tcgen05.ld.sync.aligned.16x32bx2.num{.pack}.b32 r, [taddr], immHalfSplitoff;

    This is a warp-wide instruction where all 32 threads in the warp must execute the same
    instruction and are synchronized as a warp.

    See Also:
    ---------
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-ld

    Parameters
    ----------
    regs: Expr
        Pointer to the destination registers to store the data. The number of registers accessed
        depends on the given shape and num combination (see Table 48 in PTX docs).
    tmem_addr: Expr
        Base address in Tensor Memory to load from. All threads in the warp must specify the same value.
        This is a 32-bit address where bits 31-16 represent the lane and bits 15-0 represent the column.
    shape: str
        The shape of the data load. Must be one of: "16x64b", "16x128b", "16x256b", "32x32b", "16x32bx2"
    num: str
        Number of times the shape is repeated. Must be one of: "x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128"
    pack: bool
        Whether to use packing mode where two 16-bit elements from adjacent columns are packed
        into a single 32-bit element during load.
    imm_half_splitoff: Optional[Expr]
        Required for shape "16x32bx2". Offset for the second access (taddr + imm_half_splitoff).

    Returns
    -------
    ret: Call
        The call expression, which will be lowered to a tcgen05.ld instruction with the given parameters.
    """

    if isinstance(shape, tuple):
        shape = _shape_to_string(shape)
    if isinstance(num, int):
        num = f"x{num}"

    # Handle .16x32bx2 shape specially
    if shape == "16x32bx2":
        if imm_half_splitoff is None:
            raise ValueError("imm_half_splitoff is required for shape 16x32bx2")

        # Extract immediate value - must be a compile-time constant
        try:
            imm_value = simplify_to_int(imm_half_splitoff)
        except Exception as e:
            raise ValueError("imm_half_splitoff must be a compile-time constant for shape 16x32bx2") from e

        # Validate register count
        expected_regs = _get_register_count("16x64b", num)
        if expected_regs is None:
            raise ValueError(f"Unsupported tcgen05.ld combination: 16x32bx2.{num}")

        # Build function name - include immediate value to make it unique
        base_func_name = f'tcgen05_ld_16x32bx2_{num}_imm{imm_value}'
        if pack:
            base_func_name += '_pack_16b'

        full_func_name = f'cuda_{base_func_name}'

        # Register the variant if not already registered
        if not is_primitive_function(full_func_name):
            _register_tcgen05_ld_16x32bx2_variant(num, pack, imm_value)

        # Call the function - note we only pass regs and taddr, not imm_half_splitoff
        return call_primitive_func(full_func_name, [regs, tmem_addr])

    # Handle other shapes (existing code)
    if imm_half_splitoff is not None:
        raise ValueError(f"imm_half_splitoff is only valid for shape 16x32bx2, not {shape}")

    # Validate parameters
    expected_regs = _get_register_count(shape, num)
    if expected_regs is None:
        raise ValueError(f"Unsupported tcgen05.ld combination: {shape}.{num}")

    # Build base and full function names
    base_func_name = f'tcgen05_ld_{shape}_{num}'
    if pack:
        base_func_name += '_pack_16b'

    full_func_name = f'cuda_{base_func_name}'

    # Register the variant if not already registered
    if not is_primitive_function(full_func_name):
        _register_tcgen05_ld_variant(shape, num, pack)

    # Call the function using call_primitive_func with the full name
    return call_primitive_func(full_func_name, [regs, tmem_addr])


def tcgen05_st(
    tmem_addr: Expr,
    regs: Expr,
    shape: Union[str, Tuple[int, int], Tuple[int, int, int]] = "32x32b",
    num: Union[str, int] = "x1",
    unpack: bool = False,
    imm_half_splitoff: Optional[Expr] = None,
):
    """
    Store data from registers to Tensor Memory.

    This implements the tcgen05.st instruction formats:
    - Basic: tcgen05.st.sync.aligned.shape1.num{.unpack}.b32 [taddr], r;
    - 16x32bx2: tcgen05.st.sync.aligned.16x32bx2.num{.unpack}.b32 [taddr], immHalfSplitoff, r;

    This is a warp-wide instruction where all 32 threads in the warp must execute the same
    instruction and are synchronized as a warp.

    See Also:
    ---------
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-st

    Parameters
    ----------
    tmem_addr: Expr
        Base address in Tensor Memory to store to. All threads in the warp must specify the same value.
        This is a 32-bit address where bits 31-16 represent the lane and bits 15-0 represent the column.
    regs: Expr
        Pointer to the source registers containing the data to store. The number of registers accessed
        depends on the given shape and num combination (see Table 48 in PTX docs).
    shape: str
        The shape of the data store. Must be one of: "16x64b", "16x128b", "16x256b", "32x32b", "16x32bx2"
    num: str
        Number of times the shape is repeated. Must be one of: "x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128"
    unpack: bool
        Whether to use unpacking mode where a single 32-bit element in the register is unpacked
        into two 16-bit elements and stored in adjacent columns.
    imm_half_splitoff: Optional[Expr]
        Required for shape "16x32bx2". Offset for the second access (taddr + imm_half_splitoff).

    Returns
    -------
    ret: Call
        The call expression, which will be lowered to a tcgen05.st instruction with the given parameters.
    """

    if isinstance(shape, tuple):
        shape = _shape_to_string(shape)
    if isinstance(num, int):
        num = f"x{num}"

    # Handle .16x32bx2 shape specially
    if shape == "16x32bx2":
        if imm_half_splitoff is None:
            raise ValueError("imm_half_splitoff is required for shape 16x32bx2")

        # Extract immediate value - must be a compile-time constant
        try:
            imm_value = simplify_to_int(imm_half_splitoff)
        except Exception as e:
            raise ValueError("imm_half_splitoff must be a compile-time constant for shape 16x32bx2") from e

        expected_regs = _get_register_count("16x64b", num)
        if expected_regs is None:
            raise ValueError(f"Unsupported tcgen05.st combination: 16x32bx2.{num}")

        # Build function name - include immediate value to make it unique
        base_func_name = f'tcgen05_st_16x32bx2_{num}_imm{imm_value}'
        if unpack:
            base_func_name += '_unpack_16b'

        full_func_name = f'cuda_{base_func_name}'

        # Register the variant if not already registered
        if not is_primitive_function(full_func_name):
            _register_tcgen05_st_16x32bx2_variant(num, unpack, imm_value)

        # Call the function - note we only pass taddr and regs, not imm_half_splitoff
        return call_primitive_func(full_func_name, [tmem_addr, regs])

    if imm_half_splitoff is not None:
        raise ValueError(f"imm_half_splitoff is only valid for shape 16x32bx2, not {shape}")

    expected_regs = _get_register_count(shape, num)
    if expected_regs is None:
        raise ValueError(f"Unsupported tcgen05.st combination: {shape}.{num}")

    # Build base and full function names
    base_func_name = f'tcgen05_st_{shape}_{num}'
    if unpack:
        base_func_name += '_unpack_16b'

    full_func_name = f'cuda_{base_func_name}'

    # Register the variant if not already registered
    if not is_primitive_function(full_func_name):
        _register_tcgen05_st_variant(shape, num, unpack)

    # Call the function using call_primitive_func with the full name
    return call_primitive_func(full_func_name, [tmem_addr, regs])


def tcgen05_wait(wait_type: str = "ld"):
    """
    Wait for completion of prior tcgen05.ld or tcgen05.st operations.

    See Also:
    ---------
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-wait

    Parameters
    ----------
    wait_type: str
        Type of operation to wait for. Either "ld" for load operations or "st" for store operations.

    Returns
    -------
    ret: Call
        The call expression, which will be lowered to a tcgen05.wait for the given wait_type.
    """
    if wait_type not in ["ld", "st"]:
        raise ValueError(f"Invalid wait_type: {wait_type}. Must be either 'ld' or 'st'.")

    # Construct the full function name directly
    # _register_tcgen05_wait registers 'cuda_tcgen05_wait_ld' and 'cuda_tcgen05_wait_st'
    full_func_name = f'cuda_tcgen05_wait_{wait_type}'

    # The registration is handled by @initialize() on register_tcgen05_wait()
    return call_primitive_func(full_func_name, [])


# Support for .16x32bx2 shape variants (Section 9.7.16.8.3) is implemented above.
# These variants use a different format with immHalfSplitoff parameter:
# - tcgen05.ld.sync.aligned.16x32bx2.num{.pack}.b32 r, [taddr], immHalfSplitoff;
# - tcgen05.st.sync.aligned.16x32bx2.num{.unpack}.b32 [taddr], immHalfSplitoff, r;
# The .16x32bx2 shape performs two accesses of .16x32b each, with the second access
# at address taddr + immHalfSplitoff.


def _register_tcgen05_ld_16x32bx2_variant(num: str, pack: bool, imm_value: int):
    """
    Register a tcgen05.ld instruction variant for .16x32bx2 shape.

    The .16x32bx2 shape is special: it performs two separate accesses to Tensor Memory,
    each of shape .16x32b. The first access uses the base address (taddr), and the
    second access uses taddr + immHalfSplitoff.

    Supports the format:
    tcgen05.ld.sync.aligned.16x32bx2.num{.pack}.b32 r, [taddr], immHalfSplitoff;

    Note: immHalfSplitoff must be an immediate value (constant) in PTX, not a register.
    """
    from hidet.lang import script, attrs
    from hidet.ir.dtypes import u32
    from hidet.ir.type import PointerType
    from hidet.ir.expr import cast

    # .16x32bx2 has same register count as .16x64b according to Table 48
    # This makes sense as 2 x 16x32b = 16x64b in terms of data volume
    reg_count = _get_register_count("16x64b", num)  # Using 16x64b mapping
    if reg_count is None:
        raise ValueError(f"Unsupported tcgen05.ld combination: 16x32bx2.{num}")

    # Build function name - include the immediate value in the name to make it unique
    func_name = f'cuda_tcgen05_ld_16x32bx2_{num}_imm{imm_value}'
    if pack:
        func_name += '_pack_16b'

    # Build template string
    template_parts = ['tcgen05.ld.sync.aligned', '16x32bx2', num]
    if pack:
        template_parts.append('pack::16b')
    template_parts.append('b32')

    # Create register list part of template
    reg_list = ', '.join([f'%{i}' for i in range(reg_count)])
    # Note: immHalfSplitoff is embedded directly as an immediate value
    template_string = f'{".".join(template_parts)} {{{reg_list}}}, [%{reg_count}], {imm_value};'

    @script
    def cuda_tcgen05_ld_16x32bx2(regs: ~u32, taddr: u32):  # type: ignore
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'

        # Cast the pointer to access as array
        regs_ptr = cast(regs, PointerType("uint32"))

        # Ensure taddr is in a register by creating local variable
        taddr_reg: u32 = taddr  # type: ignore

        asm(
            template=template_string,
            outputs=[regs_ptr[i] for i in range(reg_count)],
            inputs=[taddr_reg],
            is_volatile=True,
        )

    register_primitive_function(name=func_name, func_or_type=cuda_tcgen05_ld_16x32bx2)


def _register_tcgen05_st_16x32bx2_variant(num: str, unpack: bool, imm_value: int):
    """
    Register a tcgen05.st instruction variant for .16x32bx2 shape.

    The .16x32bx2 shape is special: it performs two separate accesses to Tensor Memory,
    each of shape .16x32b. The first access uses the base address (taddr), and the
    second access uses taddr + immHalfSplitoff.

    Supports the format:
    tcgen05.st.sync.aligned.16x32bx2.num{.unpack}.b32 [taddr], immHalfSplitoff, r;

    Note: immHalfSplitoff must be an immediate value (constant) in PTX, not a register.
    """
    from hidet.lang import script, attrs
    from hidet.ir.dtypes import u32
    from hidet.ir.type import PointerType
    from hidet.ir.expr import cast

    # .16x32bx2 has same register count as .16x64b according to Table 48
    # This makes sense as 2 x 16x32b = 16x64b in terms of data volume
    reg_count = _get_register_count("16x64b", num)  # Using 16x64b mapping
    if reg_count is None:
        raise ValueError(f"Unsupported tcgen05.st combination: 16x32bx2.{num}")

    # Build function name - include the immediate value in the name to make it unique
    func_name = f'cuda_tcgen05_st_16x32bx2_{num}_imm{imm_value}'
    if unpack:
        func_name += '_unpack_16b'

    # Build template string
    template_parts = ['tcgen05.st.sync.aligned', '16x32bx2', num]
    if unpack:
        template_parts.append('unpack::16b')
    template_parts.append('b32')

    # Create register list part of template
    # Note the order: [taddr], immHalfSplitoff, {registers}
    # taddr is %0, registers start at %1 (immHalfSplitoff is embedded as immediate)
    reg_list = ', '.join([f'%{i + 1}' for i in range(reg_count)])
    template_string = f'{".".join(template_parts)} [%0], {imm_value}, {{{reg_list}}};'

    @script
    def cuda_tcgen05_st_16x32bx2(taddr: u32, regs: ~u32):  # type: ignore
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'

        # Cast the pointer to access as array
        regs_ptr = cast(regs, PointerType("uint32"))

        # Ensure taddr is in a register by creating local variable
        taddr_reg: u32 = taddr  # type: ignore

        asm(template=template_string, inputs=[taddr_reg] + [regs_ptr[i] for i in range(reg_count)], is_volatile=True)

    register_primitive_function(name=func_name, func_or_type=cuda_tcgen05_st_16x32bx2)


# TODO: Implement support for reduction operations (Section 9.7.16.8.3)
# tcgen05.ld.red.sync.aligned.shape.num.redOp{.abs}{.NaN}.f32 r, redval, [taddr];
# tcgen05.ld.red.sync.aligned.shape.num.redOp.type r, redval, [taddr];
# where redOp = { .min, .max } and type = { .u32, .s32 }
# This is postponed for the time being because it is not a priority.
