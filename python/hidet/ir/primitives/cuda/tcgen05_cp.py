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
from typing import Optional, Tuple, Union, Literal

from hidet.ir.stmt import asm
from hidet.lang import script, attrs
from hidet.ir.expr import Expr
from hidet.ir.dtypes import u32, u64
from hidet.ir.primitives import is_primitive_function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func


def _register_tcgen05_cp_variant(
    shape: str,
    cta_group: int,
    multicast: Optional[str] = None,
    dst_fmt: Optional[str] = None,
    src_fmt: Optional[str] = None,
):
    """
    Dynamically register a tcgen05.cp instruction variant.

    Supports the format:
    tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;
    """
    # Build function name
    func_name = f'cuda_tcgen05_cp_cta_group_{cta_group}_{shape}'
    if multicast:
        func_name += f'_{multicast.replace("::", "_")}'
    if dst_fmt and src_fmt:
        func_name += f'_{dst_fmt}_{src_fmt}'

    # Build template string
    template_parts = ['tcgen05.cp', f'cta_group::{cta_group}', shape]
    if multicast:
        template_parts.append(multicast)
    if dst_fmt and src_fmt:
        template_parts.extend([dst_fmt, src_fmt])

    template_string = f'{".".join(template_parts)} [%0], %1;'

    @script
    def cuda_tcgen05_cp(taddr: u32, s_desc: u64):  # type: ignore
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'

        # Ensure values are in registers
        taddr_reg: u32 = taddr  # type: ignore
        s_desc_reg: u64 = s_desc  # type: ignore

        asm(template=template_string, inputs=[taddr_reg, s_desc_reg], is_volatile=True)

    register_primitive_function(name=func_name, func_or_type=cuda_tcgen05_cp)


def _register_tcgen05_shift_variant(cta_group: int):
    """
    Dynamically register a tcgen05.shift instruction variant.

    Supports the format:
    tcgen05.shift.cta_group.down [taddr];
    """

    # Build function name
    func_name = f'cuda_tcgen05_shift_cta_group_{cta_group}_down'

    # Build template string
    template_string = f'tcgen05.shift.cta_group::{cta_group}.down [%0];'

    @script
    def cuda_tcgen05_shift(taddr: u32):  # type: ignore
        attrs.func_name = func_name
        attrs.func_kind = 'cuda_internal'

        # Ensure value is in register
        taddr_reg: u32 = taddr  # type: ignore

        asm(template=template_string, inputs=[taddr_reg], is_volatile=True)

    register_primitive_function(name=func_name, func_or_type=cuda_tcgen05_shift)


def _tuple_params_to_str(params: Tuple[int, ...]) -> str:
    """
    Helper function to convert a tuple of integers to a string.
    This function is potentially used to convert the values of the parameter `shape`
    in the `tcgen05_cp` function to a string.
    """
    return f"{params[0]}x{params[1]}b"


def tcgen05_cp(
    tmem_addr: Expr,
    shared_desc: Expr,
    shape: Union[Tuple[int, int], Literal["128x256b", "4x256b", "128x128b", "64x128b", "32x128b"]],
    cta_group: Literal[1, 2] = 1,
    multicast: Optional[Literal["warpx2::02_13", "warpx2::01_23", "warpx4"]] = None,
    dst_fmt: Optional[Literal["b8x16"]] = None,
    src_fmt: Optional[Literal["b6x16_p32", "b4x16_p64"]] = None,
) -> Expr:
    """
    Copy data from shared memory to Tensor Memory asynchronously.

    This implements the tcgen05.cp instruction which initiates an asynchronous copy
    operation from shared memory to Tensor Memory.

    See Also:
    ---------
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-movement-instructions

    Parameters
    ----------
    tmem_addr: Expr
        Address in Tensor Memory where data will be copied to.
    shared_desc: Expr
        64-bit matrix descriptor representing the source matrix in shared memory.
        The format is described in Section 9.7.16.4.1 (Shared memory descriptor).
    shape: str
        The shape of data to be copied. Must be one of:
        - "128x256b": 128 lanes x 256 bits
        - "4x256b": 4 lanes x 256 bits
        - "128x128b": 128 lanes x 128 bits
        - "64x128b": 64 lanes x 128 bits (requires multicast)
        - "32x128b": 32 lanes x 128 bits (requires warpx4 multicast)
    cta_group: int
        Number of CTAs whose Tensor Memory is accessed:
        - 1: Data is copied into the Tensor Memory of the current CTA only
        - 2: Data is copied into the Tensor Memory of both current and peer CTAs
    multicast: Optional[str]
        Multicast qualifier for certain shapes:
        - "warpx2::02_13" or "warpx2::01_23": Required for shape "64x128b"
        - "warpx4": Required for shape "32x128b"
    dst_fmt: Optional[str]
        Destination format for decompression. Currently only "b8x16" is supported.
    src_fmt: Optional[str]
        Source format for decompression:
        - "b6x16_p32": 6-bit elements with 32-bit padding
        - "b4x16_p64": 4-bit elements with 64-bit padding

    Returns
    -------
    ret: Call
        The call expression for the tcgen05.cp instruction.

    Examples
    --------
    Basic copy without decompression:
    >>> tcgen05_cp(tmem_addr, shared_desc, shape="128x256b", cta_group=1)

    Copy with multicast to warp pairs:
    >>> tcgen05_cp(tmem_addr, shared_desc, shape="64x128b", cta_group=1, multicast="warpx2::02_13")

    Copy with decompression from 6-bit to 8-bit:
    >>> tcgen05_cp(tmem_addr, shared_desc, shape="128x128b", cta_group=2,
    ...            dst_fmt="b8x16", src_fmt="b6x16_p32")
    """

    if isinstance(shape, tuple):
        shape = _tuple_params_to_str(shape)

    allowed_shapes = ("128x256b", "4x256b", "128x128b", "64x128b", "32x128b")

    if shape not in allowed_shapes:
        raise ValueError(f"Invalid shape: {shape}")

    # Validate shape and multicast requirements
    if shape == "64x128b" and multicast not in ["warpx2::02_13", "warpx2::01_23"]:
        raise ValueError("Shape '64x128b' requires multicast to be 'warpx2::02_13' or 'warpx2::01_23'")
    if shape == "32x128b" and multicast != "warpx4":
        raise ValueError("Shape '32x128b' requires multicast to be 'warpx4'")
    if shape not in ["64x128b", "32x128b"] and multicast is not None:
        raise ValueError("Multicast is only valid for shapes '64x128b' and '32x128b'")

    # Validate decompression parameters
    if (dst_fmt is None) != (src_fmt is None):
        raise ValueError("Both dst_fmt and src_fmt must be specified together for decompression")

    # Validate cta_group
    if cta_group not in [1, 2]:
        raise ValueError(f"cta_group must be 1 or 2, got {cta_group}")

    # Build function name
    base_func_name = f'tcgen05_cp_cta_group_{cta_group}_{shape}'
    if multicast:
        base_func_name += f'_{multicast.replace("::", "_")}'
    if dst_fmt and src_fmt:
        base_func_name += f'_{dst_fmt}_{src_fmt}'

    full_func_name = f'cuda_{base_func_name}'

    # Register the variant if not already registered
    if not is_primitive_function(full_func_name):
        _register_tcgen05_cp_variant(shape, cta_group, multicast, dst_fmt, src_fmt)

    # Call the function
    return call_primitive_func(full_func_name, [tmem_addr, shared_desc])


def tcgen05_shift(tmem_addr: Expr, cta_group: Literal[1, 2] = 1) -> Expr:
    """
    Asynchronously shift down the rows of the matrix in Tensor Memory.

    This implements the tcgen05.shift instruction which initiates shifting of 32-byte
    elements downwards across all rows (except the last) by one row.

    See Also:
    ---------
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-shift

    Parameters
    ----------
    tmem_addr: Expr
        Base address of the matrix in Tensor Memory whose rows must be shifted down.
        The lane of the address must be aligned to 32.
    cta_group: int
        Number of CTAs whose Tensor Memory is touched:
        - 1: Shift operation is performed in the Tensor Memory of the current CTA only
        - 2: Shift operation is performed in the Tensor Memory of both current and peer CTAs

    Returns
    -------
    ret: Call
        The call expression for the tcgen05.shift instruction.

    Examples
    --------
    Shift rows in current CTA's tensor memory:
    >>> tcgen05_shift(tmem_addr, cta_group=1)

    Shift rows in both current and peer CTA's tensor memory:
    >>> tcgen05_shift(tmem_addr, cta_group=2)
    """
    # Validate cta_group
    if cta_group not in [1, 2]:
        raise ValueError(f"cta_group must be 1 or 2, got {cta_group}")

    # Build function name
    base_func_name = f'tcgen05_shift_cta_group_{cta_group}_down'
    full_func_name = f'cuda_{base_func_name}'

    # Register the variant if not already registered
    if not is_primitive_function(full_func_name):
        _register_tcgen05_shift_variant(cta_group)

    # Call the function
    return call_primitive_func(full_func_name, [tmem_addr])


# Helper function to encode matrix descriptor
def matrix_descriptor_encode(addr: u64) -> Expr:  # type: ignore
    """
    Encode a matrix address according to PTX requirements.

    matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4

    This is primarily used as a helper function to `make_tcgen05_cp_desc`.
    """
    from hidet.ir.expr import constant

    if isinstance(addr, int):
        return constant((addr & 0x3FFFF) >> 4, u64)
    else:
        return (addr & 0x3FFFF) >> 4


def make_tcgen05_cp_desc(
    leading_dim_offset: Union[int, Expr],
    stride_dim_offset: Union[int, Expr],
    swizzle_mode: int = 0,
    base_offset: int = 0,
    use_absolute_addressing: bool = False,
) -> Expr:
    """
    Create a shared memory descriptor template for tcgen05.cp operation.

    This creates the descriptor without the matrix start address, which must be
    added at runtime.

    To complete the descriptor, you need to:
    1. Get the shared memory address using cvta_generic_to_shared()
    2. Encode it: matrix_start_encoded = ((smem_addr & 0x3FFFF) >> 4)
    3. Add it to the descriptor: desc = desc_template | matrix_start_encoded

    Note: The base_offset (bits 49-51) is already set in the template and should
    NOT be overwritten with address bits.

    See Also:
    ---------
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-descriptor

    Parameters
    ----------
    leading_dim_offset: Union[int, Expr]
        Leading dimension byte offset (must be 16-byte aligned).
    stride_dim_offset: Union[int, Expr]
        Stride dimension byte offset (must be 16-byte aligned).
    swizzle_mode: int
        Swizzling mode:
        - 0: No swizzling
        - 1: 128-Byte with 32B atomic swizzling
        - 2: 128-Byte swizzling
        - 4: 64-Byte swizzling
        - 6: 32-Byte swizzling
    base_offset: int
        Matrix base offset (0-7).
    use_absolute_addressing: bool
        If True, use absolute addressing mode for leading dimension stride.
        See Section 9.7.16.3.1 of the PTX ISA for more details.

    Returns
    -------
    descriptor: Expr
        64-bit shared memory descriptor template (without matrix address).
    """
    from hidet.ir.expr import constant

    # Validate inputs
    if isinstance(leading_dim_offset, int) and leading_dim_offset % 16 != 0:
        raise ValueError("leading_dim_offset must be 16-byte aligned")
    if isinstance(stride_dim_offset, int) and stride_dim_offset % 16 != 0:
        raise ValueError("stride_dim_offset must be 16-byte aligned")

    allowed_swizzle_modes = (0, 1, 2, 4, 6)
    if swizzle_mode not in allowed_swizzle_modes:
        raise ValueError(f"Invalid swizzle_mode: {swizzle_mode}")
    if not 0 <= base_offset <= 7:
        raise ValueError(f"base_offset must be 0-7, got {base_offset}")

    # Build descriptor according to Section 9.7.16.4.1
    desc: u64 = 0  # type: ignore

    # Bits 16-29: leading dimension offset (encoded)
    desc = desc | (matrix_descriptor_encode(leading_dim_offset) << 16)

    # Bits 32-45: stride dimension offset (encoded)
    desc = desc | (matrix_descriptor_encode(stride_dim_offset) << 32)

    # Bits 46-48: Fixed constant 0b001
    desc = desc | (constant(0b001, u64) << 46)

    # Bits 49-51: Matrix base offset
    desc = desc | (constant(base_offset, u64) << 49)

    # Bit 52: Leading dimension stride mode (0: relative, 1: absolute)
    desc = desc | (constant(1 if use_absolute_addressing else 0, u64) << 52)

    # Bits 53-60: Fixed constant 0xb0
    desc = desc | (constant(0xB0, u64) << 53)

    # Bits 61-63: Swizzling mode
    desc = desc | (constant(swizzle_mode, u64) << 61)

    return desc
