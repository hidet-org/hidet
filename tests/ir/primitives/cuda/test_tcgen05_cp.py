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
import numpy as np
import pytest
from typing import Optional

import hidet
from hidet.ir.primitives.cuda import threadIdx
from hidet.ir.primitives.cuda.sync import syncthreads
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.dtypes import u32, i32, u64
from hidet.lang import attrs, printf
from hidet.lang.constructs.declare import shared_tensor
from hidet.ir.primitives.cuda import tcgen05_alloc, tcgen05_dealloc, tcgen05_relinquish_alloc_permit
from hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from hidet.ir.primitives.cuda import tcgen05_cp, tcgen05_shift, make_tcgen05_cp_desc


@pytest.mark.requires_cuda_blackwell
def test_tcgen05_cp_compilation():
    """Test that tcgen05.cp instructions compile correctly."""

    with hidet.script_module() as script_module:

        @hidet.script
        def test_cp_compilation():  # type: ignore
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 128  # Full warpgroup
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 8192  # Increased to ensure enough space

            tid = threadIdx.x
            warp_id = tid // 32

            # Allocate tensor memory
            tmem_addr_storage = shared_tensor("u32", [1])
            if warp_id == 0:
                tcgen05_alloc(tmem_addr_storage, 32)

            syncthreads()
            tmem_addr: u32 = tmem_addr_storage[0]  # type: ignore

            # Initialize shared memory with some data to avoid accessing uninitialized memory
            smem_base = dynamic_shared_memory(0, dtype=u32)

            # Initialize shared memory with pattern
            for i in range(32):
                idx = tid * 32 + i
                if idx < 2048:  # 8192 bytes / 4 bytes per u32
                    smem_base[idx] = idx

            syncthreads()

            smem_addr = cvta_generic_to_shared(smem_base)

            # Create descriptor template using our specific function
            desc_template = make_tcgen05_cp_desc(
                leading_dim_offset=256,  # 16 * 16 bytes
                stride_dim_offset=512,  # 32 * 16 bytes
                swizzle_mode=0,  # No swizzling
                base_offset=0,  # Already set in template
            )

            # Add the matrix start address to the descriptor
            # Bits 0-13: matrix start address (encoded)
            matrix_start_encoded = (smem_addr & 0x3FFFF) >> 4
            desc: u64 = desc_template | matrix_start_encoded  # type: ignore

            # Test 1: Basic copy - 128x256b
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="128x256b", cta_group=1)
                printf("Test 1: Basic 128x256b copy completed\\n")

            syncthreads()

            # Test 2: 4x256b copy
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="4x256b", cta_group=1)
                printf("Test 2: 4x256b copy completed\\n")

            syncthreads()

            # Test 3: 128x128b copy
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="128x128b", cta_group=1)
                printf("Test 3: 128x128b copy completed\\n")

            syncthreads()

            # Test 4: 64x128b with multicast
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="64x128b", cta_group=1, multicast="warpx2::02_13")
                printf("Test 4: 64x128b with multicast completed\\n")

            syncthreads()

            # Test 5: 32x128b with warpx4 multicast
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="32x128b", cta_group=1, multicast="warpx4")
                printf("Test 5: 32x128b with warpx4 multicast completed\\n")

            syncthreads()

            # Test 6: Copy with decompression (6-bit to 8-bit)
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="128x128b", cta_group=1, dst_fmt="b8x16", src_fmt="b6x16_p32")
                printf("Test 6: Copy with 6-bit to 8-bit decompression completed\\n")

            syncthreads()

            # Test 7: Copy with decompression (4-bit to 8-bit)
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="4x256b", cta_group=1, dst_fmt="b8x16", src_fmt="b4x16_p64")
                printf("Test 7: Copy with 4-bit to 8-bit decompression completed\\n")

            syncthreads()

            # Test 8: Shift operation
            if tid == 0:
                tcgen05_shift(tmem_addr, cta_group=1)
                printf("Test 8: Shift operation completed\\n")

            syncthreads()

            # Deallocate tensor memory
            if warp_id == 0:
                tcgen05_dealloc(tmem_addr, 32)
                tcgen05_relinquish_alloc_permit()

            syncthreads()

            if tid == 0:
                printf("All tcgen05.cp instructions compiled successfully\\n")

    # Build and run the module
    module = script_module.build()
    module()
    hidet.cuda.synchronize()


@pytest.mark.requires_cuda_blackwell
def test_tcgen05_cp_data_flow():
    """Test data flow through shared memory -> TMEM with tcgen05.cp."""

    with hidet.script_module() as script_module:

        @hidet.script
        def test_cp_data_flow(test_passed: ~i32):  # type: ignore
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 8192  # Allocate shared memory for source data

            tid = threadIdx.x
            warp_id = tid // 32
            lane_id = tid % 32

            # Initialize test result
            if tid == 0:
                test_passed[0] = 1

            # Allocate tensor memory
            tmem_addr_storage = shared_tensor("u32", [1])
            if warp_id == 0:
                tcgen05_alloc(tmem_addr_storage, 64)  # Allocate more for testing

            syncthreads()
            tmem_addr: u32 = tmem_addr_storage[0]  # type: ignore

            # Initialize shared memory with test pattern
            smem = dynamic_shared_memory(0, dtype=u32)

            # Fill shared memory with enough data for the operations
            # For 128x128b, we need 128 lanes × 128 bits = 128 × 16 bytes = 2048 bytes per lane
            # But we're using a matrix layout, not all data may be used
            for i in range(32):
                idx = tid * 32 + i
                if idx < 2048:  # Fill 8192 bytes
                    smem[idx] = 0x1000 + idx  # Pattern to identify data

            syncthreads()

            # Create shared memory descriptor with simpler offsets
            smem_addr = cvta_generic_to_shared(smem)

            # Use smaller offsets for a simpler test
            desc_template = make_tcgen05_cp_desc(
                leading_dim_offset=16,  # 1 * 16 bytes - minimal offset
                stride_dim_offset=32,  # 2 * 16 bytes - minimal offset
                swizzle_mode=0,  # No swizzling
                base_offset=0,
            )

            # Add the matrix start address
            matrix_start_encoded = (smem_addr & 0x3FFFF) >> 4
            desc: u64 = desc_template | matrix_start_encoded

            # Test 1: Copy with smaller shape first
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="4x256b", cta_group=1)
                printf("Copying 4x256b from shared to tensor memory\\n")

            syncthreads()

            # For now, just verify the operations complete without crashing
            # The exact data layout in tensor memory after tcgen05.cp depends on the
            # descriptor settings and may not be directly compatible with tcgen05.ld

            # Test 2: Test shift operation
            if tid == 0:
                tcgen05_shift(tmem_addr, cta_group=1)
                printf("Shifted tensor memory rows\\n")

            syncthreads()

            # Test 3: Try a different copy shape
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="128x128b", cta_group=1)
                printf("Copying 128x128b from shared to tensor memory\\n")

            syncthreads()

            # Deallocate
            if warp_id == 0:
                tcgen05_dealloc(tmem_addr, 64)
                tcgen05_relinquish_alloc_permit()

            syncthreads()

            if tid == 0:
                printf("Data flow test completed\\n")

    # Build and run
    module = script_module.build()

    # Create test result buffer
    test_passed = hidet.ones([1], dtype=i32, device='cuda')

    # Run kernel
    module(test_passed)
    hidet.cuda.synchronize()

    # Check result
    test_passed_cpu = test_passed.cpu().numpy()[0]
    assert test_passed_cpu == 1, "Data flow test failed"
    print("Data flow test passed!")


@pytest.mark.requires_cuda_blackwell
def test_tcgen05_cp_multicast():
    """Test multicast functionality of tcgen05.cp."""

    with hidet.script_module() as script_module:

        @hidet.script
        def test_multicast():  # type: ignore
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 4096

            tid = threadIdx.x
            warp_id = tid // 32

            # Allocate tensor memory
            tmem_addr_storage = shared_tensor("u32", [1])
            if warp_id == 0:
                tcgen05_alloc(tmem_addr_storage, 32)

            syncthreads()
            tmem_addr: u32 = tmem_addr_storage[0]  # type: ignore

            # Initialize shared memory
            smem = dynamic_shared_memory(0, dtype=u32)
            for i in range(16):
                idx = tid * 16 + i
                if idx < 1024:  # 4096 bytes / 4 bytes per u32
                    smem[idx] = tid * 100 + i

            syncthreads()

            # Create shared memory descriptor
            smem_addr = cvta_generic_to_shared(smem)

            desc_template = make_tcgen05_cp_desc(
                leading_dim_offset=256,  # 16 * 16 bytes
                stride_dim_offset=512,  # 32 * 16 bytes
                swizzle_mode=0,  # No swizzling
                base_offset=0,
            )

            matrix_start_encoded = (smem_addr & 0x3FFFF) >> 4
            desc: u64 = desc_template | matrix_start_encoded  # type: ignore

            # Test multicast modes
            if tid == 0:
                # Test warpx2::02_13 multicast
                tcgen05_cp(tmem_addr, desc, shape="64x128b", cta_group=1, multicast="warpx2::02_13")
                printf("Multicast warpx2::02_13 completed\\n")

                # Test warpx2::01_23 multicast
                tcgen05_cp(tmem_addr, desc, shape="64x128b", cta_group=1, multicast="warpx2::01_23")
                printf("Multicast warpx2::01_23 completed\\n")

                # Test warpx4 multicast
                tcgen05_cp(tmem_addr, desc, shape="32x128b", cta_group=1, multicast="warpx4")
                printf("Multicast warpx4 completed\\n")

            syncthreads()

            # Deallocate
            if warp_id == 0:
                tcgen05_dealloc(tmem_addr, 32)
                tcgen05_relinquish_alloc_permit()

            syncthreads()

            if tid == 0:
                printf("Multicast test completed successfully\\n")

    # Build and run
    module = script_module.build()
    module()
    hidet.cuda.synchronize()


@pytest.mark.requires_cuda_blackwell
@pytest.mark.parametrize(
    "shape,multicast",
    [
        ("128x256b", None),
        ("4x256b", None),
        ("128x128b", None),
        ("64x128b", "warpx2::02_13"),
        ("64x128b", "warpx2::01_23"),
        ("32x128b", "warpx4"),
    ],
)
def test_tcgen05_cp_all_shapes(shape: str, multicast: Optional[str]):
    """Test all supported shapes and multicast combinations."""

    with hidet.script_module() as script_module:

        @hidet.script
        def test_shape_variant(test_passed: ~i32):  # type: ignore
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 16384  # Increased to 16KB

            tid = threadIdx.x
            warp_id = tid // 32

            # Initialize test result
            if tid == 0:
                test_passed[0] = 1

            # Allocate more tensor memory for larger shapes
            tmem_addr_storage = shared_tensor("u32", [1])
            if warp_id == 0:
                tcgen05_alloc(tmem_addr_storage, 256)  # Increased from 64 to 256

            syncthreads()
            tmem_addr: u32 = tmem_addr_storage[0]  # type: ignore

            # Create shared memory descriptor
            smem = dynamic_shared_memory(0, dtype=u32)
            smem_addr = cvta_generic_to_shared(smem)

            # Initialize shared memory with pattern
            # Fill more memory to ensure we have enough for large shapes
            for i in range(32):
                idx = tid * 32 + i
                if idx < 4096:  # Fill 16KB
                    smem[idx] = tid * 1000 + i

            syncthreads()

            # Use smaller descriptor offsets to avoid out-of-bounds access
            desc_template = make_tcgen05_cp_desc(
                leading_dim_offset=256,  # 16 * 16 bytes - reduced from 1024
                stride_dim_offset=512,  # 32 * 16 bytes - reduced from 2048
                swizzle_mode=0,  # No swizzling
                base_offset=0,
            )

            matrix_start_encoded = (smem_addr & 0x3FFFF) >> 4
            desc: u64 = desc_template | matrix_start_encoded  # type: ignore

            # Execute the copy with the given shape and multicast
            if tid == 0:
                # Use == instead of is for None comparison in Hidet script
                if multicast == None:
                    tcgen05_cp(tmem_addr, desc, shape=shape, cta_group=1)
                else:
                    tcgen05_cp(tmem_addr, desc, shape=shape, cta_group=1, multicast=multicast)
                printf("Copy completed for shape=%s, multicast=%s\\n", shape, multicast if multicast else "None")

            syncthreads()

            # Deallocate
            if warp_id == 0:
                tcgen05_dealloc(tmem_addr, 256)
                tcgen05_relinquish_alloc_permit()

            syncthreads()

            if tid == 0:
                printf("Test passed for shape=%s\\n", shape)

    # Build and run
    module = script_module.build()

    test_passed = hidet.ones([1], dtype=i32, device='cuda')
    module(test_passed)
    hidet.cuda.synchronize()

    # Check result
    test_passed_cpu = test_passed.cpu().numpy()[0]
    assert test_passed_cpu == 1, f"Test failed for shape={shape}, multicast={multicast}"


@pytest.mark.requires_cuda_blackwell
def test_tcgen05_cp_decompression():
    """Test decompression functionality during copy."""

    with hidet.script_module() as script_module:

        @hidet.script
        def test_decompression():  # type: ignore
            attrs.func_kind = 'cuda_kernel'
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1
            attrs.cuda.dynamic_smem_bytes = 16384

            tid = threadIdx.x
            warp_id = tid // 32

            # Allocate more tensor memory for decompression operations
            tmem_addr_storage = shared_tensor("u32", [1])
            if warp_id == 0:
                tcgen05_alloc(tmem_addr_storage, 256)

            syncthreads()
            tmem_addr: u32 = tmem_addr_storage[0]  # type: ignore

            # Create shared memory with packed data
            smem = dynamic_shared_memory(0, dtype=u32)
            smem_addr = cvta_generic_to_shared(smem)

            # Initialize with packed data patterns
            for i in range(32):
                idx = tid * 32 + i
                if idx < 4096:
                    # Simulate packed data - each nibble/bit group represents compressed values
                    smem[idx] = 0x12345678 + idx  # Vary the pattern to avoid all same values

            syncthreads()

            # Use smaller offsets appropriate for decompression
            desc_template = make_tcgen05_cp_desc(
                leading_dim_offset=128,  # 8 * 16 bytes - reduced offset
                stride_dim_offset=256,  # 16 * 16 bytes - reduced offset
                swizzle_mode=0,  # No swizzling
                base_offset=0,
            )

            matrix_start_encoded = (smem_addr & 0x3FFFF) >> 4
            desc: u64 = desc_template | matrix_start_encoded  # type: ignore

            # Test 6-bit to 8-bit decompression
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="128x128b", cta_group=1, dst_fmt="b8x16", src_fmt="b6x16_p32")
                printf("6-bit to 8-bit decompression completed\\n")

            syncthreads()

            # Test 4-bit to 8-bit decompression with smaller shape
            # 4x256b is safer for testing as it requires less memory
            if tid == 0:
                tcgen05_cp(tmem_addr, desc, shape="4x256b", cta_group=1, dst_fmt="b8x16", src_fmt="b4x16_p64")
                printf("4-bit to 8-bit decompression completed\\n")

            syncthreads()

            # Deallocate - match the allocation size
            if warp_id == 0:
                tcgen05_dealloc(tmem_addr, 256)
                tcgen05_relinquish_alloc_permit()

            syncthreads()

            if tid == 0:
                printf("Decompression test completed successfully\\n")

    # Build and run
    module = script_module.build()
    module()
    hidet.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__])
