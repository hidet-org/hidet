cmake_minimum_required(VERSION 3.19)

project(hidet_cuda_experiments C CXX CUDA)

# e3
add_executable(e3_mma
        e3_mma/main.cpp
        e3_mma/wmma_m16n16k16_fp16.cu
        e3_mma/mma_m8n8k4_fp16.cu
)
set_target_properties(e3_mma PROPERTIES
        CUDA_ARCHITECTURES 86
        )

# e4
add_executable(e4_play
        e4_play/main.cu
        )
set_target_properties(e4_play PROPERTIES
        CUDA_ARCHITECTURES 86
        )

# e5
add_executable(e5_cutlass
        e5_cutlass/main.cu
        )
target_include_directories(e5_cutlass
        PUBLIC
        ../../3rdparty/cutlass/include
        ../../3rdparty/cutlass/tools/util/include
        )
set_target_properties(e5_cutlass PROPERTIES
        CUDA_ARCHITECTURES 86
        )

# e6
add_executable(e6_cutlass_tf32
        e6_cutlass_tf32/main.cu
        e6_cutlass_tf32/gemm.cu
        )
target_include_directories(e6_cutlass_tf32
        PUBLIC
        ../../3rdparty/cutlass/include
        ../../3rdparty/cutlass/tools/util/include
        )
set_target_properties(e6_cutlass_tf32 PROPERTIES
        CUDA_ARCHITECTURES 86
        )
# add -lineinfo option to nvcc, allowing us to get the source code from binary
# do not influence optimization, can be used in nsight compute profiling
target_compile_options(e6_cutlass_tf32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# e7
add_executable(e7_mma
        e7_mma/main.cu
        )
set_target_properties(e7_mma PROPERTIES
        CUDA_ARCHITECTURES 86
        )
# add -lineinfo option to nvcc, allowing us to get the source code from binary
# do not influence optimization, can be used in nsight compute profiling
target_compile_options(e7_mma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# e8
add_executable(e8_static_gmem
        e8_static_gmem/main.cu
        )
set_target_properties(e8_static_gmem PROPERTIES
        CUDA_ARCHITECTURES 86
        )
target_compile_options(e8_static_gmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)


