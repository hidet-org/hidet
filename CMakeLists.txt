cmake_minimum_required(VERSION 3.19)

project(hidet C CXX CUDA)

# common configs
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)

# submodules
include(cmake/TVM.cmake)

# config hidet
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
    include(${CMAKE_BINARY_DIR}/config.cmake)
else()
    include(${CMAKE_SOURCE_DIR}/config.cmake)
endif()

set(CMAKE_BUILD_TYPE ${HIDET_BUILD_TYPE})
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_library(hidet SHARED
        src/hidet/packedfunc.cpp
        src/hidet/runtime.cpp
        # matmul baselines
        src/hidet/baselines/matmul/reference.cu
        src/hidet/baselines/matmul/cublas.cpp
        src/hidet/baselines/matmul/cutlass.cu
        src/hidet/baselines/matmul/opt.cu
        # conv2d baselines
        src/hidet/baselines/conv2d/cudnn.cpp
        src/hidet/baselines/conv2d/direct.cu
        src/hidet/baselines/conv2d/reference.cu
        src/hidet/baselines/conv2d/implicit_gemm_ref.cu
        )

target_include_directories(hidet PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
        )

target_link_directories(hidet PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(NOT HIDET_CUDNN_PATH STREQUAL "Auto")
    target_include_directories(hidet PRIVATE ${HIDET_CUDNN_PATH}/include)
    target_link_directories(hidet PRIVATE ${HIDET_CUDNN_PATH}/lib64)
    message(STATUS "Use cuDNN in: ${HIDET_CUDNN_PATH}")
endif()

target_link_libraries(hidet cudart cublas cudnn)

set_target_properties(hidet PROPERTIES
        CUDA_RUNTIME_LIBRARY SHARED
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        CUDA_ARCHITECTURES "${HIDET_CUDA_ARCH}"
        )

# add -lineinfo option to nvcc, allowing us to get the source code from binary
# do not influence optimization, can be used in nsight compute profiling
target_compile_options(hidet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

