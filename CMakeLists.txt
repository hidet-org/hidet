cmake_minimum_required(VERSION 3.19)

project(hidet C CXX CUDA)

set(HIDET_SRCS
        src/hidet/packedfunc.cpp
        src/hidet/runtime.cpp
        src/hidet/baselines/matmul/reference.cu
        src/hidet/baselines/matmul/cublas.cu
        src/hidet/baselines/matmul/cutlass.cu
        src/hidet/baselines/matmul/opt.cu
        )

add_library(hidet SHARED ${HIDET_SRCS})

target_include_directories(hidet PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
        )

target_link_directories(hidet
        PRIVATE
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries(hidet cudart cublas)

set_target_properties(hidet PROPERTIES
        CUDA_RUNTIME_LIBRARY SHARED
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        CUDA_ARCHITECTURES "75"
        )

