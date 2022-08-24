cmake_minimum_required(VERSION 3.19)

project(hidet C CXX CUDA)

# common configs
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)

# config hidet
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
    include(${CMAKE_BINARY_DIR}/config.cmake)
else()
    include(${CMAKE_SOURCE_DIR}/config.cmake)
endif()

set(CMAKE_BUILD_TYPE ${HIDET_BUILD_TYPE})
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# submodules
if($HIDET_BUILD_TVM STREQUAL "ON")
    include(cmake/TVM.cmake)
endif()

# add runtime target
add_library(hidet_runtime SHARED
        src/hidet/runtime/cuda_context.cpp
        src/hidet/runtime/callbacks.cpp
        src/hidet/runtime/logging.cpp
        )
target_include_directories(hidet_runtime PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/include
        /usr/include
        )
set_target_properties(hidet_runtime PROPERTIES
        CUDA_RUNTIME_LIBRARY SHARED
        CUDA_ARCHITECTURES ${HIDET_CUDA_ARCH}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )

# hidet target
list(APPEND HIDET_SOURCES
        src/hidet/packedfunc.cpp
        src/hidet/cuda_api.cpp
        src/hidet/cuda_kernels.cu
        )
list(APPEND HIDET_SOURCES
        # matmul baselines
        src/hidet/baselines/matmul/reference.cu
        src/hidet/baselines/matmul/cublas.cpp
        src/hidet/baselines/matmul/cutlass.cu
        src/hidet/baselines/matmul/opt.cu
        # conv2d baselines
        src/hidet/baselines/conv2d/direct.cu
        src/hidet/baselines/conv2d/reference.cu
        src/hidet/baselines/conv2d/implicit_gemm_ref.cu
        )


if(NOT $HIDET_BASELINE_CUDNN STREQUAL OFF)
    list(APPEND HIDET_SOURCES
            # conv2d baselines
            src/hidet/baselines/conv2d/cudnn.cpp
            # softmax baseline
            src/hidet/baselines/cudnn_softmax.cpp
            # pool2d baseline
            src/hidet/baselines/cudnn_pool2d.cpp
            )
endif()



# add main target
add_library(hidet SHARED ${HIDET_SOURCES})

target_include_directories(hidet PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
        )

target_link_directories(hidet PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# link a specific cudnn library
if(NOT HIDET_CUDNN_PATH STREQUAL "ON" AND NOT HIDET_CUDNN_PATH STREQUAL "OFF")
    target_include_directories(hidet PRIVATE ${HIDET_CUDNN_PATH}/include)
    target_link_directories(hidet PRIVATE ${HIDET_CUDNN_PATH}/lib64)
    message(STATUS "Use cuDNN in: ${HIDET_CUDNN_PATH}")
endif()


if(NOT $HIDET_BASELINE_CUDNN STREQUAL OFF)
    target_link_libraries(hidet cudnn)
endif()

target_link_libraries(hidet cudart cublas curand)
target_link_libraries(hidet "-Wl,--no-as-needed" hidet_runtime)

set_target_properties(hidet PROPERTIES
        CUDA_RUNTIME_LIBRARY SHARED
        CUDA_ARCHITECTURES ${HIDET_CUDA_ARCH}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )

# add -lineinfo option to nvcc, allowing us to get the source code from binary
# do not influence optimization, can be used in nsight compute profiling
target_compile_options(hidet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)


# used for fast prototype primitive usage of cuda
add_subdirectory(experiments/cuda_code)
