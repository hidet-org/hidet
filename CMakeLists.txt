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

# add runtime target
add_library(hidet_runtime SHARED
        src/hidet/runtime/cuda_context.cpp
        src/hidet/runtime/cpu_context.cpp
        src/hidet/runtime/callbacks.cpp
        src/hidet/runtime/logging.cpp
        )
target_include_directories(hidet_runtime PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/include
        /usr/include
        )
target_link_directories(hidet_runtime PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(hidet_runtime cudart)
set_target_properties(hidet_runtime PROPERTIES
        CUDA_RUNTIME_LIBRARY SHARED
        CUDA_ARCHITECTURES 60
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )

# hidet target
list(APPEND HIDET_SOURCES
        src/hidet/packedfunc.cpp
        src/hidet/cuda_api.cpp
        src/hidet/cuda_kernels.cu
        )

# add main target
add_library(hidet SHARED ${HIDET_SOURCES})

target_include_directories(hidet PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_SOURCE_DIR}/include
        )

target_link_directories(hidet PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


target_link_libraries(hidet cudart cublas curand)
target_link_libraries(hidet "-Wl,--no-as-needed" hidet_runtime)

set_target_properties(hidet PROPERTIES
        CUDA_RUNTIME_LIBRARY SHARED
        CUDA_ARCHITECTURES 60
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )

# add -lineinfo option to nvcc, allowing us to get the source code from binary
# do not influence optimization, can be used in nsight compute profiling
target_compile_options(hidet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
