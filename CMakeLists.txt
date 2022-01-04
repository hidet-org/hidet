cmake_minimum_required(VERSION 3.19)

project(hidet C CXX CUDA)

set(HIDET_SRCS src/hidet/packedfunc.cpp src/hidet/runtime.cpp)

add_library(hidet SHARED ${HIDET_SRCS})

target_include_directories(hidet PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(hidet PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
set_target_properties(hidet PROPERTIES
        CUDA_ARCHITECTURES "75")

