cmake_minimum_required(VERSION 3.19)

project(hidet C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# config hidet
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
    include(${CMAKE_BINARY_DIR}/config.cmake)
else()
    include(${CMAKE_SOURCE_DIR}/config.cmake)
endif()

set(CMAKE_BUILD_TYPE ${HIDET_BUILD_TYPE})
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# add hidet_runtime target
add_library(hidet_runtime SHARED
        src/hidet/runtime/cuda/context.cpp
        src/hidet/runtime/cuda/cublas.cpp
        src/hidet/runtime/cuda/cudnn.cpp
        src/hidet/runtime/cuda/cuda.cpp
        src/hidet/runtime/cpu/context.cpp
        src/hidet/runtime/callbacks.cpp
        src/hidet/runtime/logging.cpp
        src/hidet/runtime/symbols.cpp
        src/hidet/runtime/int_fastdiv.cpp
        src/hidet/runtime/llm/tokenizer/decoders.cpp
        src/hidet/runtime/llm/tokenizer/models.cpp
        src/hidet/runtime/llm/tokenizer/normalizers.cpp
        src/hidet/runtime/llm/tokenizer/pattern.cpp
        src/hidet/runtime/llm/tokenizer/postprocessors.cpp
        src/hidet/runtime/llm/tokenizer/pretokenizers.cpp
        src/hidet/runtime/llm/tokenizer/tokenizer.cpp
        src/hidet/runtime/llm/tokenizer/utf8.cpp
        src/hidet/runtime/torch/stream.cpp
        )
target_include_directories(hidet_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include /usr/include)
set_target_properties(hidet_runtime PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# add hidet target
add_library(hidet SHARED
        src/hidet/empty.cpp  # empty source file
)
target_include_directories(hidet PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(hidet PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

execute_process(
    COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_INCLUDE_DIR "${TORCH_LIBRARY}/include")
execute_process(
    COMMAND python3 -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))"
    OUTPUT_VARIABLE NVIDIA_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CUDARUNTIME_INCLUDE_DIR "${NVIDIA_LIBRARY}/cuda_runtime/include")
execute_process(
    COMMAND python3 -c "import triton; import os; print(os.path.dirname(triton.__file__))"
    OUTPUT_VARIABLE TRITON_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TRITON_INCLUDE_DIR "${TRITON_LIBRARY}/backends/nvidia/include/")

add_library(hidet_torch_wrapper SHARED
        src/hidet/runtime/hidet_torch/cuda_stream.cpp
)
target_include_directories(hidet_torch_wrapper PRIVATE ${CMAKE_SOURCE_DIR}/include ${TORCH_INCLUDE_DIR} ${CUDARUNTIME_INCLUDE_DIR} ${TRITON_INCLUDE_DIR})
target_link_libraries(hidet_torch_wrapper ${TORCH_LIBRARY}/lib/libc10_cuda.so)
set_target_properties(hidet_torch_wrapper PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

target_link_libraries(hidet "-Wl,--no-as-needed" hidet_runtime)
