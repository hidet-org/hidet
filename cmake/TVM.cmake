set(USE_CUDA ON)
set(USE_LLVM ON)
if(NOT HIDET_CUDNN_PATH STREQUAL "Auto")
    set(USE_CUDNN ${HIDET_CUDNN_PATH})
else()
    set(USE_CUDNN ON)
endif()
set(USE_CUBLAS ON)
set(USE_CCACHE ON)
set(USE_CUTLASS ON)
message(STATUS "Config TVM...")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
add_subdirectory(3rdparty/tvm)
list(POP_BACK CMAKE_MESSAGE_INDENT)
set_target_properties(tvm tvm_runtime PROPERTIES
        COMPILE_FLAGS -Wno-unused-command-line-argument
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
