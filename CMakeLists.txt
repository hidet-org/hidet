cmake_minimum_required(VERSION 3.19)

project(hidet C CXX)

set(CMAKE_CXX_STANDARD 11)
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
        src/hidet/runtime/cuda/cuda.cpp
        src/hidet/runtime/cpu/context.cpp
        src/hidet/runtime/callbacks.cpp
        src/hidet/runtime/logging.cpp
        src/hidet/runtime/symbols.cpp
        )
target_include_directories(hidet_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include /usr/include)
set_target_properties(hidet_runtime PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# add hidet target
add_library(hidet SHARED
        src/hidet/empty.cpp  # empty source file
)
target_include_directories(hidet PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(hidet PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
target_link_libraries(hidet "-Wl,--no-as-needed" hidet_runtime)
