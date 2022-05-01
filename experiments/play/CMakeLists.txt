cmake_minimum_required(VERSION 3.19)

project(test CXX)

add_library(main SHARED main.cpp)
add_library(runtime SHARED runtime.cpp)


target_link_libraries(main runtime)
