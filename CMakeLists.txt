cmake_minimum_required(VERSION 3.22.1)
project(micrograd)
add_executable(${PROJECT_NAME} main.cpp)
add_library(lib Value.cpp)
target_link_libraries(${PROJECT_NAME} lib)




# add_library(lib STATIC Value.cpp) same as a static library
# add_library(lib DYNAMIC Value.cpp) What does it mean
# Cmake makes executables and libraries