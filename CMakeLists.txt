cmake_minimum_required(VERSION 3.22.1)
project(micrograd_plusplus)

add_executable(${PROJECT_NAME} main.cpp)

add_library(value core/Value.cpp)
target_link_libraries(${PROJECT_NAME} value)

# For testing value
# add_executable(test_value test_value.cpp)
# target_link_libraries(test_value value)




# add_library(lib STATIC Value.cpp) same as a static library
# add_library(lib DYNAMIC Value.cpp) What does it mean
# Cmake makes executables and libraries



# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)