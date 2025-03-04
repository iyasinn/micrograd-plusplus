enable_testing()

add_executable(
  value_test
  value_test.cpp
)

target_link_libraries(
  value_test
  GTest::gtest_main
  value
)

include(GoogleTest)
gtest_discover_tests(value_test)

# include(FetchContent)
# FetchContent_Declare(
#     googletest
#     GIT_REPOSITORY https://github.com/google/googletest.git
#     GIT_TAG v1.15.2
# )
# FetchContent_MakeAvailable(googletest)

# file(GLOB TEST_SOURCES "*_test.cpp")
# foreach(TEST_SOURCE ${TEST_SOURCES})
#     get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
#     add_executable(${TEST_NAME} ${TEST_SOURCE})
#     target_link_libraries(${TEST_NAME}
#         PRIVATE
#             GTest::gtest_main
#             value
#     )
#     target_include_directories(${TEST_NAME}
#         PRIVATE
#             ${gtest_SOURCE_DIR}/include
#             ${gtest_BINARY_DIR}/include
#     )
#     add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
# endforeach()