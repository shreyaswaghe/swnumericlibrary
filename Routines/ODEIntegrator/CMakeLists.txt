add_executable(testRK45 tests/testRK45.cpp)
target_link_libraries(testRK45 swnumeric)
add_test(NAME RK45Test COMMAND testRK45)
