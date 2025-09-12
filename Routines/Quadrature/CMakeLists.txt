add_executable(test_gl ${CMAKE_CURRENT_SOURCE_DIR}/tests/testGaussLegendre.cpp)
target_link_libraries(test_gl PRIVATE swnumeric)
add_test(NAME QuadratureTests COMMAND test_gl)
