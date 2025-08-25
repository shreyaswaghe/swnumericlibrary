add_executable(testVector tests/testVector.cpp)
target_link_libraries(testVector swnumeric)
add_test(NAME VectorTest COMMAND testVector)
