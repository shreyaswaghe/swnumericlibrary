add_executable(testVector tests/testVector.cpp)
target_link_libraries(testVector swnumeric)
add_test(NAME VectorTest COMMAND testVector)

add_executable(testMatrix tests/testMatrix.cpp)
target_link_libraries(testMatrix swnumeric)
add_test(NAME MatrixTest COMMAND testMatrix)
