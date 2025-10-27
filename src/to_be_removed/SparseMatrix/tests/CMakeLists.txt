
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

if(ENABLE_CUDA)
  set_source_files_properties(test_csr_matrix.cpp PROPERTIES LANGUAGE CUDA)
elseif(ENABLE_HIP)
  set_source_files_properties(test_csr_matrix.cpp PROPERTIES LANGUAGE HIP)
else(ENABLE_CUDA)
endif(ENABLE_CUDA)

add_executable(test_sparse test_csr_matrix.cpp)
target_link_libraries(test_sparse catch_main memory ${CMAKE_DL_LIBS})
if(ENABLE_CUDA)
  target_link_libraries(test_sparse kernels)
elseif(ENABLE_HIP)
  target_link_libraries(test_sparse kernels)
else(ENABLE_CUDA)
endif(ENABLE_CUDA)

ADD_UNIT_TEST(unit_test_sparse "${PROJECT_UNIT_TEST_DIR}/test_sparse")
SET_TESTS_PROPERTIES(unit_test_sparse PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
SET_PROPERTY(TEST unit_test_sparse APPEND PROPERTY LABELS "sparse")

