

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(UTEST_EXE test_numerics)
set(UTEST_NAME unit_test_numerics)

set( NUMERICS_TEST_SRCS 
  test_dense_numerics.cpp 
  test_ma_blas.cpp
  test_sparse_numerics.cpp 
  test_sparse_numerics_native.cpp 
  test_batched_operations.cpp
  test_ma_blas_extensions.cpp 
  test_tensor_operations.cpp 
  test_determinant.cpp
  test_misc_kernels.cpp
	)

if(ENABLE_CUDA)  
  foreach( X ${NUMERICS_TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif()

add_executable(${UTEST_EXE} ${NUMERICS_TEST_SRCS} ) 

target_link_libraries(${UTEST_EXE} PUBLIC catch_main qmcio memory BLAS::BLAS LAPACK::LAPACK spdlog::spdlog)
if(ENABLE_CUDA)
  target_link_libraries(${UTEST_EXE} PUBLIC kernels)
elseif(ENABLE_HIP)
  target_link_libraries(${UTEST_EXE} PUBLIC kernels)
else(ENABLE_CUDA)
endif(ENABLE_CUDA)

add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/${UTEST_EXE}")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "numerics")
