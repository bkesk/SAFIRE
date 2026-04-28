#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

message("Adding AFQMC time propagation unit tests")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID afqmc_sdet_tpropg)

set( TEST_SRCS 
	test_sparse_vs_dense_gemm.cpp
        )

add_executable(test_${TEST_ID} ${TEST_SRCS} )

target_link_libraries(test_${TEST_ID} PRIVATE catch_main) 

# don't add to unit tests, too slow and only used for performance check
#add_unit_test(unit_test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
#set_tests_properties(unit_test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
#set_property(TEST unit_test_${TEST_ID} APPEND PROPERTY LABELS "afqmc")

