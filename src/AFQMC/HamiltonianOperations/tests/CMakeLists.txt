
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID afqmc_hamiltonianoperations)

set( TEST_SRCS 
	test_hamiltonian_operations.cpp
        )

#if(ENABLE_CUDA)
#  foreach( X ${TEST_SRCS} )
#    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
#  endforeach()
#elseif(ENABLE_HIP)
#  foreach( X ${TEST_SRCS} )
#    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
#  endforeach()
#endif()

add_executable( test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} catch_main afqmc)

# MAM: Add unit tests for all HOps types and backends!
set(UTEST_NAME unit_test_${TEST_ID}_ham_chol_uc_wfn_rhf)
add_unit_test(${UTEST_NAME}
		"${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
                "--hamil ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/ham_chol_uc.h5"
                "--wfn ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/wfn_rhf.h5")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")


