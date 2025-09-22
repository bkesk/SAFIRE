
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID afqmc_hamiltonians)

set( TEST_SRCS 
	test_hamiltonian_factory.cpp
        )

if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()

add_executable(test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} catch_main afqmc)

# MAM: Enable THC tests on both CPU and GPU

if(ENABLE_COMPLEX)
  set(UTEST_NAME unit_test_${TEST_ID}_ham_chol)
  add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
              "--hamil ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/ham_chol_uc.h5")
  set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
  set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
  if (NOT ENABLE_DEVICE)
    set(UTEST_NAME unit_test_${TEST_ID}_ham_chol_sc)
    add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
                  "--hamil ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5")
    set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
    set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")

    set(UTEST_NAME unit_test_${TEST_ID}_ham_thc_sc)
    add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
                  "--hamil ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_szv/ham_thc_sc.h5")
    set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
    set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
  endif()
ELSE()
  set(UTEST_NAME unit_test_${TEST_ID}_ham_chol_dense)
  add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
    "--hamil ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5")
  set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
  set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
  if (NOT ENABLE_DEVICE)
    set(UTEST_NAME unit_test_${TEST_ID}_ham_chol)
    add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
      "--hamil ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol.h5")
    set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
    set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
  endif()
endif()

