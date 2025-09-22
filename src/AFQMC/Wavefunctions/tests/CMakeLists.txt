
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID afqmc_wavefunctions)
set( TEST_SRCS 
	test_wfn_factory.cpp
        )

if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
  set_source_files_properties(test_phmsd.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(test_phmsd_components.cpp PROPERTIES LANGUAGE CUDA)
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
  set_source_files_properties(test_phmsd.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(test_phmsd_components.cpp PROPERTIES LANGUAGE HIP)
endif()

add_executable( test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} catch_main afqmc)
list(LENGTH AFQMC_UNIT_TEST_INPUTS NUM_PAIRS)
math(EXPR ENDP "${NUM_PAIRS}-1")
foreach(I RANGE 0 ${ENDP} 2)
    list(GET AFQMC_UNIT_TEST_INPUTS ${I} HAMIL_FILE)
    math(EXPR J "${I}+1")
    list(GET AFQMC_UNIT_TEST_INPUTS ${J} WFN_FILE)
    get_filename_component(HAMIL ${HAMIL_FILE} NAME_WE)
    get_filename_component(WFN ${WFN_FILE} NAME_WE)
    set(UTEST_NAME unit_test_${TEST_ID}_${HAMIL}_${WFN})
    add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
                  "--hamil ${HAMIL_FILE}" "--wfn ${WFN_FILE}")
    set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
    set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
endforeach()

set(TEST_ID afqmc_phmsd)
add_executable(test_${TEST_ID} test_phmsd.cpp)
target_link_libraries(test_${TEST_ID} catch_main afqmc)
set(UTEST_NAME unit_test_${TEST_ID})
add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
              "--hamil ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Be_sto-3g/afqmc.h5"
              "--wfn ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Be_sto-3g/afqmc.h5")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")

set(TEST_ID afqmc_phmsd_components)
add_executable(test_${TEST_ID} test_phmsd_components.cpp)
target_link_libraries(test_${TEST_ID} catch_main afqmc)
set(UTEST_NAME unit_test_${TEST_ID})
add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
