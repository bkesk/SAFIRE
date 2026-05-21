
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID afqmc_wavefunctions)
set( TEST_SRCS 
	test_wfn_factory.cpp
        )

add_executable( test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} PRIVATE catch_main)

set (UTEST_NAME unit_test_${TEST_ID})
add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")


set(TEST_ID afqmc_phmsd)
add_executable(test_${TEST_ID} test_phmsd.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main)

set (UTEST_NAME unit_test_${TEST_ID})
add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")


set(TEST_ID afqmc_phmsd_components)
add_executable(test_${TEST_ID} test_phmsd_components.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main)

set (UTEST_NAME unit_test_${TEST_ID})
add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")

#set(TEST_ID optm_phmsd)
#add_executable(${TEST_ID} optm_phmsd.cpp)
#target_link_libraries(${TEST_ID} catch_main)
