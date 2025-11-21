
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID afqmc_driver)
set( TEST_SRCS 
	test_driver_factory.cpp
        )

add_executable( test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} catch_main afqmc)
#set(HAMIL_FILE ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense_wfn.h5)
#get_filename_component(HAMIL ${HAMIL_FILE} NAME_WE)
#set(UTEST_NAME unit_test_${TEST_ID})
#add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
#              "--hamil ${HAMIL_FILE}")
#set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
#set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")

