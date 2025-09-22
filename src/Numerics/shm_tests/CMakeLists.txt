#////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the __SFQMC_LICENSE_TYPE__
#// License.  See LICENSE file in top directory for details.
#//
#// Copyright (c) 2025 SAFIRE Developers
#//
#////////////////////////////////////////////////////////////////////////////////

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SFQMC_UNIT_TEST_DIR})

SET(SRC_DIR afqmc_shm_numerics)
SET(UTEST_EXE test_${SRC_DIR})
SET(UTEST_NAME deterministic-unit_test_${SRC_DIR})

ADD_EXECUTABLE(${UTEST_EXE} test_shm_gemm.cpp) 
TARGET_LINK_LIBRARIES(${UTEST_EXE} catch_main)

#ADD_TEST(NAME ${UTEST_NAME} COMMAND "${SFQMC_UNIT_TEST_DIR}/${UTEST_EXE}")
ADD_UNIT_TEST(${UTEST_NAME} "${SFQMC_UNIT_TEST_DIR}/${UTEST_EXE}")
SET_TESTS_PROPERTIES(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${SFQMC_UNIT_TEST_DIR})
SET_PROPERTY(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
