
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID math_distributed_nda)

if(ENABLE_SLATE)
set( TEST_SRCS 
     test_distributed.cpp
     test_slate.cpp
   )
else(ENABLE_SLATE)
set( TEST_SRCS 
     test_distributed.cpp
   )
endif(ENABLE_SLATE)

if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()
endif()

add_executable(test_${TEST_ID} ${TEST_SRCS} )
if(ENABLE_SLATE)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils nda_c h5_c slate_lapack_api slate)
else(ENABLE_SLATE)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main utils nda_c h5_c)
endif(ENABLE_SLATE)

add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
