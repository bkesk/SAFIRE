#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID sandbox)

set( TEST_SRCS
     contract_tests.cpp 
     mpi_tests.cpp
   )

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

add_executable(${TEST_ID} ${TEST_SRCS} )
target_link_libraries(${TEST_ID} PRIVATE catch_main utils numerics)

#//////////////////////////////////////////////////////////////////////////////////////

set(TEST_ID h5_tests)

set( TEST_SRCS
     h5_tests.cpp
   )

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

add_executable(${TEST_ID} ${TEST_SRCS} )
target_link_libraries(${TEST_ID} PRIVATE catch_main utils numerics)

