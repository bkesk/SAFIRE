#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_subdirectory(fft)
add_subdirectory(device_kernels)
#add_subdirectory(ac)
add_subdirectory(sparse)
#add_subdirectory(imag_axes_ft)

add_library(numerics INTERFACE)
target_link_libraries(numerics INTERFACE arch_lib fft_lib nda_c h5_c)

IF(ENABLE_SLATE)
  target_link_libraries(numerics INTERFACE slate_lapack_api slate)
ENDIF(ENABLE_SLATE)

IF(ENABLE_DEVICE)
  target_link_libraries(numerics INTERFACE sparse_lib device_kernels)
ENDIF()

install(TARGETS numerics
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(ENABLE_TBLIS)
add_subdirectory(sandbox)
endif(ENABLE_TBLIS)

IF (BUILD_UNIT_TESTS)
  add_subdirectory(fft/tests)
#  add_subdirectory(distributed_array/tests)
  add_subdirectory(shared_array/tests)
  add_subdirectory(sparse/tests)

#  IF(ENABLE_SLATE)
#    add_subdirectory(slate/tests)
#  ENDIF(ENABLE_SLATE)

  add_subdirectory(tests)
ENDIF()

