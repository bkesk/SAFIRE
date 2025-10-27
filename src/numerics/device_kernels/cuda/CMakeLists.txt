#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

if(ENABLE_CUDA)
 
  set(DEV_KERNEL_SRCS
    argmax_min.cu 
    copy_select.cu
    complex_tools.cu
    symmetry_tools.cu
    potentials.cu
    kpoint_tools.cu
  )

  foreach( X ${DEV_KERNEL_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
  add_library(cuda_kernels ${DEV_KERNEL_SRCS})
#  target_include_directories(cuda_kernels PUBLIC ${PROJECT_SOURCE_DIR}/extern/thrust)
  target_link_libraries(cuda_kernels PUBLIC io_lib arch_lib nda_c CUDA::cudart CCCL::CCCL)

  target_link_libraries(device_kernels INTERFACE cuda_kernels)

ENDIF(ENABLE_CUDA)

#if(ENABLE_DEVICE)
#if (BUILD_UNIT_TESTS)
#  add_subdirectory(tests)
#endif()
#endif()



