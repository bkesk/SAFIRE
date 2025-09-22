
#-------------------------------------------------------------------
# Sources
#-------------------------------------------------------------------

if(ENABLE_CUDA)
  set(NUMERICS_SRCS 
    detail/CUDA/Kernels/determinant.cu
    detail/CUDA/Kernels/adotpby.cu
    detail/CUDA/Kernels/fill_n.cu
    detail/CUDA/Kernels/uninitialized_fill_n.cu
    detail/CUDA/Kernels/uninitialized_copy_n.cu
    detail/CUDA/Kernels/axty.cu
    detail/CUDA/Kernels/adiagApy.cu
    detail/CUDA/Kernels/sum.cu
    detail/CUDA/Kernels/acAxpbB.cu
    detail/CUDA/Kernels/print.cu
    detail/CUDA/Kernels/setIdentity.cu
    detail/CUDA/Kernels/zero_complex_part.cu
    detail/CUDA/Kernels/copy_n_cast.cu
    detail/CUDA/Kernels/copy_select.cu
    detail/CUDA/Kernels/inplace_cast.cu
    detail/CUDA/Kernels/ajw_to_waj.cu
    detail/CUDA/Kernels/vKKwij_to_vwKiKj.cu
    detail/CUDA/Kernels/KaKjw_to_QKajw.cu
    detail/CUDA/Kernels/vbias_from_v1.cu
    detail/CUDA/Kernels/KaKjw_to_KKwaj.cu
    detail/CUDA/Kernels/batched_dot_wabn_wban.cu
    detail/CUDA/Kernels/batched_Tab_to_Klr.cu
    detail/CUDA/Kernels/dot_wabn.cu
    detail/CUDA/Kernels/Tab_to_Kl.cu
    detail/CUDA/Kernels/sampleRNG.cu
    detail/CUDA/Kernels/construct_X.cu
    detail/CUDA/Kernels/reference_operations.cu
    detail/CUDA/Kernels/term_by_term_matrix_vec.cu
    detail/CUDA/Kernels/term_by_term_matrix_mat.cu
    detail/CUDA/Kernels/axpyBatched.cu
    detail/CUDA/Kernels/Auwn_Bun_Cuw.cu
    detail/CUDA/Kernels/get_diagonal.cu
    detail/CUDA/Kernels/batched_dot.cu
    detail/CUDA/Kernels/add_scalar.cu
    detail/CUDA/Kernels/spVi_Bij_yj.cu
    detail/CUDA/Kernels/getGIJ.cu
    detail/CUDA/Kernels/extract_overlap_matrix.cu
    detail/CUDA/Kernels/construct_phmsd_R.cu
    detail/CUDA/Kernels/accumulate.cu
    detail/CUDA/Kernels/add_diagonal.cu
    detail/CUDA/Kernels/phmsd_determinants.cu
    detail/CUDA/Kernels/phmsd_inverse.cu
    detail/CUDA/Kernels/phmsd_energy.cu
    detail/CUDA/Kernels/complex_conjugate.cu
    detail/CUDA/Kernels/AGiwj_BGjwi_CG.cu
  )
elseif(ENABLE_HIP)
  set(NUMERICS_SRCS
    detail/HIP/hip_kernel_utils.cpp
    detail/HIP/Kernels/determinant.hip.cpp
    detail/HIP/Kernels/adotpby.hip.cpp
    detail/HIP/Kernels/fill_n.hip.cpp
    detail/HIP/Kernels/uninitialized_fill_n.hip.cpp
    detail/HIP/Kernels/uninitialized_copy_n.hip.cpp
    detail/HIP/Kernels/axty.hip.cpp
    detail/HIP/Kernels/adiagApy.hip.cpp
    detail/HIP/Kernels/sum.hip.cpp
    detail/HIP/Kernels/acAxpbB.hip.cpp
    detail/HIP/Kernels/print.hip.cpp
    detail/HIP/Kernels/setIdentity.hip.cpp
    detail/HIP/Kernels/zero_complex_part.hip.cpp
    detail/HIP/Kernels/batchedDot.hip.cpp
    detail/HIP/Kernels/copy_n_cast.hip.cpp
    detail/HIP/Kernels/copy_select.hip.cpp
    detail/HIP/Kernels/inplace_cast.hip.cpp
    detail/HIP/Kernels/ajw_to_waj.hip.cpp
    detail/HIP/Kernels/vKKwij_to_vwKiKj.hip.cpp
    detail/HIP/Kernels/KaKjw_to_QKajw.hip.cpp
    detail/HIP/Kernels/vbias_from_v1.hip.cpp
    detail/HIP/Kernels/KaKjw_to_KKwaj.hip.cpp
    detail/HIP/Kernels/batched_dot_wabn_wban.hip.cpp
    detail/HIP/Kernels/batched_Tab_to_Klr.hip.cpp
    detail/HIP/Kernels/dot_wabn.hip.cpp
    detail/HIP/Kernels/Tab_to_Kl.hip.cpp
    detail/HIP/Kernels/sampleGaussianRNG.hip.cpp
    detail/HIP/Kernels/construct_X.hip.cpp
    detail/HIP/Kernels/reference_operations.hip.cpp
    detail/HIP/Kernels/term_by_term_matrix_vec.hip.cpp
    detail/HIP/Kernels/axpyBatched.hip.cpp
    detail/HIP/Kernels/Auwn_Bun_Cuw.hip.cpp
    detail/HIP/Kernels/get_diagonal.hip.cpp
  )
else(ENABLE_CUDA)
endif(ENABLE_CUDA)

if(ENABLE_CUDA)
  foreach( X ${NUMERICS_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
  add_library(kernels ${NUMERICS_SRCS})
  target_include_directories(kernels PUBLIC ${PROJECT_SOURCE_DIR}/extern/thrust)  
  target_link_libraries(kernels PUBLIC memory PRIVATE spdlog::spdlog)

elseif(ENABLE_HIP)
  foreach( X ${NUMERICS_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
  add_library(kernels ${NUMERICS_SRCS})
  target_include_directories(kernels PUBLIC ${PROJECT_SOURCE_DIR}/extern/thrust)  
  target_link_libraries(kernels PUBLIC memory PRIVATE spdlog::spdlog)

else(ENABLE_CUDA)
endif(ENABLE_CUDA)



if (BUILD_UNIT_TESTS)
  subdirs(tests)
endif()

subdirs(performance)
