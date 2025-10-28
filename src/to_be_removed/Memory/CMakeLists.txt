#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set (MEMORY_SRCS
  buffer_managers.cpp
  )

if(ENABLE_CUDA)
  set(MEMORY_SRCS ${MEMORY_SRCS}
    CUDA/cuda_arch.cpp
    CUDA/cuda_utilities.cpp
    CUDA/cuda_init.cpp
  )
  foreach( X ${MEMORY_SRCS} ) 
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  set(MEMORY_SRCS ${MEMORY_SRCS}
    HIP/hip_utilities.cpp
    HIP/hip_arch.cpp
    HIP/hip_init.cpp
  )
  foreach( X ${MEMORY_SRCS} )    
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
else(ENABLE_CUDA)
endif(ENABLE_CUDA)

add_library(memory ${MEMORY_SRCS})
target_link_libraries(memory PRIVATE utils)
if(ENABLE_CUDA)
target_link_libraries(memory PRIVATE CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cusolver)
endif()
