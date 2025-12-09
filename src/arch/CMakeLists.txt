#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRCS
  arch.cpp
)

IF(ENABLE_CUDA)
SET(SRCS ${SRCS}
  CUDA/cuda_init.cpp
  CUDA/cuda_sync.cpp
)
ENDIF()

ADD_LIBRARY(arch_lib ${SRCS})
TARGET_LINK_LIBRARIES(arch_lib PUBLIC qmcio)
IF(ENABLE_CUDA)
  # since nda does not link curand
  TARGET_LINK_LIBRARIES(arch_lib PUBLIC CUDA::curand)
ENDIF()

install(TARGETS arch_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


