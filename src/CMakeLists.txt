#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

  configure_file(${PROJECT_SOURCE_DIR}/src/config.h.cmake.in
    ${PROJECT_BINARY_DIR}/src/config.h)

  include(${PROJECT_SOURCE_DIR}/cmake/unit_test.cmake)

  include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/extern/boost_multi")
  include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/extern/mpi_wrapper")
  include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/extern/cxxopts")
  include_directories("${PROJECT_SOURCE_DIR}/src/io")

  subdirs(IO)
  subdirs(arch)
  subdirs(utilities)
#  subdirs(Memory)
#  subdirs(Numerics)
  subdirs(numerics)
#  subdirs(SparseMatrix)

  subdirs(AFQMC)

#  subdirs(main)

