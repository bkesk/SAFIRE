#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

  configure_file(${PROJECT_SOURCE_DIR}/src/config.h.cmake.in
    ${PROJECT_BINARY_DIR}/src/config.h)

  include(${PROJECT_SOURCE_DIR}/cmake/unit_test.cmake)

  include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/extern/boost_multi")
  include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/extern/mpi_wrapper")
  include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/extern/cxxopts")
  include_directories("${PROJECT_SOURCE_DIR}/src/io")

  add_subdirectory(IO)
  add_subdirectory(arch)
  add_subdirectory(utilities)
#  add_subdirectory(Memory)
#  add_subdirectory(Numerics)
  add_subdirectory(numerics)
#  add_subdirectory(SparseMatrix)

  add_subdirectory(AFQMC)

#  add_subdirectory(main)

