#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

configure_file(${PROJECT_SOURCE_DIR}/src/config.h.cmake.in
  ${PROJECT_BINARY_DIR}/src/config.h)

include(${PROJECT_SOURCE_DIR}/cmake/unit_test.cmake)

add_library(safire_lib STATIC)

add_subdirectory(IO)
add_subdirectory(arch)
add_subdirectory(utilities)
add_subdirectory(numerics)

add_subdirectory(AFQMC)

target_include_directories(safire_lib PUBLIC
  ${PROJECT_SOURCE_DIR}/src/
  ${PROJECT_BINARY_DIR}/src/
)

target_link_libraries(safire_lib PUBLIC nda::nda_c mpi3 Boost::headers cxxopts::cxxopts safire_warnings)

if(ENABLE_CUDA)
  target_link_libraries(safire_lib PRIVATE CUDA::curand CUDA::cudart CCCL::CCCL)
endif()

if(ENABLE_SPDLOG)
  target_link_libraries(safire_lib PUBLIC spdlog::spdlog)
endif()

if(ENABLE_CPPTRACE)
  target_link_libraries(safire_lib PUBLIC cpptrace::cpptrace)
endif()


add_executable(safire main.cpp)
target_link_libraries(safire PRIVATE safire_lib) 

install(TARGETS safire
        RUNTIME DESTINATION bin)

