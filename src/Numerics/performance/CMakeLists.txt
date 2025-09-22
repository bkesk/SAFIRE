#////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the __SFQMC_LICENSE_TYPE__
#// License.  See LICENSE file in top directory for details.
#//
#// Copyright (c) 2025 SAFIRE Developers
#//
#////////////////////////////////////////////////////////////////////////////////

message("Building performance executable ")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

if(ENABLE_CUDA)
  set_source_files_properties(performance.cpp data_transfer.cpp PROPERTIES LANGUAGE CUDA)
elseif(ENABLE_HIP)
  set_source_files_properties(performance.cpp data_transfer.cpp PROPERTIES LANGUAGE HIP)
else()
endif()
add_executable(qmc-performance performance.cpp)
target_include_directories(qmc-performance PUBLIC "${LIBXML2_INCLUDE_DIR}")

target_link_libraries(qmc-performance memory BLAS::BLAS LAPACK::LAPACK spdlog::spdlog)
if(ENABLE_CUDA OR ENABLE_HIP)
  target_link_libraries(qmc-performance kernels)
else()
endif()

if(ENABLE_CUDA OR ENABLE_HIP)
add_executable(qmc-data_transfer data_transfer.cpp) 
target_include_directories(qmc-data_transfer PUBLIC "${LIBXML2_INCLUDE_DIR}")
target_link_libraries(qmc-data_transfer memory kernels spdlog::spdlog)
endif()
