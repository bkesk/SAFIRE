######################################################################
# Set version and policies 
######################################################################
cmake_minimum_required(VERSION 3.18.0)

######################################################################
# Set project and version 
######################################################################

set(AF_APP_VERSION_MAJOR 1)
set(AF_APP_VERSION_MINOR 0)
set(AF_APP_VERSION_PATCH 0)
set(AF_APP_VERSION "${AF_APP_VERSION_MAJOR}.${AF_APP_VERSION_MINOR}.${AF_APP_VERSION_PATCH}")

project(safire) 
enable_language(CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(PROJECT_CMAKE ${CMAKE_SOURCE_DIR}/cmake)

set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Library directory.")
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Binary directory.")

include_directories( ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src )

######################################################################
# CUDA/HIP build options. Read first, to be able to set policies! 
######################################################################
option(ENABLE_CUDA " Build device code with CUDA backend." OFF)
option(ENABLE_HIP " Build device code with HIP backend." OFF)

######################################################################
# CMake Policies 
######################################################################
# find_package uses <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)
# Include file check macros honor CMAKE_REQUIRED_LIBRARIES. 
cmake_policy(SET CMP0075 NEW)
# Initialize CMAKE_CUDA_ARCHITECTURES when CMAKE_CUDA_COMPILER_ID is NVIDIA. 
if(ENABLE_CUDA OR ENABLE_HIP)
 cmake_policy(SET CMP0104 NEW)
 if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
  cmake_policy(SET CMP0146 NEW)
 endif()
endif()

# Use both BOOST_ROOT and Boost_ROOT variables
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
  cmake_policy(SET CMP0144 NEW)
endif()
# Do not use FindBoost.cmake
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0167 NEW)
endif()

include(CMakeDependentOption)
include(FetchContent)

######################################################################
# CTest
######################################################################
SET(PROJECT_UNIT_TEST_DIR ${PROJECT_BINARY_DIR}/tests/bin)
enable_testing()
include( CTest )


######################################################################
# Build level
######################################################################
if ( NOT CMAKE_BUILD_TYPE ) 
  set( CMAKE_BUILD_TYPE Release )
endif()
string( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
message( STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(AFAPP_RELEASE 1)
endif()

######################################################################
# Check C++ standard and stdlib. Currently requires 20. 
######################################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

######################################################################
# set options
######################################################################
option(ENABLE_ROCM " Build device code with ROCM backend" OFF)
option(ENABLE_OPENMP " Build device code with OPENMP backend" OFF)
option(USE_NCCL "Use NCCL library with CUDA backend." OFF)
option(HAVE_MKL "Assume MKL BLAS/LAPACK library." OFF)
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_STATIC "Link to static libraries" OFF)
option(ENABLE_TIMERS "Enable internal timers" ON)
OPTION(BUILD_DOCUMENTATION "Enable compiling C++ Docs" OFF)
option(ENABLE_SPDLOG "Build with spdlog" ON)
option(ENABLE_CPPTRACE "Build with cpptrace support" ON)

option(ENABLE_FFTW "Build using FFTW3 library." ON)
option(ENABLE_SLATE "Build using Slate library." OFF)
option(ENABLE_CUTENSOR "Build using cutensor library. Must set CUTENSOR_ROOT" OFF)
option(ENABLE_TBLIS "Build using tblis library. Must set TBLIS_ROOT" OFF)
option(COMPILE_NDA_TESTS "Build with nda tests" OFF)
option(NDA_PYTHON_SUPPORT "Build nda with python support" OFF)
option(FETCH_CCCL "Fetch CCCL and enable support. Only used with ENABLE_CUDA" OFF)
option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." ON)

if( ENABLE_CUDA OR ENABLE_HIP OR ENABLE_OPENMP)
  set(ENABLE_DEVICE ON)
else()
  set(ENABLE_DEVICE OFF)
endif() 

######################################################################
# CUDA
######################################################################
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 12.0.0 REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  string(APPEND CMAKE_CUDA_FLAGS " -Drestrict=__restrict__ -std=c++20 --expt-relaxed-constexpr --extended-lambda -Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\" -Wdeprecated-declarations -w") 

  include_directories(${CUDA_INCLUDE_DIRS})
  message( STATUS " CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

  # Trying to link these as PUBLIC in the afqmc library target
  #link_libraries( CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cusolver )
  
  # allow synthetic reference memory access from the host in multi withot warnings
  ADD_DEFINITIONS(-D_DISABLE_CUDA_SLOW)

endif()

######################################################################
# ROCM 
######################################################################
if(ENABLE_ROCM)
  message(STATUS "ROCM_ROOT: ${ROCM_ROOT}")
  add_library(ROCM::libraries INTERFACE IMPORTED)
  # temporarily put rocsolver rocrand here for convenience, should be moved to Platforms.
  set_target_properties(ROCM::libraries PROPERTIES INTERFACE_INCLUDE_DIRECTORIES  "${ROCM_ROOT}/include"
                                                   INTERFACE_LINK_LIBRARIES "-L${ROCM_ROOT}/lib;-lrocsolver;-lrocrand")
endif(ENABLE_ROCM)

######################################################################
#  HIP 
######################################################################
if(ENABLE_HIP)
  if(NOT ENABLE_ROCM)
    message(FATAL_ERROR "ROCM is required to use HIP. Please set ENABLE_ROCM=ON.")
  endif()
  set(CMAKE_MODULE_PATH "${ROCM_ROOT}/hip/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP REQUIRED)

  add_library(HIP::HIP INTERFACE IMPORTED)
  # temporarily put hipsparse hipblas here for convenience, should be moved to Platforms.
  set_target_properties(HIP::HIP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ROCM_ROOT}/include"
                                            INTERFACE_COMPILE_DEFINITIONS "ENABLE_HIP"
                                            INTERFACE_LINK_LIBRARIES "-L${ROCM_ROOT}/lib;-lhipsparse;-lhipblas;-lamdhip64")
endif(ENABLE_HIP)

######################################################################
# MPI
######################################################################
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
# Mark MPI includes as SYSTEM to suppress warnings
set_target_properties(MPI::MPI_CXX PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:MPI::MPI_CXX,INTERFACE_INCLUDE_DIRECTORIES>
)
link_libraries(MPI::MPI_CXX) 
message(STATUS "MPI Executable for running programs:" ${MPIEXEC_EXECUTABLE})
message(STATUS "MPI number of processors detected on the host system: " ${MPIEXEC_MAX_NUMPROCS})
include(CMakePrintHelpers)
cmake_print_properties(TARGETS MPI::MPI_CXX PROPERTIES INTERFACE_LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES)

######################################################################
# nda: a C++ library providing multi-dimensional array classes
######################################################################
if(DEFINED NDA)
  set(FETCHCONTENT_SOURCE_DIR_NDA ${NDA})
endif()
# set nda options before FetchContent_Declare
set(CutensorSupport ${ENABLE_CUTENSOR} CACHE INTERNAL "")
set(CudaSupport ${ENABLE_CUDA} CACHE INTERNAL "")
set(TblisSupport ${ENABLE_TBLIS} CACHE INTERNAL "")
set(Build_Tests ${COMPILE_NDA_TESTS} CACHE INTERNAL "")
if(COQUI_PYTHON_SUPPORT)
  # nda_py library is needed in CoQui python version
  set(PythonSupport ON CACHE INTERNAL "")
else()
  set(PythonSupport ${NDA_PYTHON_SUPPORT} CACHE INTERNAL "")
endif()
FetchContent_Declare(
        nda
        GIT_REPOSITORY "https://github.com/triqs/nda.git"
        GIT_TAG        tensor 
)
FetchContent_MakeAvailable(nda)

set_property(GLOBAL PROPERTY nda_FOUND TRUE)
  
if(ENABLE_CUDA AND ENABLE_CUTENSOR)
  add_compile_definitions(ENABLE_CUTENSOR)
endif()
if(ENABLE_TBLIS)
  add_compile_definitions(ENABLE_TBLIS)
endif()

######################################################################
# CCCL from github (useful if CudaToolkit is old 
######################################################################
if(ENABLE_CUDA AND FETCH_CCCL)
  # set nevanlinna options before FetchContent_Declare
  FetchContent_Declare(
        cccl
        GIT_REPOSITORY "https://github.com/NVIDIA/cccl.git"
        GIT_TAG        main
  )
  FetchContent_MakeAvailable(cccl)
  include_directories(${cccl_SOURCE_DIR}/c++)
endif()


######################################################################
# BLAS/LAPACK 
######################################################################

#if(HAVE_MKL)
#  message(STATUS "HAVE_MKL is set: requiring MKL BLAS")
#  set(BLA_VENDOR "Intel10_64ilp")
#endif()

#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#cmake_print_properties(TARGETS BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES) 
#cmake_print_properties(TARGETS LAPACK::LAPACK PROPERTIES INTERFACE_LINK_LIBRARIES)

#if(NOT HAVE_MKL)
#  get_target_property(_mkl_libs BLAS::BLAS INTERFACE_LINK_LIBRARIES)
#  if("${_mkl_libs}" MATCHES "[/\\]mkl")
#      message(STATUS "Found Intel MKL: setting HAVE_MKL=ON")
#      set(HAVE_MKL ON)
#  endif()
#endif()

######################################################################
# HDF5
######################################################################
find_package(ZLIB)
set(HDF5_PREFER_PARALLEL TRUE CACHE BOOL "Request parallel/serial HDF5 library")
find_package(HDF5 REQUIRED COMPONENTS C)
message(STATUS "HDF5 version: ${HDF5_VERSION}")
if(HDF5_IS_PARALLEL)
  message(STATUS "Parallel HDF5 library found")
  set(ENABLE_PHDF5 TRUE CACHE BOOL "Enable code paths using parallel HDF5")
else(HDF5_IS_PARALLEL)
  message(STATUS "Serial HDF5 library found")
  set(ENABLE_PHDF5 FALSE CACHE BOOL "Enable code paths using parallel HDF5")
endif(HDF5_IS_PARALLEL)
add_library(IO::HDF5 INTERFACE IMPORTED)
set_target_properties(IO::HDF5 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
                                          INTERFACE_COMPILE_DEFINITIONS "H5_USE_16_API"
                                          INTERFACE_LINK_LIBRARIES "${HDF5_LIBRARIES}")
if(ENABLE_PHDF5)
  set_property(TARGET IO::HDF5 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "ENABLE_PHDF5")
endif(ENABLE_PHDF5)

######################################################################
# Boost: Header-only
######################################################################
set(Boost_NO_BOOST_CMAKE on)
find_package(Boost 1.61.0 REQUIRED)
# Add Boost include directories as SYSTEM to suppress warnings
if(Boost_INCLUDE_DIRS)
  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
endif()
add_definitions(-DBOOST_LOCALE_HIDE_AUTO_PTR)

######################################################################
# toml++: c++ parser for toml files
######################################################################
if(DEFINED TOMLPP)
  set(FETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS ${TOMLPP})
endif()

FetchContent_Declare( 
        tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

######################################################################
# spdlog - disable warnings for external library
######################################################################
# Save current warning flags
if(ENABLE_SPDLOG)
  if(DEFINED SPDLOG) 
    set(FETCHCONTENT_SOURCE_DIR_SPDLOG ${SPDLOG})
  endif()

  # Configure spdlog as header-only
  set(SPDLOG_FMT_EXTERNAL OFF)
  set(SPDLOG_BUILD_SHARED OFF)

  FetchContent_Declare(
          spdlog
          GIT_REPOSITORY https://github.com/gabime/spdlog.git
          GIT_TAG        v1.8.3
          SYSTEM
  )
  FetchContent_MakeAvailable(spdlog)
  # turn off warnings for spdlog library itself
  set_property(TARGET spdlog PROPERTY SYSTEM TRUE)
  # Suppress all warnings when compiling spdlog's own source files
  target_compile_options(spdlog PRIVATE -w)
  add_compile_definitions(ENABLE_SPDLOG)

  include_directories(SYSTEM ${spdlog_SOURCE_DIR}/include)
endif()

######################################################################
# cpptrace 
######################################################################
if(ENABLE_CPPTRACE)
  check_include_file("stdint.h" HAVE_STDINT_H)
  check_include_file("unistd.h" HAVE_UNISTD_H)
  check_include_file("fcntl.h" HAVE_FCNTL_H)
  # MAM: Failing in MacOSX, find correct way
  add_compile_definitions( $<$<COMPILE_LANGUAGE:C>:PACKAGE_VERSION="0"> )
  FetchContent_Declare(
    cpptrace
    GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG        v0.8.2 # <HASH or TAG>
  )
  FetchContent_MakeAvailable(cpptrace)
endif(ENABLE_CPPTRACE)

######################################################################
# FFTW: not using MKL's FFTW yet, figure out how if requested 
######################################################################
if(ENABLE_FFTW)
  find_package(FFTW REQUIRED DOUBLE_OPENMP_LIB) 
  add_compile_definitions(ENABLE_FFTW)
endif()


######################################################################
# catch
######################################################################
######################################################################
# catch2
######################################################################
if(DEFINED CATCH2)
  set(FETCHCONTENT_SOURCE_DIR_CATCH2 ${CATCH2})
endif()

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.10
) 
  #GIT_TAG        v3.4.0
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include( Catch )

######################################################################
add_definitions( -Drestrict=__restrict__ -DADD_ )
add_compile_options(-Wno-narrowing)

configure_file(${PROJECT_SOURCE_DIR}/src/safire.settings ${PROJECT_BINARY_DIR}/bin/safire.settings)
install(FILES ${PROJECT_BINARY_DIR}/bin/safire.settings DESTINATION bin)

#Create an Interface target for compiler warnings
add_library(${PROJECT_NAME}_warnings INTERFACE)
target_compile_options(${PROJECT_NAME}_warnings
  INTERFACE
    -Wall
    -Wextra
    -Wfloat-conversion
    -Wpedantic
    -Wno-sign-compare
    $<$<CXX_COMPILER_ID:GNU>:-Wno-comma-subscript>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi> # Disable notes about ABI changes
    $<$<CXX_COMPILER_ID:GNU>:-Wshadow=local>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-deprecated-comma-subscript>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-unknown-warning-option>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wshadow>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wimplicit-float-conversion>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-gcc-compat>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-c++20-extensions>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-c++20-compat>
    $<$<CXX_COMPILER_ID:IntelLLVM>:-Wno-tautological-constant-compare>
    # CUDA NVCC compiler warnings
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20208> # esa_on_defaulted_function_ignored
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=128> # unreachable_code
)

subdirs(src)


#######################################################################
# C++ Docs
########################################################################
if(NOT IS_SUBPROJECT AND BUILD_DOCUMENTATION)
  add_subdirectory(${PROJECT_SOURCE_DIR}/docs_cpp)
endif()

#######################################################################
# Git info
########################################################################
find_package(Git)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  set(IS_GIT_PROJECT 1)
endif()
