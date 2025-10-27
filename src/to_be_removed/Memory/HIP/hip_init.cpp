////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#include "Utilities/app_loggers.h"

#include "hip_init.h"
#include "hip_utilities.h"
#include "Memory/device_pointers.hpp"
#include "Memory/HIP/hip_utilities.h"
#include "Platforms/Host/OutputManager.h"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

namespace arch
{
extern hipblasHandle_t afqmc_hipblas_handle;
//extern  cublasXtHandle_t afqmc_cublasXt_handle;
extern hipsparseHandle_t afqmc_hipsparse_handle;
extern rocsolver_handle afqmc_rocsolver_handle;
extern rocrand_generator afqmc_rocrand_generator;

} // namespace arch

/*
namespace device {

  extern boost::multi::array<std::complex<double>,1,
                             device::device_allocator<std::complex<double>>>
                            *cusparse_buffer;

}
*/

namespace qmc_hip
{
extern bool afqmc_hip_handles_init;
extern hipsparseMatDescr_t afqmc_hipsparse_matrix_descr;

extern std::vector<hipStream_t> afqmc_hip_streams;

// need a cleanup routine
void HIP_INIT(boost::mpi3::shared_communicator& node, unsigned long long int iseed)
{
  if (afqmc_hip_handles_init)
    return;
  afqmc_hip_handles_init = true;

  int num_devices = 0;
  hipGetDeviceCount(&num_devices);
  app_log(1, " Running in node with {} GPUs. ", num_devices);
  hipDeviceProp_t dev;
  hip_check(hipGetDeviceProperties(&dev, 0), "hipGetDeviceProperties");
  app_log(1, " HIP compute capability: {}.{} \n ", dev.major, dev.minor);
  app_log(1, " Device Name: {} ", dev.name);
  if (num_devices < node.size())
  {
    app_error("Error: # GPU < # tasks in node. ");
    app_error("# GPU: {} ", num_devices);
    app_error("# tasks: {} ", node.size());
    app_error_flush();
    APP_ABORT("");
  }
  else if (num_devices > node.size())
  {
    __app_warn__("WARNING: Unused devices !!!!!!!!!!!!!! ");
    __app_warn__("         # tasks: {} ", node.size());
    __app_warn__("         # number of devices: {} ", num_devices);
  }

  hip_check(hipSetDevice(node.rank()), "hipSetDevice()");

  hipblas_check(hipblasCreate(&arch::afqmc_hipblas_handle), "hipblasCreate");
  //    cublas_check(cublasXtCreate (& arch::afqmc_cublasXt_handle ), "cublasXtCreate");
  int devID[8]{0, 1, 2, 3, 4, 5, 6, 7};
  //    cublas_check(cublasXtDeviceSelect(arch::afqmc_cublasXt_handle, 1, devID), "cublasXtDeviceSelect");
  //    cublas_check(cublasXtSetPinningMemMode(arch::afqmc_cublasXt_handle, CUBLASXT_PINNING_ENABLED),
  //                                            "cublasXtSetPinningMemMode");
  // Does not appear to necessary with rocsolver
  //curand_check(hiprandCreateGenerator(&arch::afqmc_curand_generator, HIPRAND_RNG_PSEUDO_DEFAULT),
  //hiprand_check(hiprandCreateGenerator(&arch::afqmc_rocrand_generator, HIPRAND_RNG_PSEUDO_MT19937),
  //"hiprandCreateGenerator");
  //hiprand_check(hiprandSetPseudoRandomGeneratorSeed(arch::afqmc_rocrand_generator,iseed),
  //"hiprandSetPseudoRandomGeneratorSeed");

  hipsolver_check(rocsolver_create_handle(&arch::afqmc_rocsolver_handle), "rocsolver_create_handle");
  hiprand_check(rocrand_create_generator(&arch::afqmc_rocrand_generator, ROCRAND_RNG_PSEUDO_MTGP32),
                "rocrand_create_generator");
  hiprand_check(rocrand_set_seed(arch::afqmc_rocrand_generator, iseed), "rocrand_set_seed");
  hipsparse_check(hipsparseCreate(&arch::afqmc_hipsparse_handle), "hipsparseCreate");
  hipsparse_check(hipsparseCreateMatDescr(&afqmc_hipsparse_matrix_descr),
                  "hipsparseCreateMatDescr: Matrix descriptor initialization failed");
  hipsparseSetMatType(afqmc_hipsparse_matrix_descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
  hipsparseSetMatIndexBase(afqmc_hipsparse_matrix_descr, HIPSPARSE_INDEX_BASE_ZERO);

  /*
    device::cusparse_buffer = new boost::multi::array<std::complex<double>,1,
                                 device::device_allocator<std::complex<double>>>(
                                 (typename boost::multi::layout_t<1u>::extensions_type{1},
                                 device::device_allocator<std::complex<double>>{}));
*/
}

} // namespace qmc_hip
