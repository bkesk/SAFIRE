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

#include "cuda_init.h"
#include "Utilities/AppAbort.hpp"
#include "Memory/CUDA/cuda_utilities.h"
#include "Memory/device_pointers.hpp"

#include "multi/array.hpp"
#include "multi/array_ref.hpp"

namespace arch
{

// MAM: This should all be monostate classes with get_instance interfaces.
//      They can be initialized upon first access.

extern cublasHandle_t global_cublas_handle;
//extern  cublasXtHandle_t global_cublasXt_handle;
extern cusparseHandle_t global_cusparse_handle;
extern cusolverDnHandle_t global_cusolverDn_handle;
extern curandGenerator_t global_rand_generator;

} // namespace arch

namespace qmc_cuda
{
extern bool global_cuda_handles_init;
extern cusparseMatDescr_t global_cusparse_matrix_descr;

extern std::vector<cudaStream_t> global_cuda_streams;

// need a cleanup routine
void CUDA_INIT(boost::mpi3::shared_communicator& node, unsigned long long int iseed)
{
  if (global_cuda_handles_init)
    return;
  global_cuda_handles_init = true;

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  app_log(1, " Running in node with {} GPU. ", num_devices);
  cudaDeviceProp dev;
  cuda_check(cudaGetDeviceProperties(&dev, 0), "cudaGetDeviceProperties");
  app_log(1, " CUDA compute capability: {}.{} \n ", dev.major, dev.minor);
  app_log(1, " Device Name: {} ", dev.name);
  if (dev.major <= 6)
  {
    app_log(1, " Warning CUDA major compute capability < 6.0");
  }
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
    app_warning("WARNING: Unused devices !!!!!!!!!!!!!! ");
    app_warning("         # tasks: {} ", node.size());
    app_warning("         # number of devices: {} ", num_devices);
  }

  cuda_check(cudaSetDevice(node.rank()), "cudaSetDevice()");

  cublas_check(cublasCreate(&arch::global_cublas_handle), "cublasCreate");
  //    cublas_check(cublasXtCreate (& arch::global_cublasXt_handle ), "cublasXtCreate");
  //int devID[8]{0, 1, 2, 3, 4, 5, 6, 7};
  //    cublas_check(cublasXtDeviceSelect(arch::global_cublasXt_handle, 1, devID), "cublasXtDeviceSelect");
  //    cublas_check(cublasXtSetPinningMemMode(arch::global_cublasXt_handle, CUBLASXT_PINNING_ENABLED),
  //                                            "cublasXtSetPinningMemMode");
  cusolver_check(cusolverDnCreate(&arch::global_cusolverDn_handle), "cusolverDnCreate");
  curand_check(curandCreateGenerator(&arch::global_rand_generator, CURAND_RNG_PSEUDO_MT19937),
               "curandCreateGenerator");
  curand_check(curandSetPseudoRandomGeneratorSeed(arch::global_rand_generator, iseed),
               "curandSetPseudoRandomGeneratorSeed");

  cusparse_check(cusparseCreate(&arch::global_cusparse_handle), "cusparseCreate");
  cusparse_check(cusparseCreateMatDescr(&global_cusparse_matrix_descr),
                 "cusparseCreateMatDescr: Matrix descriptor initialization failed");
  cusparseSetMatType(global_cusparse_matrix_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(global_cusparse_matrix_descr, CUSPARSE_INDEX_BASE_ZERO);

}

} // namespace qmc_cuda
