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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include "Memory/CUDA/cuda_init.h"
#include "Memory/CUDA/cuda_utilities.h"
#include "cuda_arch.h"
#include "Memory/device_pointers.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "cublas_v2.h"
//#include "cublasXt.h"
#include "cusparse.h"
#include "cusolverDn.h"
#include "curand.h"

namespace arch
{
cublasHandle_t global_cublas_handle;
//  cublasXtHandle_t global_cublasXt_handle;
cusparseHandle_t global_cusparse_handle;
cusolverDnHandle_t global_cusolverDn_handle;
curandGenerator_t global_rand_generator;

cudaMemcpyKind tocudaMemcpyKind(MEMCOPYKIND v)
{
  switch (v)
  {
  case memcopyH2H: {
    return cudaMemcpyHostToHost;
  }
  case memcopyH2D: {
    return cudaMemcpyHostToDevice;
  }
  case memcopyD2H: {
    return cudaMemcpyDeviceToHost;
  }
  case memcopyD2D: {
    return cudaMemcpyDeviceToDevice;
  }
  case memcopyDefault: {
    return cudaMemcpyDefault;
  }
  }
  return cudaMemcpyDefault;
}

void INIT(boost::mpi3::shared_communicator& node, unsigned long long int iseed) { qmc_cuda::CUDA_INIT(node, iseed); }

curandGenerator_t make_device_rng(unsigned long long int iseed)
{
  curandGenerator_t rng;
  qmc_cuda::curand_check(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937),
               "curandCreateGenerator");
  qmc_cuda::curand_check(curandSetPseudoRandomGeneratorSeed(rng, iseed),
               "curandSetPseudoRandomGeneratorSeed");
  return rng;
}

void memset(void* devPtr, int value, size_t count, const std::string& message)
{ 
  cudaError_t status = cudaMemset(devPtr, value, count);
  if (status != cudaSuccess)
  { 
    if (message != "")
    { 
      std::cerr << "Error: " << message << std::endl;
    }
    std::cerr << " Error when calling cudaMemset: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Error: cudaMemset returned error code.");
  }
}

void memset2D(void* devPtr, size_t p, int value, size_t w, size_t h, const std::string& message)
{
  cudaError_t status = cudaMemset2D(devPtr, p, value, w, h);
  if (status != cudaSuccess)
  {
    if (message != "")
    {
      std::cerr << "Error: " << message << std::endl;
    }
    std::cerr << " Error when calling cudaMemset2D: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Error: cudaMemset2D returned error code.");
  }
}

void memcopy(void* dst, const void* src, size_t count, MEMCOPYKIND kind, const std::string& message)
{
  cudaError_t status = cudaMemcpy(dst, src, count, tocudaMemcpyKind(kind));
  if (status != cudaSuccess)
  {
    if (message != "")
    {
      std::cerr << "Error: " << message << std::endl;
    }
    std::cerr << " Error when calling cudaMemcpy: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Error: cudaMemcpy returned error code.");
  }
}

void memcopy2D(void* dst,
               size_t dpitch,
               const void* src,
               size_t spitch,
               size_t width,
               size_t height,
               MEMCOPYKIND kind,
               const std::string& message)
{
  cudaError_t status = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, tocudaMemcpyKind(kind));
  if (status != cudaSuccess)
  {
    if (message != "")
    {
      std::cerr << "Error: " << message << std::endl;
    }
    std::cerr << " Error when calling cudaMemcpy2D: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Error: cudaMemcpy2D returned error code.");
  }
}

void malloc(void** devPtr, size_t size, const std::string& message)
{
  cudaError_t status = cudaMalloc(devPtr, size);
  if (status != cudaSuccess)
  {
    std::cerr << " Error allocating " << size / 1024.0 / 1024.0 << " MBs on GPU." << std::endl;
    if (message != "")
    {
      std::cerr << " Error from: " << message << std::endl;
    }
    std::cerr << " Error when calling cudaMalloc: " << cudaGetErrorString(status) << std::endl;
    throw std::runtime_error("Error: cudaMalloc returned error code.");
  }
}

void free(void* p, const std::string& message)
{
  cudaError_t status = cudaFree(p);
  if (status != cudaSuccess)
  {
    if (message != "")
    {
      std::cerr << " Error from: " << message << std::endl;
    }
    std::cerr << " Error from calling cudaFree: " << cudaGetErrorString(status) << std::endl;
  }
}

} // namespace arch

