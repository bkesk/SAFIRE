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

#ifndef CUDA_ARCH_HPP
#define CUDA_ARCH_HPP

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include "Memory/CUDA/cuda_init.h"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "cublas_v2.h"
//#include "cublasXt.h"
#include "cusparse.h"
#include "cusolverDn.h"
#include "curand.h"


namespace arch
{
extern curandGenerator_t global_rand_generator;
extern cublasHandle_t global_cublas_handle;
extern cusparseHandle_t global_cusparse_handle;
extern cusolverDnHandle_t global_cusolverDn_handle;

enum MEMCOPYKIND
{
  memcopyH2H     = cudaMemcpyHostToHost,
  memcopyH2D     = cudaMemcpyHostToDevice,
  memcopyD2H     = cudaMemcpyDeviceToHost,
  memcopyD2D     = cudaMemcpyDeviceToDevice,
  memcopyDefault = cudaMemcpyDefault
};

cudaMemcpyKind tocudaMemcpyKind(MEMCOPYKIND v);

curandGenerator_t make_device_rng(unsigned long long int iseed);

void INIT(boost::mpi3::shared_communicator& node, unsigned long long int iseed = 911ULL);

void memset(void* devPtr, int value, size_t count, const std::string& message = "");

void memset2D(void* devPtr, size_t p, int value, size_t w, size_t h, 
		const std::string& message = "");

void memcopy(void* dst, const void* src, size_t count, MEMCOPYKIND kind = memcopyDefault,
		const std::string& message = "");

void memcopy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, 
		size_t height, MEMCOPYKIND kind = memcopyDefault, const std::string& message = "");

void malloc(void** devPtr, size_t size, const std::string& message = "");

void free(void* p, const std::string& message = "");

} // namespace arch

#endif
