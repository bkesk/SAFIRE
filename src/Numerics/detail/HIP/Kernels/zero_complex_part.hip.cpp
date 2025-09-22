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
#include <complex>
#include <hip/hip_runtime.h>
#include <thrust/complex.h>
#include <hip/hip_runtime.h>
#include "Numerics/detail/HIP/hip_kernel_utils.h"

namespace kernels
{
template<typename T>
__global__ void kernel_zero_complex_part(int n, thrust::complex<T>* x)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n)
    x[i] = thrust::complex<T>(x[i].real(), 0.0);
}

void zero_complex_part(int n, std::complex<double>* x)
{
  int block_dim = 256;
  int grid_dim  = (n + block_dim - 1) / block_dim;
  hipLaunchKernelGGL(kernel_zero_complex_part, dim3(grid_dim), dim3(block_dim), 0, 0, n,
                     reinterpret_cast<thrust::complex<double>*>(x));
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

void zero_complex_part(int n, std::complex<float>* x)
{
  int block_dim = 256;
  int grid_dim  = (n + block_dim - 1) / block_dim;
  hipLaunchKernelGGL(kernel_zero_complex_part, dim3(grid_dim), dim3(block_dim), 0, 0, n,
                     reinterpret_cast<thrust::complex<float>*>(x));
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

void zero_complex_part(int n, double* x) { return; }

void zero_complex_part(int n, float* x) { return; }

} // namespace kernels
