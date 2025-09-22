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
//#include "hip_settings.h"
//#include "hip_utilities.h"
#include "Numerics/detail/HIP/Kernels/hip_settings.h"
#include "Numerics/detail/HIP/hip_kernel_utils.h"

namespace kernels
{
// simple
// A[k][i] = B[k][i][i]
template<typename T>
__global__ void kernel_get_diagonal_strided(int nk,
                                            int ni,
                                            thrust::complex<T> const* B,
                                            int ldb,
                                            int stride,
                                            thrust::complex<T>* A,
                                            int lda)
{
  int k = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i < ni) && (k < nk))
    A[k * lda + i] = B[k * stride + i * ldb + i];
}

// A[k][i] = B[k][i][i]
void get_diagonal_strided(int nk,
                          int ni,
                          std::complex<double> const* B,
                          int ldb,
                          int stride,
                          std::complex<double>* A,
                          int lda)
{
  size_t nthr = 32;
  size_t nbks = (ni + nthr - 1) / nthr;
  dim3 grid_dim(nbks, nk, 1);
  hipLaunchKernelGGL(kernel_get_diagonal_strided, dim3(grid_dim), dim3(nthr), 0, 0, nk, ni,
                     reinterpret_cast<thrust::complex<double> const*>(B), ldb, stride,
                     reinterpret_cast<thrust::complex<double>*>(A), lda);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

void get_diagonal_strided(int nk,
                          int ni,
                          std::complex<float> const* B,
                          int ldb,
                          int stride,
                          std::complex<float>* A,
                          int lda)
{
  size_t nthr = 32;
  size_t nbks = (ni + nthr - 1) / nthr;
  dim3 grid_dim(nbks, nk, 1);
  hipLaunchKernelGGL(kernel_get_diagonal_strided, dim3(grid_dim), dim3(nthr), 0, 0, nk, ni,
                     reinterpret_cast<thrust::complex<float> const*>(B), ldb, stride,
                     reinterpret_cast<thrust::complex<float>*>(A), lda);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

} // namespace kernels
