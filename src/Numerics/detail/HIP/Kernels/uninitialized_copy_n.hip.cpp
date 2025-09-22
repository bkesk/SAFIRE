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
#include <type_traits>
#include <thrust/complex.h>
#include "Numerics/detail/HIP/hip_kernel_utils.h"
#include "Numerics/detail/HIP/Kernels/hip_settings.h"
//#include "Numerics/detail/HIP/Kernels/strided_range.hpp"

namespace kernels
{
template<typename T, typename Size>
__global__ void kernel_uninitialized_copy_n(Size N, T const* x, Size incx, T* arr, Size incy)
{
  for (Size ip = Size(threadIdx.x); ip < N; ip += Size(blockDim.x))
  {
    arr[ip * incy] = x[ip * incx];
  }
}


void uninitialized_copy_n(int N, double const* first, int incx, double* array, int incy)
{
  int N_(8 * DEFAULT_BLOCK_SIZE);
  int nblk((N + N_ - 1) / N_);
  int nthr(DEFAULT_BLOCK_SIZE);
  hipLaunchKernelGGL(kernel_uninitialized_copy_n, dim3(nblk), dim3(nthr), 0, 0, N, first, incx, array, incy);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

void uninitialized_copy_n(int N, std::complex<double> const* first, int incx, std::complex<double>* array, int incy)
{
  long N_(8 * DEFAULT_BLOCK_SIZE);
  long nblk((N + N_ - 1) / N_);
  long nthr(DEFAULT_BLOCK_SIZE);
  hipLaunchKernelGGL(kernel_uninitialized_copy_n, dim3(nblk), dim3(nthr), 0, 0, N, first, incx, array, incy);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

void uninitialized_copy_n(int N, int const* first, int incx, int* array, int incy)
{
  long N_(8 * DEFAULT_BLOCK_SIZE);
  long nblk((N + N_ - 1) / N_);
  long nthr(DEFAULT_BLOCK_SIZE);
  hipLaunchKernelGGL(kernel_uninitialized_copy_n, dim3(nblk), dim3(nthr), 0, 0, N, first, incx, array, incy);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

// long
void uninitialized_copy_n(long N, double const* first, long incx, double* array, long incy)
{
  long N_(8 * DEFAULT_BLOCK_SIZE);
  long nblk((N + N_ - 1) / N_);
  long nthr(DEFAULT_BLOCK_SIZE);
  hipLaunchKernelGGL(kernel_uninitialized_copy_n, dim3(nblk), dim3(nthr), 0, 0, N, first, incx, array, incy);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

void uninitialized_copy_n(long N, std::complex<double> const* first, long incx, std::complex<double>* array, long incy)
{
  long N_(8 * DEFAULT_BLOCK_SIZE);
  long nblk((N + N_ - 1) / N_);
  long nthr(DEFAULT_BLOCK_SIZE);
  hipLaunchKernelGGL(kernel_uninitialized_copy_n, dim3(nblk), dim3(nthr), 0, 0, N, first, incx, array, incy);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

void uninitialized_copy_n(long N, int const* first, long incx, int* array, long incy)
{
  long N_(8 * DEFAULT_BLOCK_SIZE);
  long nblk((N + N_ - 1) / N_);
  long nthr(DEFAULT_BLOCK_SIZE);
  hipLaunchKernelGGL(kernel_uninitialized_copy_n, dim3(nblk), dim3(nthr), 0, 0, N, first, incx, array, incy);
  qmc_hip::hip_kernel_check(hipGetLastError());
  qmc_hip::hip_kernel_check(hipDeviceSynchronize());
}

} // namespace kernels
