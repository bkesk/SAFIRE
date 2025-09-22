////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#include <cassert>
#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{
template<typename T>
__global__ void kernel_adiagApy(int N, T const alpha, T const* A, int lda, T* y, int incy)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < N)
  {
    y[i * incy] += alpha * A[i * lda + i];
  }
}

template<typename T>
__global__ void kernel_adiagApy(int N,
                                thrust::complex<T> const alpha,
                                thrust::complex<T> const* A,
                                int lda,
                                thrust::complex<T>* y,
                                int incy)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < N)
  {
    y[i * incy] += alpha * A[i * lda + i];
  }
}

void adiagApy(int N, double const alpha, double const* A, int lda, double* y, int incy)
{
  int block_dim = 256;
  int grid_dim  = (N + block_dim - 1) / block_dim;
  kernel_adiagApy<<<grid_dim, block_dim>>>(N, alpha, A, lda, y, incy);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void adiagApy(int N,
              std::complex<double> const alpha,
              std::complex<double> const* A,
              int lda,
              std::complex<double>* y,
              int incy)
{
  int block_dim = 256;
  int grid_dim  = (N + block_dim - 1) / block_dim;
  kernel_adiagApy<<<grid_dim, block_dim>>>(N, static_cast<thrust::complex<double> const>(alpha),
                                           reinterpret_cast<thrust::complex<double> const*>(A), lda,
                                           reinterpret_cast<thrust::complex<double>*>(y), incy);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void adiagApy(int N, float const alpha, float const* A, int lda, float* y, int incy)
{
  int block_dim = 256;
  int grid_dim  = (N + block_dim - 1) / block_dim;
  kernel_adiagApy<<<grid_dim, block_dim>>>(N, alpha, A, lda, y, incy);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void adiagApy(int N,
              std::complex<float> const alpha,
              std::complex<float> const* A,
              int lda,
              std::complex<float>* y,
              int incy)
{
  int block_dim = 128;
  int grid_dim  = (N + block_dim - 1) / block_dim;
  kernel_adiagApy<<<grid_dim, block_dim>>>(N, static_cast<thrust::complex<float> const>(alpha),
                                           reinterpret_cast<thrust::complex<float> const*>(A), lda,
                                           reinterpret_cast<thrust::complex<float>*>(y), incy);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

} // namespace kernels
