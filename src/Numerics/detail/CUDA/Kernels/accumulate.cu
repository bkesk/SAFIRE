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
#include <thrust/system/cuda/detail/core/util.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"
#include "Numerics/detail/define.hpp"

namespace kernels
{

// y[n][i] += alpha * sum_j A[n][j][i]  for dim==0
//         += alpha * sum_j A[n][i][j]  for dim==1
template<typename T1, typename T2>
__global__ void kernel_accumulate_impl(int dim, int nrow, int ncol, 
                thrust::complex<T1> const alpha, thrust::complex<T1> const* A, int lda, long Astride,
                thrust::complex<T2>* y, int incy, long ystride, int batchSize)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T1>, REDUCE_BLOCK_SIZE> cache;  
  int n = blockIdx.y;
  if( n >= batchSize ) return;  
  int i = blockIdx.x;
  int j = threadIdx.z + 4 * blockDim.z * blockIdx.z;
  cache[threadIdx.z] = thrust::complex<T1>(0.0);
  if (dim == 0)
  {
    int const jN = min(j+4*blockDim.z, nrow);
    auto A_( A + n*Astride + j*lda + i); 
    for (int j_ = j; j_ < jN; j_ += blockDim.z, A_ += blockDim.z*lda)
      cache[threadIdx.z] += *A_; 
  }
  else
  {
    int const jN = min(j+4*blockDim.z, ncol);
    auto A_( A + n*Astride + i*lda + j);
    for (int j_ = j; j_ < jN; j_ += blockDim.z, A_ += blockDim.z)
      cache[threadIdx.z] += *A_;
  }
  __syncthreads(); 
  j = REDUCE_BLOCK_SIZE / 2;
  while (j > 0)
  {
    if (threadIdx.z < j)
      cache[threadIdx.z] += cache[threadIdx.z + j];
    __syncthreads();
    j /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.z == 0)
  {
    T1 re   = (alpha * cache[0]).real();
    T1 im   = (alpha * cache[0]).imag();
    T2* re_ = reinterpret_cast<T2*>(y + n*ystride + i * incy);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }
}

template<typename T1, typename T2>
__global__ void kernel_accumulate_impl(int dim, int nrow, int ncol,
                T1 const alpha, T1 const* A, int lda, long Astride,
                thrust::complex<T2>* y, int incy, long ystride, int batchSize)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<T1, REDUCE_BLOCK_SIZE> cache;  
  int n = blockIdx.y;
  if( n >= batchSize ) return;
  int i = blockIdx.x; 
  int j = threadIdx.z + 4 * blockDim.z * blockIdx.z;
  cache[threadIdx.z] = 0.0;
  if (dim == 0)
  { 
    int const jN = min(j+4*blockDim.z, nrow);
    auto A_( A + n*Astride + j*lda + i); 
    for (int j_ = j; j_ < jN; j_ += blockDim.z, A_ += blockDim.z*lda)
      cache[threadIdx.z] += *A_;
  }
  else
  { 
    int const jN = min(j+4*blockDim.z, ncol);
    auto A_( A + n*Astride + i*lda + j);
    for (int j_ = j; j_ < jN; j_ += blockDim.z, A_ += blockDim.z)
      cache[threadIdx.z] += *A_;
  }
  __syncthreads(); 
  j = REDUCE_BLOCK_SIZE / 2;
  while (j > 0)
  { 
    if (threadIdx.z < j) 
      cache[threadIdx.z] += cache[threadIdx.z + j];
    __syncthreads();
    j /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.z == 0)
  { 
    T1 re   = alpha * cache[0];
    T2* re_ = reinterpret_cast<T2*>(y + n*ystride + i * incy);
    atomicAdd(re_, re);
  }
}

template<typename T1, typename T2>
__global__ void kernel_accumulate_impl(int dim, int nrow, int ncol,
                T1 const alpha, T1 const* A, int lda, long Astride,
                T2* y, int incy, long ystride, int batchSize)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<T1, REDUCE_BLOCK_SIZE> cache;  
  int n = blockIdx.y;
  if( n >= batchSize ) return;
  int i = blockIdx.x; 
  int j = threadIdx.z + 4 * blockDim.z * blockIdx.z;
  cache[threadIdx.z] = 0.0;
  if (dim == 0)
  { 
    int const jN = min(j+4*blockDim.z, nrow);
    auto A_( A + n*Astride + j*lda + i); 
    for (int j_ = j; j_ < jN; j_ += blockDim.z, A_ += blockDim.z*lda)
      cache[threadIdx.z] += *A_;
  }
  else
  { 
    int const jN = min(j+4*blockDim.z, ncol);
    auto A_( A + n*Astride + i*lda + j);
    for (int j_ = j; j_ < jN; j_ += blockDim.z, A_ += blockDim.z)
      cache[threadIdx.z] += *A_;
  }
  __syncthreads(); 
  j = REDUCE_BLOCK_SIZE / 2;
  while (j > 0)
  { 
    if (threadIdx.z < j) 
      cache[threadIdx.z] += cache[threadIdx.z + j];
    __syncthreads();
    j /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.z == 0)
  { 
    T1 re   = (alpha * cache[0]);
    auto re_ = (y + n*ystride + i * incy);
    atomicAdd(re_, re);
  }
}

template<class T1, class T2>
void accumulate_impl(int dim, int nrow, int ncol, std::complex<T1> const alpha,
                std::complex<T1> const* A, int lda, long Astride,
                std::complex<T2>* y, int incy, long ystride, int batchSize)
{ 
  int ni = ( dim == 0 ? ncol : nrow );
  int nj = ( dim == 0 ? nrow : ncol );
  int zgrid_dim = ( nj + 4*REDUCE_BLOCK_SIZE-1 ) / (4*REDUCE_BLOCK_SIZE);
  dim3 grid_dim(ni, batchSize, zgrid_dim);
  dim3 block_dim(1, 1, REDUCE_BLOCK_SIZE);
  kernel_accumulate_impl<<<grid_dim, block_dim>>>(dim, nrow, ncol, 
                         static_cast<thrust::complex<T1> const>(alpha), 
                         reinterpret_cast<thrust::complex<T1> const*>(A), lda, Astride,
                         reinterpret_cast<thrust::complex<T2>*>(y), incy, ystride, batchSize); 
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}


template<class T1, class T2>
void accumulate_impl(int dim, int nrow, int ncol, T1 const alpha, T1 const* A, int lda, long Astride,
                std::complex<T2>* y, int incy, long ystride, int batchSize)
{
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4);
  int ny = std::min(8, (ncol+3)/4);
  dim3 block_dim(nx, ny, 1);
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, batchSize);
  kernel_accumulate_impl<<<grid_dim, block_dim>>>(dim, nrow, ncol, alpha, A, lda, Astride,
                         reinterpret_cast<thrust::complex<T2>*>(y), incy, ystride, batchSize); 
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<class T1, class T2>
void accumulate_impl(int dim, int nrow, int ncol, T1 const alpha, T1 const* A, int lda, long Astride,
                T2* y, int incy, long ystride, int batchSize)
{
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4);
  int ny = std::min(8, (ncol+3)/4);
  dim3 block_dim(nx, ny, 1); 
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, batchSize);
  kernel_accumulate_impl<<<grid_dim, block_dim>>>(dim, nrow, ncol, alpha, A, lda, Astride,
                         y, incy, ystride, batchSize); 
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}   


// template instantiations for various combiations of float/double
template void accumulate_impl(int dim, int nrow, int ncol, double const alpha, double const* A, 
            int lda, long Astride, std::complex<double>* y, int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, float const alpha, float const* A, 
            int lda, long Astride, std::complex<double>* y, int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, float const alpha, float const* A, 
            int lda, long Astride, std::complex<float>* y, int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, double const alpha, double const* A, 
            int lda, long Astride, std::complex<float>* y, int incy, long ystride, int batchSize);

template void accumulate_impl(int dim, int nrow, int ncol, std::complex<double> const alpha, 
            std::complex<double> const* A, int lda, long Astride, std::complex<double>* y, 
            int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, std::complex<float> const alpha, 
            std::complex<float> const* A, int lda, long Astride, std::complex<double>* y, 
            int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, std::complex<float> const alpha, 
            std::complex<float> const* A, int lda, long Astride, std::complex<float>* y, 
            int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, std::complex<double> const alpha, 
            std::complex<double> const* A, int lda, long Astride, std::complex<float>* y, 
            int incy, long ystride, int batchSize);

template void accumulate_impl(int dim, int nrow, int ncol, double const alpha, double const* A,
            int lda, long Astride, double* y, int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, float const alpha, float const* A,
            int lda, long Astride, double* y, int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, float const alpha, float const* A,
            int lda, long Astride, float* y, int incy, long ystride, int batchSize);
template void accumulate_impl(int dim, int nrow, int ncol, double const alpha, double const* A,
            int lda, long Astride, float* y, int incy, long ystride, int batchSize);

} // namespace kernels

