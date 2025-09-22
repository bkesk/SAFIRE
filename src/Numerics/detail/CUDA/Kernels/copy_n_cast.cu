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
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"

namespace kernels
{
template<typename T, typename Q>
__global__ void kernel_copy_n_cast(T const* A, long N, Q* B)
{
  long N0(8 * blockDim.x * blockIdx.x);
  T const* A_(A + N0);
  Q* B_(B + N0);
  long N_(min( long(8 * blockDim.x) , N - N0));
  for (long ip = threadIdx.x; ip < N_; ip += blockDim.x)
  {
    B_[ip] = static_cast<Q>(A_[ip]);
  }
}

template<typename T, typename Q>
__global__ void kernel_copy_n_cast(thrust::complex<T> const* A, long N, thrust::complex<Q>* B)
{
  long N0(8 * blockDim.x * blockIdx.x);
  thrust::complex<T> const* A_(A + N0);
  thrust::complex<Q>* B_(B + N0);
  long N_(min(long(8 * blockDim.x), N - N0));
  for (long ip = threadIdx.x; ip < N_; ip += blockDim.x)
  {
    B_[ip] = static_cast<thrust::complex<Q>>(A_[ip]);
  }
}

template<typename T, typename Q>
__global__ void kernel_copy_n_cast(int N, int M, T const* A, int lda, long Astr, 
                                                 Q* B, int ldb, long Bstr, int nb)
{ 
  if(blockIdx.y < N and blockIdx.z < nb) {
    int M0(8 * blockDim.x * blockIdx.x);
    A += blockIdx.z*Astr + blockIdx.y*lda + M0;
    B += blockIdx.z*Bstr + blockIdx.y*ldb + M0;
    int M_(min( 8 * blockDim.x , M - M0));
    for (int ip = threadIdx.x; ip < M_; ip += blockDim.x)
      B[ip] = static_cast<Q>(A[ip]);
  }
}

template<typename T, typename Q>
__global__ void kernel_copy_n_cast(int N, int M, thrust::complex<T> const* A, int lda, long Astr, 
                                                 thrust::complex<Q>* B, int ldb, long Bstr, int nb)
{
  if(blockIdx.y < N and blockIdx.z < nb) {
    int M0(8 * blockDim.x * blockIdx.x);
    A += blockIdx.z*Astr + blockIdx.y*lda + M0;
    B += blockIdx.z*Bstr + blockIdx.y*ldb + M0;
    int M_(min( 8 * blockDim.x , M - M0));
    for (int ip = threadIdx.x; ip < M_; ip += blockDim.x)
      B[ip] = static_cast<thrust::complex<Q>>(A[ip]);
  }
}

void copy_n_cast(double const* A, long n, float* B)
{
  long n_(8 * DEFAULT_BLOCK_SIZE);
  size_t nblk((n + n_ - 1) / n_);
  size_t nthr(DEFAULT_BLOCK_SIZE);
  kernel_copy_n_cast<<<nblk, nthr>>>(A, n, B);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}
void copy_n_cast(float const* A, long n, double* B)
{
  long n_(8 * DEFAULT_BLOCK_SIZE);
  size_t nblk((n + n_ - 1) / n_);
  size_t nthr(DEFAULT_BLOCK_SIZE);
  kernel_copy_n_cast<<<nblk, nthr>>>(A, n, B);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}
void copy_n_cast(std::complex<double> const* A, long n, std::complex<float>* B)
{
  long n_(8 * DEFAULT_BLOCK_SIZE);
  size_t nblk((n + n_ - 1) / n_);
  size_t nthr(DEFAULT_BLOCK_SIZE);
  kernel_copy_n_cast<<<nblk, nthr>>>(reinterpret_cast<thrust::complex<double> const*>(A), n,
                                     reinterpret_cast<thrust::complex<float>*>(B));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}
void copy_n_cast(std::complex<float> const* A, long n, std::complex<double>* B)
{
  long n_(8 * DEFAULT_BLOCK_SIZE);
  size_t nblk((n + n_ - 1) / n_);
  size_t nthr(DEFAULT_BLOCK_SIZE);
  kernel_copy_n_cast<<<nblk, nthr>>>(reinterpret_cast<thrust::complex<float> const*>(A), n,
                                     reinterpret_cast<thrust::complex<double>*>(B));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// this assumes that no dimension goes over ~2Gb, fix if problematic!
template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, T1 const* A, int lda, long Astride,
                      T2* B, int ldb, long Bstride, int nbatch)
{
  int M_(8 * DEFAULT_BLOCK_SIZE);
  int nblk((M + M_ - 1) / M_);
  dim3 grid_dim(nblk, N, nbatch);
  kernel_copy_n_cast<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(N, M, A, lda, Astride, 
                                                             B, ldb, Bstride, nbatch);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, std::complex<T1> const* A, int lda, long Astride,
                      std::complex<T2>* B, int ldb, long Bstride, int nbatch)
{
  int M_(8 * DEFAULT_BLOCK_SIZE);
  int nblk((M + M_ - 1) / M_);
  dim3 grid_dim(nblk, N, nbatch);
  kernel_copy_n_cast<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(N,M,
                    reinterpret_cast<thrust::complex<T1> const*>(A), lda, Astride,
                    reinterpret_cast<thrust::complex<T2>*>(B),ldb,Bstride,nbatch);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template void copy_n_cast_impl(int,int,double const*,int,long,float*,int,long,int);
template void copy_n_cast_impl(int,int,float const*,int,long,double*,int,long,int);
template void copy_n_cast_impl(int,int,double const*,int,long,double*,int,long,int);
template void copy_n_cast_impl(int,int,float const*,int,long,float*,int,long,int);
template void copy_n_cast_impl(int,int,std::complex<float> const*,int,long,
                                       std::complex<float>*,int,long,int);
template void copy_n_cast_impl(int,int,std::complex<double> const*,int,long,
                                       std::complex<double>*,int,long,int);
template void copy_n_cast_impl(int,int,std::complex<double> const*,int,long,
                                       std::complex<float>*,int,long,int);
template void copy_n_cast_impl(int,int,std::complex<float> const*,int,long,
                                       std::complex<double>*,int,long,int);

} // namespace kernels
