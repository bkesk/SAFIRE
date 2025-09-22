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
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"
#include "Numerics/detail/define.hpp"

namespace kernels
{

template<typename T, typename T2>
__global__ void kernel_tbt_mm(ma::TENSOR_OPERATIONS op, int nrow, int ncol, T const alpha, 
                              T const* A0, int lda, long Astride, 
                              T2 * B0, int ldb, long Bstride, int batchSize)
{
  int batch = blockIdx.z;
  if( batch > batchSize ) return;  
  T2 const alp(static_cast<T2>(alpha));
  int i0 ( blockIdx.x*4*blockDim.x + threadIdx.x );
  int j0 ( blockIdx.y*4*blockDim.y + threadIdx.y );
  int const iN ( min(i0+4*blockDim.x, nrow) );
  int const jN ( min(j0+4*blockDim.y, ncol) );
    
  T const* A( A0 + batch*Astride ); 
  T2* B( B0 + batch*Bstride );

  if( op == ma::TOp_PLUS )
  {
    // B[i,j] += a * A[i,j]   
    for (int i = i0; i < iN; i += blockDim.x) {
      T const* A_(A + i*lda);
      T2* B_(B + i*ldb);
      for (int j = j0; j < jN; j += blockDim.y)
        B_[j] += alp * static_cast<T2>(A_[j]); 
    }
  } else if( op == ma::TOp_MINUS ) { 
    // B[i,j] -= a * A[i,j]   
    for (int i = i0; i < iN; i += blockDim.x) {
      T const* A_(A + i*lda);
      T2* B_(B + i*ldb);
      for (int j = j0; j < jN; j += blockDim.y)
        B_[j] -= alp * static_cast<T2>(A_[j]); 
    }
  } else if( op == ma::TOp_MUL ) { 
    // B[i,j] *= a * A[i,j]   
    for (int i = i0; i < iN; i += blockDim.x) {
      T const* A_(A + i*lda);
      T2* B_(B + i*ldb);
      for (int j = j0; j < jN; j += blockDim.y)
        B_[j] *= alp * static_cast<T2>(A_[j]); 
    }
  } else if( op == ma::TOp_DIV ) { 
    // B[i,j] /= a * A[i,j]   
    for (int i = i0; i < iN; i += blockDim.x) {
      T const* A_(A + i*lda);
      T2* B_(B + i*ldb);
      for (int j = j0; j < jN; j += blockDim.y)
        B_[j] /= (alp * static_cast<T2>(A_[j])); 
    }
  }
}

template<class T1, class T2>
void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
                std::complex<T1> const alpha, std::complex<T1> const* A, int lda, long Astride, 
                std::complex<T2>* B, int ldb, long Bstride, int batchSize)
{ 
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4);
  int ny = std::min(8, (ncol+3)/4);
  dim3 block_dim(nx, ny, 1);
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, batchSize);
  kernel_tbt_mm<<<grid_dim, block_dim>>>(op, nrow, ncol, 
                          static_cast<thrust::complex<T1> const>(alpha), 
                          reinterpret_cast<thrust::complex<T1> const*>(A), lda, Astride,
                          reinterpret_cast<thrust::complex<T2> *>(B), ldb, Bstride, batchSize);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<class T1, class T2>
void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
                T1 const alpha, T1 const* A, int lda, long Astride,
                std::complex<T2>* B, int ldb, long Bstride, int batchSize)
{
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4);
  int ny = std::min(8, (ncol+3)/4);
  dim3 block_dim(nx, ny, 1);
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, batchSize);
  kernel_tbt_mm<<<grid_dim, block_dim>>>(op, nrow, ncol, alpha, A, lda, Astride, 
                          reinterpret_cast<thrust::complex<T2> *>(B), ldb, Bstride, batchSize);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}


// template instantiations for various combiations of float/double
template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol, 
        std::complex<double> const alpha, std::complex<double> const* A, int lda, long Astride, 
        std::complex<double> * B, int ldb, long Bstride, int batchSize); 
template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
        std::complex<float> const alpha, std::complex<float> const* A, int lda, long Astride,
        std::complex<double> * B, int ldb, long Bstride, int batchSize);
template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
        std::complex<float> const alpha, std::complex<float> const* A, int lda, long Astride,
        std::complex<float> * B, int ldb, long Bstride, int batchSize);
template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
        std::complex<double> const alpha, std::complex<double> const* A, int lda, long Astride, 
        std::complex<float> * B, int ldb, long Bstride, int batchSize);

template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
        double const alpha, double const* A, int lda, long Astride,
        std::complex<double> * B, int ldb, long Bstride, int batchSize);
template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
        float const alpha, float const* A, int lda, long Astride,
        std::complex<double> * B, int ldb, long Bstride, int batchSize);
template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
        float const alpha, float const* A, int lda, long Astride,
        std::complex<float> * B, int ldb, long Bstride, int batchSize);
template void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
        double const alpha, double const* A, int lda, long Astride,
        std::complex<float> * B, int ldb, long Bstride, int batchSize);

} // namespace kernels

