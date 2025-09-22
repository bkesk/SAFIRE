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
__global__ void kernel_tbt_mv(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol, 
                              T* A0, int lda, int Astride, T2 const alpha, T2 const* x0, 
                              int incx, int Xstride, int batchSize)
{
  int batch = blockIdx.z;
  if( batch > batchSize ) return;  
  T const a_(static_cast<T>(alpha));
  int i0 ( blockIdx.x*4*blockDim.x + threadIdx.x );
  int j0 ( blockIdx.y*4*blockDim.y + threadIdx.y );
  int const iN ( min(i0+4*blockDim.x, nrow) );
  int const jN ( min(j0+4*blockDim.y, ncol) );
    
  T* A( A0 + batch*Astride ); 
  T2 const* x( x0 + batch*Xstride );

  if( op == ma::TOp_PLUS )
  {
    if (dim == 0)
    {
      // A[i,j] op ( a * x[i] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T v_(a_ * static_cast<T>(x[i * incx]));
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] += v_; 
      }
    }
    else
    {
      // A[i,j] op ( a * x[j] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] += ( a_ * static_cast<T>(x[j * incx]) ); 
      }
    }
  } else if( op == ma::TOp_MINUS ) { 
    if (dim == 0)
    { 
      // A[i,j] op ( a * x[i] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T v_(a_ * static_cast<T>(x[i * incx]));
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] -= v_;
      }
    }
    else
    { 
      // A[i,j] op ( a * x[j] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] -= ( a_ * static_cast<T>(x[j * incx]) );
      }
    }
  } else if( op == ma::TOp_MUL ) { 
    if (dim == 0)
    { 
      // A[i,j] op ( a * x[i] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T v_(a_ * static_cast<T>(x[i * incx]));
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] *= v_;
      }
    }
    else
    { 
      // A[i,j] op ( a * x[j] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] *= ( a_ * static_cast<T>(x[j * incx]) );
      }
    }
  } else if( op == ma::TOp_DIV ) { 
    if (dim == 0)
    { 
      // A[i,j] op ( a * x[i] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T v_(a_ * static_cast<T>(x[i * incx]));
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] /= v_;
      }
    }
    else
    { 
      // A[i,j] op ( a * x[j] )  
      for (int i = i0; i < iN; i += blockDim.x) {
        T* A_(A + i*lda);
        for (int j = j0; j < jN; j += blockDim.y)
          A_[j] /= ( a_ * static_cast<T>(x[j * incx]) );
      }
    }
  }
}

template<class T1, class T2>
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol, 
                      std::complex<T1>* A, int lda, T2 const alpha, T2 const* x, int incx)
{
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4); 
  int ny = std::min(8, (ncol+3)/4); 
  dim3 block_dim(nx, ny, 1);
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, 1);
  kernel_tbt_mv<<<grid_dim, block_dim>>>(op, dim, nrow, ncol, 
                                  reinterpret_cast<thrust::complex<T1>*>(A), lda, 0,
                                  alpha, x, incx, 0, 1);  
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<class T1, class T2>
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<T1>* A, int lda, 
                      std::complex<T2> const alpha, std::complex<T2> const* x, int incx)
{
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4);
  int ny = std::min(8, (ncol+3)/4);
  dim3 block_dim(nx, ny, 1);
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, 1);
  kernel_tbt_mv<<<grid_dim, block_dim>>>(op, dim, nrow, ncol,
                                  reinterpret_cast<thrust::complex<T1>*>(A), lda, 0,
                                  static_cast<thrust::complex<T2> const>(alpha),         
                                  reinterpret_cast<thrust::complex<T2> const*>(x), incx, 0, 1);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<class T1, class T2>
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                std::complex<T1>* A, int lda, int Astride, 
                T2 const alpha, T2 const* x, int incx, int Xstride, int batchSize)
{ 
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4);
  int ny = std::min(8, (ncol+3)/4);
  dim3 block_dim(nx, ny, 1);
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, batchSize);
  kernel_tbt_mv<<<grid_dim, block_dim>>>(op, dim, nrow, ncol, 
                                  reinterpret_cast<thrust::complex<T1>*>(A), lda, Astride,
                                  alpha, x, incx, Xstride, batchSize);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<class T1, class T2>
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<T1>* A, int lda, int Astride,
                      std::complex<T2> const alpha, std::complex<T2> const* x, int incx,
                      int Xstride, int batchSize)
{ 
  // each block works on a sub matrix of size (4*nx, 4*ny)
  int nx = std::min(8, (nrow+3)/4);
  int ny = std::min(8, (ncol+3)/4);
  dim3 block_dim(nx, ny, 1);
  int xgrid_dim = ( nrow + 4*nx-1 ) / (4*nx);
  int ygrid_dim = ( ncol + 4*ny-1 ) / (4*ny);
  dim3 grid_dim(xgrid_dim, ygrid_dim, batchSize);
  kernel_tbt_mv<<<grid_dim, block_dim>>>(op, dim, nrow, ncol,
                                  reinterpret_cast<thrust::complex<T1>*>(A), lda, Astride,
                                  static_cast<thrust::complex<T2> const>(alpha),         
                                  reinterpret_cast<thrust::complex<T2> const*>(x), incx,
                                  Xstride, batchSize);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// template instantiations for various combiations of float/double
template
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<double>* A, int lda, 
                      std::complex<double> const alpha, std::complex<double> const* x, int incx);

template
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<double>* A, int lda, 
                      double const alpha, double const* x, int incx);

template
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<float>* A, int lda, 
                      std::complex<float> const alpha, std::complex<float> const* x, int incx);

template
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<float>* A, int lda, 
                      float const alpha, float const* x, int incx);

template
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<float>* A, int lda,
                      std::complex<double> const alpha, std::complex<double> const* x, int incx);

template
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<double>* A, int lda,
                      std::complex<float> const alpha, std::complex<float> const* x, int incx);

template
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<double>* A, int lda, int Astride,
                      std::complex<double> const alpha, std::complex<double> const* x, int incx,
                      int Xstride, int batchSize);

template
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<double>* A, int lda, int Astride,
                      double const alpha, double const* x, int incx, int Xstride, int batchSize);

template
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<float>* A, int lda, int Astride,
                      std::complex<float> const alpha, std::complex<float> const* x, int incx,
                      int Xstride, int batchSize);

template
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<float>* A, int lda, int Astride,
                      float const alpha, float const* x, int incx, int Xstride, int batchSize);

template
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<float>* A, int lda, int Astride,
                      std::complex<double> const alpha, std::complex<double> const* x, int incx,
                      int Xstride, int batchSize);

template
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<double>* A, int lda, int Astride,
                      std::complex<float> const alpha, std::complex<float> const* x, int incx,
                      int Xstride, int batchSize);

} // namespace kernels
