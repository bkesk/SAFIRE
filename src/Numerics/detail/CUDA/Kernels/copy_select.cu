//////////////////////////////////////////////////////////////////////
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

// expand = true,
//   B[ index[n] ][:] = beta B[ index[n] ][:] + alpha A[n][:]
// expand = false,
//   B[n][:] = beta B[n][:] + alpha A[ index[n] ][:]
template<class T1,
         class T2,
         class T3,
         class T4,
         class index_t
        >
__global__ void kernel_copy_select(int N, int M, thrust::complex<T1> alpha, 
                                          thrust::complex<T2> const* A, int lda,
                                          long Astride,
                                          thrust::complex<T3> beta, 
                                          thrust::complex<T4> * B, int ldb,
                                          long Bstride,
                                          index_t const* index, int nbatch, bool expand)
{
  int i(blockIdx.y);
  int ibatch(blockIdx.z);
  if( i < N ) {  
    thrust::complex<T4> a_ = static_cast<thrust::complex<T4>>(alpha);
    thrust::complex<T4> b_ = static_cast<thrust::complex<T4>>(beta);
    long id = long(index[i]);
    int M0(4 * blockDim.x * blockIdx.x);
    if( expand ) {
      A += ibatch*Astride + i*lda + M0;
      B += ibatch*Bstride + id*ldb + M0;
    } else {
      A += ibatch*Astride + id*lda + M0;
      B += ibatch*Bstride + i*ldb + M0;
    }
    int M_(min(4 * blockDim.x, M - M0));
    if( abs(beta) == T3(0.0) ) {
      for (int ip = threadIdx.x; ip < M_; ip += blockDim.x)
        B[ip] = a_ * static_cast<thrust::complex<T4>>(A[ip]);
    } else {
      for (int ip = threadIdx.x; ip < M_; ip += blockDim.x)
        B[ip] = b_ * B[ip] + a_ * static_cast<thrust::complex<T4>>(A[ip]);
    }
  }
}

// only specialization for std::complex, since I don't think I need others
template<typename T1, typename T2, typename T3, typename T4, typename Size>
void copy_select(int N, int M, std::complex<T1> alpha, std::complex<T2> const* A, int lda,
                            long Astride, std::complex<T3> beta, std::complex<T4>* B, int ldb,
                            long Bstride, Size const* index, int nbatch, bool expand)
{
  int m_(4 * DEFAULT_BLOCK_SIZE);
  size_t nblk((M + m_ - 1) / m_);
  dim3 grid_dim(nblk, size_t(N), nbatch);
  size_t nthr(DEFAULT_BLOCK_SIZE);
  kernel_copy_select<<<grid_dim, nthr>>>(N, M,
                                static_cast<thrust::complex<T1>>(alpha),
                                reinterpret_cast<thrust::complex<T2> const*>(A), lda, Astride,
                                static_cast<thrust::complex<T3>>(beta),
                                reinterpret_cast<thrust::complex<T4>*>(B), ldb, Bstride,
                                index, nbatch, expand);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// template specializations
template void copy_select(int,int,std::complex<float>,std::complex<float> const*,int,long,
                                  std::complex<float>,std::complex<float> *,int,long,
                                  size_t const*,int,bool); 
template void copy_select(int,int,std::complex<float>,std::complex<float> const*,int,long,
                                  std::complex<double>,std::complex<double> *,int,long,
                                  size_t const*,int,bool); 
template void copy_select(int,int,std::complex<double>,std::complex<double> const*,int,long,
                                  std::complex<float>,std::complex<float> *,int,long,
                                  size_t const*,int,bool); 
template void copy_select(int,int,std::complex<double>,std::complex<double> const*,int,long,
                                  std::complex<double>,std::complex<double> *,int,long,
                                  size_t const*,int,bool); 
template void copy_select(int,int,std::complex<float>,std::complex<float> const*,int,long,
                                  std::complex<float>,std::complex<float> *,int,long,
                                  int const*,int,bool);
template void copy_select(int,int,std::complex<float>,std::complex<float> const*,int,long,
                                  std::complex<double>,std::complex<double> *,int,long,
                                  int const*,int,bool);
template void copy_select(int,int,std::complex<double>,std::complex<double> const*,int,long,
                                  std::complex<float>,std::complex<float> *,int,long,
                                  int const*,int,bool);
template void copy_select(int,int,std::complex<double>,std::complex<double> const*,int,long,
                                  std::complex<double>,std::complex<double> *,int,long,
                                  int const*,int,bool);

} // namespace kernels
