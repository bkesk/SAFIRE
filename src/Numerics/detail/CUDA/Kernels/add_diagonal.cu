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

//  A[n][i][y[i]] += w[n] 
template<class T1, class T2>
__global__ void kernel_add_diagonal(int N, int* y, thrust::complex<T1> const* w, int incx, 
                                    thrust::complex<T2>* A, int lda, long Astride, int nb)
{ 
  int ibatch(blockIdx.y);
  if( ibatch < nb ) {  
    int i0(4 * blockDim.x * blockIdx.x + threadIdx.x);
    int iN(min(i0 + 4 * blockDim.x, N));
    thrust::complex<T2> w_ = static_cast<thrust::complex<T2>>(w[incx*ibatch]);
    A += ibatch*Astride;
    for (int ip = i0; ip < iN; ip += blockDim.x)
      A[ip*lda + y[ip]] += w_;
  }
}

template<typename T1, typename T2>
void add_diagonal(int N, int* y, std::complex<T1> const* w, int incx, 
                                 std::complex<T2>* A, int lda, long Astride, int nb)
{
  int n_(4 * DEFAULT_BLOCK_SIZE);
  size_t nblk((N + n_ - 1) / n_);
  dim3 grid_dim(nblk, nb, 1);
  size_t nthr(DEFAULT_BLOCK_SIZE);
  kernel_add_diagonal<<<grid_dim, nthr>>>(N, y, 
            reinterpret_cast<thrust::complex<T1> const*>(w), incx, 
            reinterpret_cast<thrust::complex<T2> *>(A), lda, Astride, nb); 
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template void add_diagonal(int N, int* y, std::complex<double> const* w, int incx, 
                           std::complex<double>* A, int lda, long Astride, int nb);
template void add_diagonal(int N, int* y, std::complex<float> const* w, int incx, 
                           std::complex<float>* A, int lda, long Astride, int nb);
template void add_diagonal(int N, int* y, std::complex<float> const* w, int incx, 
                           std::complex<double>* A, int lda, long Astride, int nb);
template void add_diagonal(int N, int* y, std::complex<double> const* w, int incx, 
                           std::complex<float>* A, int lda, long Astride, int nb);

} // kernels
