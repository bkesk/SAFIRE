//////////////////////////////////////////////////////////////////////
// File created by:
// Miguel A. Morales
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

namespace kernels
{


// C[g] = sum_n,w,i,j Xw[w] * A[n][g][i][w][j] * B[n][g][j][w][i]
template<class T1, class T2, class T3, class T4>
__global__ void kernel_AGiwj_BGjwi_CG_v1(int nbatch, int ng, int ni, int nwalk, int nj,
        thrust::complex<T1> const* Xw, thrust::complex<T2> const* A,
        thrust::complex<T3> const* B, thrust::complex<T4> * C)
{

  // x-> batch
  int batch = blockIdx.x;
  // y-> G
  int iG = blockIdx.y;
  // z-> walk
  int iw = blockIdx.z;

  A += ( batch*ng + iG) * ni * nwalk * nj; 
  B += ( batch*ng + iG) * ni * nwalk * nj; 

  // i/j over block, assumes 16x16 block
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T2>, 256> Ac;
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T4>, 256> cache;

  int tid = threadIdx.y*blockDim.x + threadIdx.x; 
  int tid2 = threadIdx.x*blockDim.y + threadIdx.y; 

  Ac[tid] = thrust::complex<T2>(0.0);
  cache[tid] = thrust::complex<T4>(0.0);

  //int i = threadIdx.x;
  //int j = threadIdx.y; 
/* Naive implementation */
/* 
  if(threadIdx.x < ni and threadIdx.y < nj)    
    for(int i=threadIdx.x; i<ni; i+=blockDim.x) 
      for(int j=threadIdx.y; j<nj; j+=blockDim.y)
        cache[tid] += static_cast<thrust::complex<T4>>(A[ ( i * nwalk + iw) * nj + j ]) *
                      static_cast<thrust::complex<T4>>(B[ ( j * nwalk + iw) * ni + i ]); 
*/
  // assumes blockDim.x == blockDim.y
  int nbi = ( ni + blockDim.x - 1 ) / blockDim.x; 
  int nbj = ( nj + blockDim.y - 1 ) / blockDim.y;
  for(int ib=0; ib<nbi; ib++) {
    for(int jb=0; jb<nbj; jb++) {
      // copy A into shm, notice the transposition of indexes 

      int i = ib*blockDim.x + threadIdx.y;
      int j = jb*blockDim.y + threadIdx.x;
      __syncthreads();
      if(i < ni and j < nj)
        Ac[tid] = A[ ( i * nwalk + iw) * nj + j ];
      // contract  
      i = ib*blockDim.x + threadIdx.x;
      j = jb*blockDim.y + threadIdx.y;  
      __syncthreads();
      if(i<ni and j<nj)
        cache[tid] += static_cast<thrust::complex<T4>>(Ac[tid2]) *
                      static_cast<thrust::complex<T4>>(B[ ( j * nwalk + iw) * ni + i ]);

/*
      int i = ib*blockDim.x + threadIdx.x;
      int j = jb*blockDim.y + threadIdx.y;
      if(i<ni and j<nj)
        cache[tid] += static_cast<thrust::complex<T4>>(A[ ( i * nwalk + iw) * nj + j ]) *
                      static_cast<thrust::complex<T4>>(B[ ( j * nwalk + iw) * ni + i ]); 
*/

    }   
  }   

  __syncthreads(); 
  int i = (blockDim.x*blockDim.y) / 2;
  while (i > 0)
  {
    if (tid < i)
      cache[tid] += cache[tid + i];
    __syncthreads();
    i /= 2; 
  }
  if (tid == 0)
  {
    thrust::complex<T4> wgt = static_cast<thrust::complex<T4>>(Xw[iw]);
    T4 re   = (wgt * cache[0]).real();
    T4 im   = (wgt * cache[0]).imag();
    T4* re_ = reinterpret_cast<T4*>(C + iG);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }  

}


// keeping this simple, assuming contiguous arrays
template<class T1, class T2, class T3, class T4>
void AGiwj_BGjwi_CG_impl(int nbatch, int ng, int ni, int nwalk, int nj,
        std::complex<T1> const* Xw, std::complex<T2> const* A, 
        std::complex<T3> const* B, std::complex<T4> * C) 
{
  dim3 grid_dim(nbatch, ng, nwalk);
  dim3 block_dim(16, 16, 1);
  kernel_AGiwj_BGjwi_CG_v1<<<grid_dim, block_dim>>>(nbatch, ng, ni, nwalk, nj,
                                    reinterpret_cast<thrust::complex<T1> const*>(Xw), 
                                    reinterpret_cast<thrust::complex<T2> const*>(A), 
                                    reinterpret_cast<thrust::complex<T3> const*>(B), 
                                    reinterpret_cast<thrust::complex<T4>*>(C));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template void AGiwj_BGjwi_CG_impl(int,int,int,int,int,std::complex<double> const*, 
    std::complex<double> const*,std::complex<double> const*,std::complex<double> *);
template void AGiwj_BGjwi_CG_impl(int,int,int,int,int,std::complex<float> const*, 
    std::complex<float> const*,std::complex<float> const*,std::complex<float> *);
template void AGiwj_BGjwi_CG_impl(int,int,int,int,int,std::complex<double> const*, 
    std::complex<float> const*,std::complex<float> const*,std::complex<float> *);
template void AGiwj_BGjwi_CG_impl(int,int,int,int,int,std::complex<double> const*, 
    std::complex<float> const*,std::complex<float> const*,std::complex<double> *);


}
