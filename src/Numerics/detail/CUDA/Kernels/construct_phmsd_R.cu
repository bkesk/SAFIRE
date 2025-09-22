#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{

// Version 3: computes Rbuff[nwalk, ndet, nex, nact], reconstructs orbs on the fly 
//  orbs[i] = refc[i] if i is not in iexcit[0:nex-1]
//          = iexcit[ p+nex ]  if iexcit[p]==i
template<typename T1, typename T2, typename T3>
__global__ void kernel_construct_phmsd_R(int nwalk, 
                                         int ndet,
                                         int nex,
                                         int nact,
                                         int nelec,
                                         int const* iexcit,
                                         int const* orbs,
                                         thrust::complex<T1> const* T,
                                         int ldT,
                                         long Tstride,
                                         thrust::complex<T2> const* I,
                                         thrust::complex<T3>* Rbuff)
{
  int idet  = blockIdx.x%ndet;
  int iwalk = blockIdx.x/ndet;   
  int i     = threadIdx.y + blockDim.y * blockIdx.y;
  int p     = threadIdx.z + blockDim.z * blockIdx.z;

  if (iwalk < nwalk and p<nex and i<nelec and idet<ndet) {
    // using the fact that blockIdx.x = iwalk*ndet + idet
    auto iexcit_ = iexcit + idet*2*nex;
    auto orbs_i = orbs[i];
    for (int q = 0; q < nex; ++q)
      if(i == iexcit_[q])
        orbs_i = iexcit_[q+nex];
    auto Rp( Rbuff + blockIdx.x*nex*nact + p*nact + orbs_i );
    auto Ip( I + blockIdx.x*nex*nex + p*nex ); // I[idet][p]
    auto T_( T + iwalk*Tstride );
    for (int q = 0; q < nex; ++q) {
      *Rp -= static_cast<thrust::complex<T3>>(Ip[q]) * 
             static_cast<thrust::complex<T3>>(T_[iexcit_[q+nex]*ldT + i]);
      if(i == iexcit_[q]) 
        *Rp += static_cast<thrust::complex<T3>>(Ip[q]);
    }
 }
}

//  weights[ndet][nwalk]
__global__ void kernel_reduce_phmsd_R(int nwalk, 
                                      int ndet,
                                      int det_per_thread,
                                      int nex,
                                      int nact,
                                      int nelec,
                                      int const* iexcit,
                                      int const* orbs,
                                      thrust::complex<double> const* weights,
                                      long ldw,
                                      thrust::complex<double>* Rbuff,
                                      thrust::complex<double>* R)
{
  int grid_dim_x = (ndet + det_per_thread - 1) / det_per_thread;  
  int idet  = blockIdx.x%grid_dim_x;
  int iwalk = blockIdx.x/grid_dim_x;   
  int a    = threadIdx.y + blockDim.y * blockIdx.y;
  int i    = threadIdx.z + blockDim.z * blockIdx.z;

  if (iwalk < nwalk and a < nact and i < nelec and idet*det_per_thread < ndet) {
    // using the fact that blockIdx.x = iwalk*ndet + idet
    auto iexcit_ = iexcit + idet*det_per_thread*2*nex;
    auto Rp = Rbuff + (iwalk*ndet + idet*det_per_thread)*nex*nact;
    thrust::complex<double> y(0.0,0.0);
    int n = min(ndet, (idet+1)*det_per_thread);
    for (int id=idet*det_per_thread; id < n; id++, 
                                             Rp+=nex*nact, 
                                             iexcit_+=2*nex)
    { 
      thrust::complex<double> w_ = weights[id*ldw + iwalk];  
      auto orbs_i = orbs[i];
      for (int q = 0; q < nex; ++q)
        if(i == iexcit_[q])
          orbs_i = iexcit_[q+nex];
      for(int p=0; p<nex; p++)
        if(i == iexcit_[p])
          y += w_ * Rp[p*nact + a]; 
      if(a == orbs_i)
        y += w_; 
    }
    double re   = y.real();
    double im   = y.imag();
    double* re_ = reinterpret_cast<double*>(R + ( iwalk*nelec + i )*nact + a);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }
}


template<typename T1, typename T2, typename T3>
void construct_phmsd_R(int nwalk, 
                       int ndet,
                       int nex,
                       int nact,
                       int nelec,
                       int const* iexcit,
                       int const* orbs,
                       std::complex<T1> const* T,
                       int ldT,
                       long Tstride,
                       std::complex<T2> const* I,
                       std::complex<T3>* Rbuff)
{
  int yblock_dim = std::min(32,nelec);  // nelec 
  int zblock_dim = std::min(8,nex);  // nex
  int grid_dim_y = (nelec + yblock_dim - 1) / yblock_dim;
  int grid_dim_z = (nex + zblock_dim - 1) / zblock_dim;
  dim3 grid_dim(nwalk*ndet, grid_dim_y, grid_dim_z);
  dim3 block_dim(1, yblock_dim, zblock_dim);
  kernel_construct_phmsd_R<<<grid_dim, block_dim>>>(nwalk, ndet, nex, nact, nelec, iexcit, orbs,
                                reinterpret_cast<thrust::complex<T1> const*>(T),
                                ldT, Tstride,
                                reinterpret_cast<thrust::complex<T2> const*>(I),
                                reinterpret_cast<thrust::complex<T3>*>(Rbuff));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void reduce_phmsd_R(int nwalk,
                    int ndet,
                    int nex,
                    int nact,
                    int nelec,
                    int const* iexcit,
                    int const* orbs,
                    std::complex<double> const* weights,
                    long ldw, 
                    std::complex<double>* Rbuff,
                    std::complex<double>* R)
{
  int ndet_per_thread = 16;  // ndet
  int yblock_dim = std::min(8,nact);   // nact
  int zblock_dim = std::min(8,nelec);  // nelec
  int grid_dim_x = (ndet + ndet_per_thread - 1) / ndet_per_thread;
  int grid_dim_y = (nact + yblock_dim - 1) / yblock_dim;
  int grid_dim_z = (nelec + zblock_dim - 1) / zblock_dim;
  dim3 grid_dim(nwalk*grid_dim_x, grid_dim_y, grid_dim_z);
  dim3 block_dim(1, yblock_dim, zblock_dim);
  kernel_reduce_phmsd_R<<<grid_dim, block_dim>>>(nwalk, ndet, ndet_per_thread, nex, 
                                nact, nelec, iexcit, orbs,
                                reinterpret_cast<thrust::complex<double> const*>(weights), ldw,
                                reinterpret_cast<thrust::complex<double>*>(Rbuff),
                                reinterpret_cast<thrust::complex<double>*>(R));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template void construct_phmsd_R(int nwalk, int ndet, int nex, int nact, int nelec, int const* iexcit,
            int const* orbs, std::complex<double> const* T, int ldT, long Tstride, 
                             std::complex<double> const* I,
                             std::complex<double>* Rbuff);
template void construct_phmsd_R(int nwalk, int ndet, int nex, int nact, int nelec, int const* iexcit,
            int const* orbs, std::complex<double> const* T, int ldT, long Tstride,
                             std::complex<double> const* I,
                             std::complex<float>* Rbuff);


}
