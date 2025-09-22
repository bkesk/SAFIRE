#include <stdexcept>
#include <cassert>
#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <thrust/system/cuda/detail/core/util.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#include "Memory/CUDA/cuda_utilities.h"

#define __NTHR__ 512 
#define __NCHOL__ 128
#define __BZ__ 32 

namespace kernels
{

template<typename T1, typename T2, typename T3>
__global__ void kernel_ph_excited_energy_real_dense_chol_Tpna_first(int nwalk, int ndet, int nex, 
        int nact, int nelec, int nchol, int const* iexcit, int const* refc, 
        thrust::complex<T1> const* T,
        thrust::complex<T1> const* R, 
        thrust::complex<T2> const* wgt, int ldW,
        thrust::complex<T3> *EX, int ldEX,
        thrust::complex<T3> *EJ, int ldEJ,
        thrust::complex<T1> *KE, int ldKE, long KEstride)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T3>, 2*__NTHR__+1> cache;
  extern __shared__ int occps[];
  using Type = thrust::complex<T3>;
  using Type1 = thrust::complex<T1>;

  Type const TWO(2.0);
  Type const HALF(0.5);
  int idet = blockIdx.y;
  int iwalk = blockIdx.z;
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  T += iwalk*long(nelec*nchol*nact);
  R += long(iwalk*ndet+idet)*long(nex*nact); 
  KE += idet*KEstride + iwalk*ldKE;
  iexcit += idet*2*nex;
  Type keJ(0.0);

  cache[threadIdx.x] = Type(0.0);
  cache[threadIdx.x+blockDim.x] = Type(0.0);
  
  // calculate occ string
  {
    int i=threadIdx.x;
    while(i < nelec) {
      occps[i] = refc[i];
      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ie1];
        if(ip == i) {
          occps[i] = iexcit[ie1 + nex];
          break;
        }
      }    
      i+=blockDim.x;
    }
  }  

  // calculate eJ0
  if(blockIdx.x==0)
  {
    int c=threadIdx.x;
    while(c < nchol) {
      Type e_(0.0);  
      for(int i=0; i<nelec; i++)
        e_ += T[ (i*nchol+c)*nact + refc[i] ]; 
      cache[threadIdx.x] += e_*e_;
      c += blockDim.x;
    }
    __syncthreads();
    int i = blockDim.x / 2;
    while (i > 0)
    {
      if (threadIdx.x < i) 
        cache[threadIdx.x] += cache[threadIdx.x + i];
      __syncthreads();
      i /= 2; //not sure bitwise operations are actually faster
    }
    if(threadIdx.x==0)
      cache[blockDim.x] -= cache[0];
    cache[threadIdx.x] = Type(0.0);
  }

  if(n < nchol) {
    for(int ie1=0; ie1<nex; ie1++) {
      int ip = iexcit[ie1];
      auto Tip(T + (ip*nchol+n)*nact);
      auto Rp(R + ie1*nact);
      // ie1==ie2 term

      Type e1_(0.0);
      for(int a=0; a<nact; a++)
        e1_ += static_cast<Type>( Tip[a] * Rp[a] );
      cache[threadIdx.x] += e1_*e1_;
      keJ += e1_; 

      // R[p]*R[q] terms
      for(int ie2=ie1+1; ie2<nex; ie2++) {
        int iq = iexcit[ie2];
        auto Tiq(T + (iq*nchol+n)*nact);
        auto Rq(R + ie2*nact);
        e1_=Type(0.0);  
        for(int a=0; a<nact; a++)
          e1_ += static_cast<Type>( Tiq[a] * Rp[a] );
        Type e2_(0.0);
        for(int a=0; a<nact; a++)
          e2_ += static_cast<Type>( Tip[a] * Rq[a] );
        cache[threadIdx.x] += TWO * e1_*e2_;
      }

    }  
    // Rdiag-Rdiag terms
    for(int i=0; i<nelec; i++) {
      int Oi = occps[i];
      int ri = refc[i];
      if( not(Oi==ri) ) 
        cache[threadIdx.x] += T[(i*nchol+n)*nact + Oi]*T[(i*nchol+n)*nact + Oi] - 
                    T[(i*nchol+n)*nact + ri]*T[(i*nchol+n)*nact + ri];
      for(int j=i+1; j<nelec; j++) {
        int Oj = occps[j];
        int rj = refc[j];
        if( Oi!=ri or Oj!=rj )
          cache[threadIdx.x] += TWO * (T[(i*nchol+n)*nact + Oj]*T[(j*nchol+n)*nact + Oi] - 
                             T[(i*nchol+n)*nact + rj]*T[(j*nchol+n)*nact + ri]);
      }
    }

    // R[diagonal]*R[diagonal] J-term
    for(int i=0; i<nelec; i++) 
      keJ += static_cast<Type>( T[ (i*nchol+n)*nact + occps[i] ] );

    KE[n] = static_cast<Type1>(keJ);
    cache[threadIdx.x+blockDim.x] += keJ*keJ;
  }
  __syncthreads(); 
  n = blockDim.x / 2;
  while (n > 0)
  { 
    if (threadIdx.x < n) {
      cache[threadIdx.x] += cache[threadIdx.x + n];
      cache[threadIdx.x+blockDim.x] += cache[blockDim.x + threadIdx.x + n];
    }
    __syncthreads();
    n /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
  {
    // ex[iw][d] *= -0.5*wgt[d][iw]
    cache[0] *= static_cast<Type>(-HALF*wgt[idet*ldW+iwalk]);
    // eJ[iw][d] *= 0.5*wgt[d][iw]
    cache[blockDim.x] *= static_cast<Type>(HALF*wgt[idet*ldW+iwalk]);
    T3 re   = cache[0].real();
    T3 im   = cache[0].imag();
    // EX[iw] += eX[iw][d]
    T3* re_ = reinterpret_cast<T3*>(EX + iwalk * ldEX);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
    re   = cache[blockDim.x].real();
    im   = cache[blockDim.x].imag();
    // EJ[iw] += eJ[iw][d]
    re_ = reinterpret_cast<T3*>(EJ + iwalk * ldEJ);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }
}

// second step for the R[p]*Rdiag term, with parallelization over 'a' only
template<typename T1, typename T2, typename T3>
__global__ void kernel_ph_excited_energy_real_dense_chol_Tpna_second(int nwalk, int ndet, int nex, 
        int nact, int nelec, int nchol, int const* iexcit, int const* refc, 
        thrust::complex<T1> const* T,
        thrust::complex<T1> const* R, 
        thrust::complex<T2> const* wgt, int ldW,
        thrust::complex<T3> *EX, int ldEX)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T3>, __BZ__> cache;
  extern __shared__ int occps[];
  using Type = thrust::complex<T3>;
  using Type1 = thrust::complex<T1>;

  Type const TWO(2.0);
  Type const HALF(0.5);
  int idet = blockIdx.y;
  int iwalk = blockIdx.z;
  T += iwalk*long(nelec*nchol*nact);
  R += long(iwalk*ndet+idet)*long(nex*nact); 
  iexcit += idet*2*nex;

  cache[threadIdx.x] = Type(0.0);
  
  // calculate occ string
  {
    int i=threadIdx.x;
    while(i < nelec) {
      occps[i] = refc[i];
      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ie1];
        if(ip == i) {
          occps[i] = iexcit[ie1 + nex];
          break;
        }
      }    
      i+=blockDim.x;
    }
  }  
  __syncthreads(); 

  if(threadIdx.x < nact) {
    int n1 = min(nchol, (blockIdx.x+1)*__NCHOL__);
    for(int n=blockIdx.x*__NCHOL__; n<n1; n++) 
    {
      for(int ie1=0; ie1<nex; ie1++) {
        int ip = iexcit[ie1];
        auto Tip(T + (ip*nchol+n)*nact);
        auto Rp(R + ie1*nact);

       // R[p]*R[diagonal] term
        for(int j=0; j<nelec; j++) {
          int Oj = occps[j];
          auto Tj(T + (j*nchol+n)*nact);
          Type e1_(0.0);
          for(int a=threadIdx.x; a<nact; a+=blockDim.x)
            e1_ += static_cast<Type>( Tj[a] * Rp[a] );
          cache[threadIdx.x] += e1_*static_cast<Type>(Tip[Oj]);
        }
      }  
    }
  }
  __syncthreads(); 
  int n = blockDim.x / 2;
  while (n > 0)
  { 
    if (threadIdx.x < n) 
      cache[threadIdx.x] += cache[threadIdx.x + n];
    __syncthreads();
    n /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
  {
    // ex[iw][d] *= -0.5*wgt[d][iw]
    // factor of TWO from above cancels factor of HALF here!    
    cache[0] *= static_cast<Type>(-wgt[idet*ldW+iwalk]);
    T3 re   = cache[0].real();
    T3 im   = cache[0].imag();
    // EX[iw] += eX[iw][d]
    T3* re_ = reinterpret_cast<T3*>(EX + iwalk * ldEX);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }
}


template<typename T1, typename T2, typename T3, typename T4>
__global__ void kernel_ph_excited_1body_energy(int nwalk, int ndet, int nex, int nact, int nelec,
        int const* iexcit, int const* refc, thrust::complex<T1> const* S,
        thrust::complex<T2> const* R, thrust::complex<T3> const* wgt, int ldW,
        thrust::complex<T4> *E, int ldE)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T4>, 256> cache;
  using Type = thrust::complex<T4>;

  int idet = blockIdx.x%ndet;
  int iwalk = blockIdx.x/ndet;
  if(idet >= ndet or iwalk >= nwalk)
    return;

  S += iwalk*nelec*nact;
  R += (iwalk*ndet+idet)*nex*nact;
  iexcit += idet*2*nex;
  int tid = threadIdx.z*blockDim.y + threadIdx.y;
  int nthr = blockDim.y*blockDim.z; 
  cache[tid] = Type(0.0);

  int p = blockDim.z*blockIdx.z + threadIdx.z;
  int a0 = 4*blockDim.y*blockIdx.y + threadIdx.y;
  if( p < nex and a0 < nact) 
  {
    int aN = min(a0+4*int(blockDim.y),nact);
    int ip = iexcit[p];
    for(int a=a0; a<aN; a+=blockDim.y)  
      cache[tid] += static_cast<Type>(S[ip*nact+a])*static_cast<Type>(R[p*nact+a]); 
    if(blockIdx.y == 0 and threadIdx.y==0)
      cache[tid] += static_cast<Type>(S[ip*nact+iexcit[p+nex]]) - 
                    static_cast<Type>(S[ip*nact+refc[ip]]);
  }

  __syncthreads();
  int n = nthr / 2;
  while (n > 0)
  {
    if (tid < n) 
      cache[tid] += cache[tid + n];
    __syncthreads();
    n /= 2; //not sure bitwise operations are actually faster
  }
  if (tid == 0)
  { 
    cache[0] *= static_cast<Type>(wgt[idet*ldW+iwalk]);
    T4 re   = cache[0].real();
    T4 im   = cache[0].imag();
    // EX[iw] += eX[iw][d]
    T4* re_ = reinterpret_cast<T4*>(E + iwalk * ldE);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }

}

template<typename T1, typename T2, typename T3>
void ph_excited_2body_energy_dense_cholesky_Tpna(int nwalk, int ndet, int nex, int nact, int nelec,
        int nchol, int const* iexcit, int const* refc, std::complex<T1> const* T,
        std::complex<T1> const* R, std::complex<T2> const* wgt, int ldW,
        std::complex<T3> *EX, int ldEX,
        std::complex<T3> *EJ, int ldEJ,
        std::complex<T1> *KE, int ldKE, long KEstride)
{
  // MAM: current kernels are 1-2x slower when running all walkers simultaneously
  //      looping over walkers for better performance
  int bsize = __NTHR__;
  int nc = __NCHOL__;
  int gdim = (nchol + bsize - 1) / bsize;
  dim3 block_dim(bsize,1,1);
  dim3 grid_dim(gdim, ndet, 1);
  int gdim2 = (nchol + nc - 1) / nc;
  dim3 grid_dim2(gdim2, ndet, 1);
  dim3 block_dim2(__BZ__, 1, 1);
  int sm_size(nelec*sizeof(int));

  long Tstr = long(nelec)*long(nchol)*long(nact);
  long Rstr = long(ndet)*long(nex*nact);
  for( int iw=0; iw<nwalk; iw++) 
  {
    kernel_ph_excited_energy_real_dense_chol_Tpna_first<<<grid_dim, block_dim, sm_size>>>(
            1, ndet, nex, nact, nelec, nchol, iexcit, refc,
            reinterpret_cast<thrust::complex<T1> const*>(T)+iw*Tstr,
            reinterpret_cast<thrust::complex<T1> const*>(R)+iw*Rstr,
            reinterpret_cast<thrust::complex<T2> const*>(wgt)+iw, ldW,
            reinterpret_cast<thrust::complex<T3> *>(EX)+iw*ldEX, ldEX,
            reinterpret_cast<thrust::complex<T3> *>(EJ)+iw*ldEJ, ldEJ,
            reinterpret_cast<thrust::complex<T1> *>(KE)+iw*ldKE, ldKE, KEstride);

    kernel_ph_excited_energy_real_dense_chol_Tpna_second<<<grid_dim2, block_dim2, sm_size>>>(
            1, ndet, nex, nact, nelec, nchol, iexcit, refc,
            reinterpret_cast<thrust::complex<T1> const*>(T)+iw*Tstr,
            reinterpret_cast<thrust::complex<T1> const*>(R)+iw*Rstr,
            reinterpret_cast<thrust::complex<T2> const*>(wgt)+iw, ldW,
            reinterpret_cast<thrust::complex<T3> *>(EX)+iw*ldEX, ldEX);
  }
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<typename T1, typename T2, typename T3, typename T4>
void ph_excited_1body_energy(int nwalk, int ndet, int nex, int nact, int nelec,
        int const* iexcit, int const* refc, std::complex<T1> const* S,
        std::complex<T2> const* R, std::complex<T3> const* wgt, int ldW,
        std::complex<T4> *E, int ldE)
{
  // more or less optimized for smaller problems and lots of determinants, 
  // write another partitioning for few determinants and large nchol/nelec/nact
  int bsize = 4; 
  int gdimy = (nact + 255) / 256;
  int gdimz = (nex + bsize - 1) / bsize;
  dim3 block_dim(1,64,bsize);
  dim3 grid_dim(ndet*nwalk,gdimy,gdimz);
  kernel_ph_excited_1body_energy<<<grid_dim, block_dim>>>(
            nwalk, ndet, nex, nact, nelec, iexcit, refc,
            reinterpret_cast<thrust::complex<T1> const*>(S),
            reinterpret_cast<thrust::complex<T2> const*>(R),
            reinterpret_cast<thrust::complex<T3> const*>(wgt), ldW,
            reinterpret_cast<thrust::complex<T4> *>(E), ldE);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template void ph_excited_2body_energy_dense_cholesky_Tpna(int,int,int,int,int,int,
        int const*,int const*,std::complex<double> const*,std::complex<double> const*,
        std::complex<double> const*,int,std::complex<double> *,int,
        std::complex<double> *,int,std::complex<double> *, int, long);
template void ph_excited_2body_energy_dense_cholesky_Tpna(int,int,int,int,int,int,
        int const*,int const*,std::complex<float> const*,std::complex<float> const*,
        std::complex<double> const*,int,std::complex<double> *,int,
        std::complex<double> *,int,std::complex<float> *, int, long);
template void ph_excited_2body_energy_dense_cholesky_Tpna(int,int,int,int,int,int,
        int const*,int const*,std::complex<float> const*,std::complex<float> const*,
        std::complex<float> const*,int,std::complex<double> *,int,
        std::complex<double> *,int,std::complex<float> *, int, long);

template void ph_excited_1body_energy(int,int,int,int,int,
        int const*,int const*,std::complex<double> const*,std::complex<double> const*,
        std::complex<double> const*,int,std::complex<double> *,int);
template void ph_excited_1body_energy(int,int,int,int,int,
        int const*,int const*,std::complex<double> const*,std::complex<float> const*,
        std::complex<double> const*,int,std::complex<double> *,int);

}
