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

namespace kernels
{
// very sloppy, needs improvement!!!!
// v1[nkpts][nchol_max][nwalk]
// vb[2*nchol_tot][nwalk]
// ncholpQ0 includes factor of 2
template<typename T, typename T2, typename T3>
__global__ void kernel_vbias_from_v1(int nwalk,
                                     int nkpts,
                                     int nchol_max,
                                     int* Qsym,
                                     int* kminus,
                                     int* ncholpQ,
                                     int* ncholpQ0,
                                     thrust::complex<T3> const alpha,
                                     thrust::complex<T> const* v1,
                                     thrust::complex<T2>* vb)
{
  // This is inefficient now, eventually map to assigned Qs and only launch based on this
  // essentially only need a map to the assigned Q's and use blockIdx.x to index into this array
  int Q = blockIdx.x;
  if (Q >= nkpts || blockIdx.y > 1)
    return;
  if (Qsym[Q] < 0)
    return;
  int Qm  = kminus[Q];
  int nc0 = ncholpQ0[Q];
  int nc  = ncholpQ[Q];
  int ncm = ncholpQ[Qm];
  // now redefine Qm based on Qsym
  if (Qsym[Q] > 0)
    Qm = nkpts + Qsym[Q] - 1;

  if (blockIdx.y == 0)
  {
    // v+
    thrust::complex<T2>* vb_(vb + nc0 * nwalk);
    thrust::complex<T> const* v1_(v1 + Q * nchol_max * nwalk);
    thrust::complex<T> const* v2_(v1 + Qm * nchol_max * nwalk);
    thrust::complex<T2> alp(static_cast<thrust::complex<T2>>(alpha));
    // v+ = a*(v[Q]+v[-Q])
    if (threadIdx.x < nc && threadIdx.y < nwalk)
    {
      for (int n = threadIdx.x; n < nc; n += blockDim.x)
        for (int w = threadIdx.y; w < nwalk; w += blockDim.y)
          vb_[n * nwalk + w] += alp * static_cast<thrust::complex<T2>>(v1_[n * nwalk + w]);
    }
    if (threadIdx.x < ncm && threadIdx.y < nwalk)
    {
      for (int n = threadIdx.x; n < ncm; n += blockDim.x)
        for (int w = threadIdx.y; w < nwalk; w += blockDim.y)
          vb_[n * nwalk + w] += alp * static_cast<thrust::complex<T2>>(v2_[n * nwalk + w]);
    }
  }
  else if (blockIdx.y == 1)
  {
    // v-
    thrust::complex<T2>* vb_(vb + (nc0 + nc) * nwalk);
    thrust::complex<T> const* v1_(v1 + Q * nchol_max * nwalk);
    thrust::complex<T> const* v2_(v1 + Qm * nchol_max * nwalk);
    // v- = -a*i*(v[Q]-v[-Q])
    thrust::complex<T2> ialpha(static_cast<thrust::complex<T2>>(alpha) * thrust::complex<T2>(0.0, 1.0));
    if (threadIdx.x < nc && threadIdx.y < nwalk)
    {
      for (int n = threadIdx.x; n < nc; n += blockDim.x)
        for (int w = threadIdx.y; w < nwalk; w += blockDim.y)
          vb_[n * nwalk + w] -= ialpha * static_cast<thrust::complex<T2>>(v1_[n * nwalk + w]);
    }
    if (threadIdx.x < ncm && threadIdx.y < nwalk)
    {
      for (int n = threadIdx.x; n < ncm; n += blockDim.x)
        for (int w = threadIdx.y; w < nwalk; w += blockDim.y)
          vb_[n * nwalk + w] += ialpha * static_cast<thrust::complex<T2>>(v2_[n * nwalk + w]);
    }
  }
}

template<typename T1, typename T2, typename T3>
void vbias_from_v1(int nwalk,
                   int nkpts,
                   int nchol_max,
                   int* Qsym,
                   int* kminus,
                   int* ncholpQ,
                   int* ncholpQ0,
                   std::complex<T1> const alpha,
                   std::complex<T2> const* v1,
                   std::complex<T3>* vb)
{
  int xblock_dim = 32;
  int yblock_dim = std::min(nwalk, 16);
  dim3 block_dim(xblock_dim, yblock_dim, 1);
  dim3 grid_dim(nkpts, 2, 1);
  kernel_vbias_from_v1<<<grid_dim, block_dim>>>(nwalk, nkpts, nchol_max, Qsym, kminus, ncholpQ, ncholpQ0,
                                                static_cast<thrust::complex<T1> const>(alpha),
                                                reinterpret_cast<thrust::complex<T2> const*>(v1),
                                                reinterpret_cast<thrust::complex<T3>*>(vb));
  qmc_cuda::cuda_check(cudaGetLastError(), "vbias_from_v1");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "vbias_from_v1");
}

template void vbias_from_v1(int,int,int,int*,int*,int*,int*,
	std::complex<float> const,std::complex<float> const*,std::complex<float>*);
template void vbias_from_v1(int,int,int,int*,int*,int*,int*,
	std::complex<float> const,std::complex<double> const*,std::complex<float>*);
template void vbias_from_v1(int,int,int,int*,int*,int*,int*,
	std::complex<double> const,std::complex<double> const*,std::complex<double>*);
template void vbias_from_v1(int,int,int,int*,int*,int*,int*,
	std::complex<double> const,std::complex<float> const*,std::complex<double>*);


} // namespace kernels
