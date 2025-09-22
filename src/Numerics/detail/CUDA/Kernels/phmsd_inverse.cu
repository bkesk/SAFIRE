
#include <stdexcept>
#include <cassert>
#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <thrust/system/cuda/detail/core/util.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"


namespace cuda_device_functions
{

using T = thrust::complex<double>;

__device__ inline T _D3x3_( T const a11, T const a12, T const a13,
                     T const a21, T const a22, T const a23, 
                     T const a31, T const a32, T const a33)
{
  return a11 * (a22*a33-a23*a32) - a21 * (a12*a33-a13*a32) + a31 * (a12*a23-a13*a22);
}

#ifdef NDEBUG
__device__ inline T _D4x4_( T const a11, T const a12, T const a13, T const a14,
                            T const a21, T const a22, T const a23, T const a24,
                            T const a31, T const a32, T const a33, T const a34,
                            T const a41, T const a42, T const a43, T const a44 )
{
  return (a11 * (a22 * (a33 * a44 - a43 * a34) - a32 * (a23 * a44 - a43 * a24) + a42 * (a23 * a34 - a33 * a24)) -
          a21 * (a12 * (a33 * a44 - a43 * a34) - a32 * (a13 * a44 - a43 * a14) + a42 * (a13 * a34 - a33 * a14)) +
          a31 * (a12 * (a23 * a44 - a43 * a24) - a22 * (a13 * a44 - a43 * a14) + a42 * (a13 * a24 - a23 * a14)) -
          a41 * (a12 * (a23 * a34 - a33 * a24) - a22 * (a13 * a34 - a33 * a14) + a32 * (a13 * a24 - a23 * a14)));
}

__device__ inline T _D5x5_( T const a11, T const a12, T const a13, T const a14, T const a15,
                            T const a21, T const a22, T const a23, T const a24, T const a25,
                            T const a31, T const a32, T const a33, T const a34, T const a35,
                            T const a41, T const a42, T const a43, T const a44, T const a45,
                            T const a51, T const a52, T const a53, T const a54, T const a55 )
{
  return (a11 *
              (a22 * (a33 * (a44 * a55 - a54 * a45) - a43 * (a34 * a55 - a54 * a35) + a53 * (a34 * a45 - a44 * a35)) -
               a32 * (a23 * (a44 * a55 - a54 * a45) - a43 * (a24 * a55 - a54 * a25) + a53 * (a24 * a45 - a44 * a25)) +
               a42 * (a23 * (a34 * a55 - a54 * a35) - a33 * (a24 * a55 - a54 * a25) + a53 * (a24 * a35 - a34 * a25)) -
               a52 * (a23 * (a34 * a45 - a44 * a35) - a33 * (a24 * a45 - a44 * a25) + a43 * (a24 * a35 - a34 * a25))) -
          a21 *
              (a12 * (a33 * (a44 * a55 - a54 * a45) - a43 * (a34 * a55 - a54 * a35) + a53 * (a34 * a45 - a44 * a35)) -
               a32 * (a13 * (a44 * a55 - a54 * a45) - a43 * (a14 * a55 - a54 * a15) + a53 * (a14 * a45 - a44 * a15)) +
               a42 * (a13 * (a34 * a55 - a54 * a35) - a33 * (a14 * a55 - a54 * a15) + a53 * (a14 * a35 - a34 * a15)) -
               a52 * (a13 * (a34 * a45 - a44 * a35) - a33 * (a14 * a45 - a44 * a15) + a43 * (a14 * a35 - a34 * a15))) +
          a31 *
              (a12 * (a23 * (a44 * a55 - a54 * a45) - a43 * (a24 * a55 - a54 * a25) + a53 * (a24 * a45 - a44 * a25)) -
               a22 * (a13 * (a44 * a55 - a54 * a45) - a43 * (a14 * a55 - a54 * a15) + a53 * (a14 * a45 - a44 * a15)) +
               a42 * (a13 * (a24 * a55 - a54 * a25) - a23 * (a14 * a55 - a54 * a15) + a53 * (a14 * a25 - a24 * a15)) -
               a52 * (a13 * (a24 * a45 - a44 * a25) - a23 * (a14 * a45 - a44 * a15) + a43 * (a14 * a25 - a24 * a15))) -
          a41 *
              (a12 * (a23 * (a34 * a55 - a54 * a35) - a33 * (a24 * a55 - a54 * a25) + a53 * (a24 * a35 - a34 * a25)) -
               a22 * (a13 * (a34 * a55 - a54 * a35) - a33 * (a14 * a55 - a54 * a15) + a53 * (a14 * a35 - a34 * a15)) +
               a32 * (a13 * (a24 * a55 - a54 * a25) - a23 * (a14 * a55 - a54 * a15) + a53 * (a14 * a25 - a24 * a15)) -
               a52 * (a13 * (a24 * a35 - a34 * a25) - a23 * (a14 * a35 - a34 * a15) + a33 * (a14 * a25 - a24 * a15))) +
          a51 *
              (a12 * (a23 * (a34 * a45 - a44 * a35) - a33 * (a24 * a45 - a44 * a25) + a43 * (a24 * a35 - a34 * a25)) -
               a22 * (a13 * (a34 * a45 - a44 * a35) - a33 * (a14 * a45 - a44 * a15) + a43 * (a14 * a35 - a34 * a15)) +
               a32 * (a13 * (a24 * a45 - a44 * a25) - a23 * (a14 * a45 - a44 * a15) + a43 * (a14 * a25 - a24 * a15)) -
               a42 * (a13 * (a24 * a35 - a34 * a25) - a23 * (a14 * a35 - a34 * a15) + a33 * (a14 * a25 - a24 * a15))));
}
#endif

__device__ inline void I3x3(T const a11, T const a12, T const a13,
                            T const a21, T const a22, T const a23,
                            T const a31, T const a32, T const a33, T const scl, T* M)
{
  M[0] = (a22 * a33 - a32 * a23) * scl;
  M[1] = (a13 * a32 - a12 * a33) * scl;
  M[2] = (a12 * a23 - a13 * a22) * scl;
  M[3] = (a23 * a31 - a21 * a33) * scl;
  M[4] = (a11 * a33 - a13 * a31) * scl;
  M[5] = (a13 * a21 - a11 * a23) * scl;
  M[6] = (a21 * a32 - a22 * a31) * scl;
  M[7] = (a12 * a31 - a11 * a32) * scl;
  M[8] = (a11 * a22 - a12 * a21) * scl;
}

#ifdef NDEBUG
__device__ void I4x4(T const a11, T const a12, T const a13, T const a14,
                     T const a21, T const a22, T const a23, T const a24,
                     T const a31, T const a32, T const a33, T const a34, 
                     T const a41, T const a42, T const a43, T const a44, T const scl, T* M)
{
  M[0] = scl * _D3x3_( 
    a22, a23, a24,
    a32, a33, a34,
    a42, a43, a44
    ); 
  M[4] = -scl * _D3x3_( 
    a21, a23, a24,
    a31, a33, a34,
    a41, a43, a44
    );
  M[8] = scl * _D3x3_( 
    a21, a22, a24,
    a31, a32, a34,
    a41, a42, a44
    );
  M[12] = -scl * _D3x3_( 
    a21, a22, a23, 
    a31, a32, a33, 
    a41, a42, a43 
    );

  M[1] = -scl * _D3x3_( 
    a12, a13, a14,
    a32, a33, a34,
    a42, a43, a44
    );
  M[5] = scl * _D3x3_( 
    a11, a13, a14,
    a31, a33, a34,
    a41, a43, a44
    );
  M[9] = -scl * _D3x3_( 
    a11, a12, a14,
    a31, a32, a34,
    a41, a42, a44
    );
  M[13] = scl * _D3x3_( 
    a11, a12, a13, 
    a31, a32, a33,
    a41, a42, a43
    );

  M[2] = scl * _D3x3_( 
    a12, a13, a14,
    a22, a23, a24,
    a42, a43, a44
    );
  M[6] = -scl * _D3x3_( 
    a11, a13, a14,
    a21, a23, a24,
    a41, a43, a44
    );
  M[10] = scl * _D3x3_( 
    a11, a12, a14,
    a21, a22, a24,
    a41, a42, a44
    );
  M[14] = -scl * _D3x3_( 
    a11, a12, a13, 
    a21, a22, a23, 
    a41, a42, a43 
    );

  M[3] = -scl * _D3x3_( 
    a12, a13, a14,
    a22, a23, a24,
    a32, a33, a34
    );
  M[7] = scl * _D3x3_( 
    a11, a13, a14,
    a21, a23, a24,
    a31, a33, a34
    );
  M[11] = -scl * _D3x3_( 
    a11, a12, a14,
    a21, a22, a24,
    a31, a32, a34
    );
  M[15] = scl * _D3x3_( 
    a11, a12, a13, 
    a21, a22, a23, 
    a31, a32, a33 
    );
}

__device__ void I5x5(T const a11, T const a12, T const a13, T const a14, T const a15,
                     T const a21, T const a22, T const a23, T const a24, T const a25,
                     T const a31, T const a32, T const a33, T const a34, T const a35,      
                     T const a41, T const a42, T const a43, T const a44, T const a45,
                     T const a51, T const a52, T const a53, T const a54, T const a55,
                     T const scl, T* M)
{
  M[0] = scl * _D4x4_(
    a22,a23,a24,a25, 
    a32,a33,a34,a35, 
    a42,a43,a44,a45, 
    a52,a53,a54,a55 
    );
  M[5] = -scl * _D4x4_(
    a21,a23,a24,a25,
    a31,a33,a34,a35,
    a41,a43,a44,a45,
    a51,a53,a54,a55  
    );
  M[10] = scl * _D4x4_(
    a21,a22,a24,a25,
    a31,a32,a34,a35,
    a41,a42,a44,a45,
    a51,a52,a54,a55  
    );
  M[15] = -scl * _D4x4_(
    a21,a22,a23,a25,
    a31,a32,a33,a35,
    a41,a42,a43,a45,
    a51,a52,a53,a55  
    );
  M[20] = scl * _D4x4_(
    a21,a22,a23,a24,
    a31,a32,a33,a34,
    a41,a42,a43,a44,
    a51,a52,a53,a54 
    );

  M[1] = -scl * _D4x4_(
    a12,a13,a14,a15,
    a32,a33,a34,a35,
    a42,a43,a44,a45,
    a52,a53,a54,a55  
    );
  M[6] = scl * _D4x4_(
    a11,a13,a14,a15,
    a31,a33,a34,a35,
    a41,a43,a44,a45,
    a51,a53,a54,a55  
    );
  M[11] = -scl * _D4x4_(
    a11,a12,a14,a15,
    a31,a32,a34,a35,
    a41,a42,a44,a45,
    a51,a52,a54,a55  
    );
  M[16] = scl * _D4x4_(
    a11,a12,a13,a15,
    a31,a32,a33,a35,
    a41,a42,a43,a45,
    a51,a52,a53,a55  
    );
  M[21] = -scl * _D4x4_(
    a11,a12,a13,a14,
    a31,a32,a33,a34,
    a41,a42,a43,a44,
    a51,a52,a53,a54 
    );

  M[2] = scl * _D4x4_(
    a12,a13,a14,a15,
    a22,a23,a24,a25,
    a42,a43,a44,a45,
    a52,a53,a54,a55  
    );
  M[7] = -scl * _D4x4_(
    a11,a13,a14,a15,
    a21,a23,a24,a25,
    a41,a43,a44,a45,
    a51,a53,a54,a55  
    );
  M[12] = scl * _D4x4_(
    a11,a12,a14,a15,
    a21,a22,a24,a25,
    a41,a42,a44,a45,
    a51,a52,a54,a55  
    );
  M[17] = -scl * _D4x4_(
    a11,a12,a13,a15,
    a21,a22,a23,a25,
    a41,a42,a43,a45,
    a51,a52,a53,a55  
    );
  M[22] = scl * _D4x4_(
    a11,a12,a13,a14,
    a21,a22,a23,a24,
    a41,a42,a43,a44,
    a51,a52,a53,a54 
    );

  M[3] = -scl * _D4x4_(
    a12,a13,a14,a15,
    a22,a23,a24,a25,
    a32,a33,a34,a35,
    a52,a53,a54,a55  
    );
  M[8] = scl * _D4x4_(
    a11,a13,a14,a15,
    a21,a23,a24,a25,
    a31,a33,a34,a35,
    a51,a53,a54,a55  
    );
  M[13] = -scl * _D4x4_(
    a11,a12,a14,a15,
    a21,a22,a24,a25,
    a31,a32,a34,a35,
    a51,a52,a54,a55  
    );
  M[18] = scl * _D4x4_(
    a11,a12,a13,a15,
    a21,a22,a23,a25,
    a31,a32,a33,a35,
    a51,a52,a53,a55  
    );
  M[23] = -scl * _D4x4_(
    a11,a12,a13,a14,
    a21,a22,a23,a24,
    a31,a32,a33,a34,
    a51,a52,a53,a54 
    );

  M[4] = scl * _D4x4_(
    a12,a13,a14,a15,
    a22,a23,a24,a25,
    a32,a33,a34,a35,
    a42,a43,a44,a45
    );
  M[9] = -scl * _D4x4_(
    a11,a13,a14,a15,
    a21,a23,a24,a25,
    a31,a33,a34,a35,
    a41,a43,a44,a45
    );
  M[14] = scl * _D4x4_(
    a11,a12,a14,a15,
    a21,a22,a24,a25,
    a31,a32,a34,a35,
    a41,a42,a44,a45
    );
  M[19] = -scl * _D4x4_(
    a11,a12,a13,a15,
    a21,a22,a23,a25,
    a31,a32,a33,a35,
    a41,a42,a43,a45
    );
  M[24] = scl * _D4x4_(
    a11,a12,a13,a14,
    a21,a22,a23,a24,
    a31,a32,a33,a34,
    a41,a42,a43,a44
    );
}
#endif

}


namespace kernels
{
__global__ void kernel_inv_nex1(int nwalk, int ndet, int const* iexcit,
        thrust::complex<double> const* T, int ldT, long Tstride,
//        thrust::complex<double> const* ov, int ldo,
        thrust::complex<double>* M)
{
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y; 
  if ( iwalk < nwalk and idet < ndet ) {
    thrust::complex<double> det = T[iwalk*Tstride + iexcit[idet*2 + 1]*ldT + iexcit[idet*2]];
    if( abs(det) == 0.0 )
      M[(iwalk*ndet + idet)] = thrust::complex<double>(0.0);
    else
      M[(iwalk*ndet + idet)] = thrust::complex<double>(1.0) / det; 
  }  
}

__global__ void kernel_inv_nex2(int nwalk, int ndet, int const* iexcit,
        thrust::complex<double> const* T, int ldT, long Tstride,
//        thrust::complex<double> const* ov, int ldo,
        thrust::complex<double>* M)
{ 
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  { 
    auto Tw(T + iwalk*Tstride);
    int ie0(iexcit[4*idet]);
    int ie1(iexcit[4*idet+1]);
    int ie2(iexcit[4*idet+2]);
    int ie3(iexcit[4*idet+3]);
//    thrust::complex<double> scl = thrust::complex<double>(1.0) / ov[idet*ldo + iwalk];
    thrust::complex<double> a11 = Tw[ie2*ldT + ie0];
    thrust::complex<double> a12 = Tw[ie2*ldT + ie1];
    thrust::complex<double> a21 = Tw[ie3*ldT + ie0];
    thrust::complex<double> a22 = Tw[ie3*ldT + ie1];
    thrust::complex<double> scl = (a11*a22-a12*a21); 
    M += 4*(iwalk*ndet + idet);
    if( abs(scl) == 0.0 ) {
      M[0] = thrust::complex<double>(0.0);
      M[1] = thrust::complex<double>(0.0);
      M[2] = thrust::complex<double>(0.0);
      M[3] = thrust::complex<double>(0.0);
    } else {
      scl = thrust::complex<double>(1.0) / scl;
      M[0] =  a22 * scl; 
      M[1] = -a12 * scl; 
      M[2] = -a21 * scl; 
      M[3] =  a11 * scl; 
    }
  }
}

__global__ void kernel_inv_nex3(int nwalk, int ndet, int const* iexcit,
        thrust::complex<double> const* T, int ldT, long Tstride,
//        thrust::complex<double> const* ov, int ldo,
        thrust::complex<double>* M)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<double>, 9*DOT_BLOCK_SIZE> cache;
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  { 
    auto Tw(T + iwalk*Tstride);
    int e0[3], e1[3]; 
    for(int i=0; i<3; i++) { 
      e0[i] = iexcit[6*idet+i];
      e1[i] = iexcit[6*idet+i+3];
    }
    auto A(cache + threadIdx.x*9);
    for(int i=0, ij=0; i<3; i++)
      for(int j=0; j<3; j++, ij++)
        A[ij] = Tw[e1[i]*ldT + e0[j]];
//    thrust::complex<double> scl = thrust::complex<double>(1.0) / ov[idet*ldo + iwalk];
    thrust::complex<double> scl = cuda_device_functions::_D3x3_(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]); 
    M += 9*(iwalk*ndet + idet);
    if( abs(scl) == 0.0 ) {
      M[0] = thrust::complex<double>(0.0);
      M[1] = thrust::complex<double>(0.0);
      M[2] = thrust::complex<double>(0.0);
      M[3] = thrust::complex<double>(0.0);
      M[4] = thrust::complex<double>(0.0);
      M[5] = thrust::complex<double>(0.0);
      M[6] = thrust::complex<double>(0.0);
      M[7] = thrust::complex<double>(0.0);
      M[8] = thrust::complex<double>(0.0);
    } else {
      scl = thrust::complex<double>(1.0) / scl;
      cuda_device_functions::I3x3(A[0], A[1], A[2], 
                                A[3], A[4], A[5], 
                                A[6], A[7], A[8], scl, M);
    }
  }
}

#ifdef NDEBUG
__global__ void kernel_inv_nex4(int nwalk, int ndet, int const* iexcit,
        thrust::complex<double> const* T, int ldT, long Tstride,
//        thrust::complex<double> const* ov, int ldo,
        thrust::complex<double>* M)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<double>, 16*DOT_BLOCK_SIZE> cache;
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  {
    auto Tw(T + iwalk*Tstride);
    int e0[4], e1[4];
    for(int i=0; i<4; i++) {
      e0[i] = iexcit[8*idet+i];
      e1[i] = iexcit[8*idet+i+4];
    }
    auto A(cache + threadIdx.x*16);
    for(int i=0, ij=0; i<4; i++)
      for(int j=0; j<4; j++, ij++)
        A[ij] = Tw[e1[i]*ldT + e0[j]];
    thrust::complex<double> scl = cuda_device_functions::_D4x4_(A[0],  A[1],  A[2],  A[3], 
                                                                A[4],  A[5],  A[6],  A[7],
                                                                A[8],  A[9],  A[10], A[11], 
                                                                A[12], A[13], A[14], A[15]); 
    M += 16*(iwalk*ndet + idet);
    if( abs(scl) == 0.0 ) {
      M[0] = thrust::complex<double>(0.0);
      M[1] = thrust::complex<double>(0.0);
      M[2] = thrust::complex<double>(0.0);
      M[3] = thrust::complex<double>(0.0);
      M[4] = thrust::complex<double>(0.0);
      M[5] = thrust::complex<double>(0.0);
      M[6] = thrust::complex<double>(0.0);
      M[7] = thrust::complex<double>(0.0);
      M[8] = thrust::complex<double>(0.0);
      M[9] = thrust::complex<double>(0.0);
      M[10] = thrust::complex<double>(0.0);
      M[11] = thrust::complex<double>(0.0);
      M[12] = thrust::complex<double>(0.0);
      M[13] = thrust::complex<double>(0.0);
      M[14] = thrust::complex<double>(0.0);
      M[15] = thrust::complex<double>(0.0);
    } else {
      scl = thrust::complex<double>(1.0) / scl;
      cuda_device_functions::I4x4(A[0],  A[1],  A[2],  A[3],
                                A[4],  A[5],  A[6],  A[7],
                                A[8],  A[9],  A[10], A[11],
                                A[12], A[13], A[14], A[15], scl, M);
    }
  }
}

__global__ void kernel_inv_nex5(int nwalk, int ndet, int const* iexcit,
        thrust::complex<double> const* T, int ldT, long Tstride,
//        thrust::complex<double> const* ov, int ldo,
        thrust::complex<double>* M)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<double>, 25*DOT_BLOCK_SIZE> cache;
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  {
    auto Tw(T + iwalk*Tstride);
    int e0[5], e1[5];
    for(int i=0; i<5; i++) {
      e0[i] = iexcit[10*idet+i];
      e1[i] = iexcit[10*idet+i+5];
    }
    auto A(cache + threadIdx.x*25);
    for(int i=0, ij=0; i<5; i++)
      for(int j=0; j<5; j++, ij++)
        A[ij] = Tw[e1[i]*ldT + e0[j]];
//    thrust::complex<double> scl = thrust::complex<double>(1.0) / ov[idet*ldo + iwalk];
    thrust::complex<double> scl = cuda_device_functions::_D5x5_(A[0],  A[1],  A[2],  A[3],  A[4],
                                         A[5],  A[6],  A[7],  A[8],  A[9],
                                         A[10], A[11], A[12], A[13], A[14],
                                         A[15], A[16], A[17], A[18], A[19],
                                         A[20], A[21], A[22], A[23], A[24]);
    M += 25*(iwalk*ndet + idet);
    if( abs(scl) == 0.0 ) {
      M[0] = thrust::complex<double>(0.0);
      M[1] = thrust::complex<double>(0.0);
      M[2] = thrust::complex<double>(0.0);
      M[3] = thrust::complex<double>(0.0);
      M[4] = thrust::complex<double>(0.0);
      M[5] = thrust::complex<double>(0.0);
      M[6] = thrust::complex<double>(0.0);
      M[7] = thrust::complex<double>(0.0);
      M[8] = thrust::complex<double>(0.0);
      M[9] = thrust::complex<double>(0.0);
      M[10] = thrust::complex<double>(0.0);
      M[11] = thrust::complex<double>(0.0);
      M[12] = thrust::complex<double>(0.0);
      M[13] = thrust::complex<double>(0.0);
      M[14] = thrust::complex<double>(0.0);
      M[15] = thrust::complex<double>(0.0);
      M[16] = thrust::complex<double>(0.0);
      M[17] = thrust::complex<double>(0.0);
      M[18] = thrust::complex<double>(0.0);
      M[19] = thrust::complex<double>(0.0);
      M[20] = thrust::complex<double>(0.0);
      M[21] = thrust::complex<double>(0.0);
      M[22] = thrust::complex<double>(0.0);
      M[23] = thrust::complex<double>(0.0);
      M[24] = thrust::complex<double>(0.0);
    } else {
      scl = thrust::complex<double>(1.0) / scl; 
      cuda_device_functions::I5x5(A[0],  A[1],  A[2],  A[3],  A[4],
                                A[5],  A[6],  A[7],  A[8],  A[9],
                                A[10], A[11], A[12], A[13], A[14],
                                A[15], A[16], A[17], A[18], A[19],
                                A[20], A[21], A[22], A[23], A[24], scl, M);
    }
  }
}
#endif

void phmsd_inv(int nwalk, int ndet, int nex, int const* iexcit, std::complex<double> const* T,
               int ldT, long Tstride, //std::complex<double> const* ov, int ldo,
               std::complex<double>* Minv)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE; 
  dim3 grid_dim(grid_dim_x, nwalk, 1); 
  switch(nex)
  {
    case 1:
    {
      kernel_inv_nex1<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                    reinterpret_cast<thrust::complex<double> const*>(T),
                                    ldT, Tstride,
//                                    reinterpret_cast<thrust::complex<double> const*>(ov),ldo,
                                    reinterpret_cast<thrust::complex<double>*>(Minv));
      break;
    }
    case 2:
    {
      kernel_inv_nex2<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                    reinterpret_cast<thrust::complex<double> const*>(T),
                                    ldT, Tstride,
//                                    reinterpret_cast<thrust::complex<double> const*>(ov),ldo,
                                    reinterpret_cast<thrust::complex<double>*>(Minv));
      break;
    }
    case 3:
    {
      kernel_inv_nex3<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                    reinterpret_cast<thrust::complex<double> const*>(T),
                                    ldT, Tstride,
//                                    reinterpret_cast<thrust::complex<double> const*>(ov),ldo,
                                    reinterpret_cast<thrust::complex<double>*>(Minv));
      break;
    }
#ifdef NDEBUG
    case 4:
    {
      kernel_inv_nex4<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                    reinterpret_cast<thrust::complex<double> const*>(T),
                                    ldT, Tstride,
//                                    reinterpret_cast<thrust::complex<double> const*>(ov),ldo,
                                    reinterpret_cast<thrust::complex<double>*>(Minv));
      break;
    }
    case 5:
    {
      kernel_inv_nex5<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                    reinterpret_cast<thrust::complex<double> const*>(T),
                                    ldT, Tstride,
//                                    reinterpret_cast<thrust::complex<double> const*>(ov),ldo,
                                    reinterpret_cast<thrust::complex<double>*>(Minv));
      break;
    }
#endif
    default:
    {
      throw std::runtime_error("out of bounds");
    } 
  }
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

} // kernels
