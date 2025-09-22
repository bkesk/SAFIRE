
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

#ifdef NDEBUG
namespace cuda_device_functions
{

using T = thrust::complex<double>;

//template<class T>
__device__ T D4x4( T const a11, T const a12, T const a13, T const a14, 
                   T const a21, T const a22, T const a23, T const a24, 
                   T const a31, T const a32, T const a33, T const a34, 
                   T const a41, T const a42, T const a43, T const a44 ) 
{
  return (a11 * (a22 * (a33 * a44 - a43 * a34) - a32 * (a23 * a44 - a43 * a24) + a42 * (a23 * a34 - a33 * a24)) -
          a21 * (a12 * (a33 * a44 - a43 * a34) - a32 * (a13 * a44 - a43 * a14) + a42 * (a13 * a34 - a33 * a14)) +
          a31 * (a12 * (a23 * a44 - a43 * a24) - a22 * (a13 * a44 - a43 * a14) + a42 * (a13 * a24 - a23 * a14)) -
          a41 * (a12 * (a23 * a34 - a33 * a24) - a22 * (a13 * a34 - a33 * a14) + a32 * (a13 * a24 - a23 * a14)));
}

//template<class T>
__device__  T D5x5( T const a11, T const a12, T const a13, T const a14, T const a15, 
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

//template<class T>
__device__  T D6x6(
        T const a11, T const a12, T const a13, T const a14, T const a15, T const a16,
        T const a21, T const a22, T const a23, T const a24, T const a25, T const a26,
        T const a31, T const a32, T const a33, T const a34, T const a35, T const a36,
        T const a41, T const a42, T const a43, T const a44, T const a45, T const a46,
        T const a51, T const a52, T const a53, T const a54, T const a55, T const a56,
        T const a61, T const a62, T const a63, T const a64, T const a65, T const a66)
{
  T D1 = D5x5( 
  a22,a23,a24,a25,a26,  
  a32,a33,a34,a35,a36,  
  a42,a43,a44,a45,a46,  
  a52,a53,a54,a55,a56,  
  a62,a63,a64,a65,a66);  
  T D2 = D5x5( 
  a12,a13,a14,a15,a16,
  a32,a33,a34,a35,a36,
  a42,a43,a44,a45,a46,
  a52,a53,a54,a55,a56,
  a62,a63,a64,a65,a66);
  T D3 = D5x5( 
  a12,a13,a14,a15,a16,
  a22,a23,a24,a25,a26,
  a42,a43,a44,a45,a46,
  a52,a53,a54,a55,a56,
  a62,a63,a64,a65,a66);
  T D4 = D5x5( 
  a12,a13,a14,a15,a16,
  a22,a23,a24,a25,a26,
  a32,a33,a34,a35,a36,
  a52,a53,a54,a55,a56,
  a62,a63,a64,a65,a66);
  T D5 = D5x5( 
  a12,a13,a14,a15,a16,
  a22,a23,a24,a25,a26,
  a32,a33,a34,a35,a36,
  a42,a43,a44,a45,a46,
  a62,a63,a64,a65,a66);
  T D6 = D5x5( 
  a12,a13,a14,a15,a16,
  a22,a23,a24,a25,a26,
  a32,a33,a34,a35,a36,
  a42,a43,a44,a45,a46,
  a52,a53,a54,a55,a56);
  return a11*D1 - a21*D2 + a31*D3 - a41*D4 + a51*D5 - a61*D6;
}

/*
//template<class T>
__device__  T D7x7(
        T const a11, T const a12, T const a13, T const a14, T const a15, T const a16, T const a17,
        T const a21, T const a22, T const a23, T const a24, T const a25, T const a26, T const a27,
        T const a31, T const a32, T const a33, T const a34, T const a35, T const a36, T const a37,
        T const a41, T const a42, T const a43, T const a44, T const a45, T const a46, T const a47,
        T const a51, T const a52, T const a53, T const a54, T const a55, T const a56, T const a57,
        T const a61, T const a62, T const a63, T const a64, T const a65, T const a66, T const a67,
        T const a71, T const a72, T const a73, T const a74, T const a75, T const a76, T const a77)
{
  T D1 = D6x6( 
  a22,a23,a24,a25,a26,a27,
  a32,a33,a34,a35,a36,a37,
  a42,a43,a44,a45,a46,a47,
  a52,a53,a54,a55,a56,a57,
  a62,a63,a64,a65,a66,a67,
  a72,a73,a74,a75,a76,a77);
  T D2 = D6x6(
  a12,a13,a14,a15,a16,a17,         
  a32,a33,a34,a35,a36,a37,         
  a42,a43,a44,a45,a46,a47,         
  a52,a53,a54,a55,a56,a57,         
  a62,a63,a64,a65,a66,a67,
  a72,a73,a74,a75,a76,a77);
  T D3 = D6x6(
  a12,a13,a14,a15,a16,a17,
  a22,a23,a24,a25,a26,a27,
  a42,a43,a44,a45,a46,a47,
  a52,a53,a54,a55,a56,a57,
  a62,a63,a64,a65,a66,a67,
  a72,a73,a74,a75,a76,a77);
  T D4 = D6x6(
  a12,a13,a14,a15,a16,a17,
  a22,a23,a24,a25,a26,a27,
  a32,a33,a34,a35,a36,a37,
  a52,a53,a54,a55,a56,a57,
  a62,a63,a64,a65,a66,a67,
  a72,a73,a74,a75,a76,a77);
  T D5 = D6x6(
  a12,a13,a14,a15,a16,a17,
  a22,a23,a24,a25,a26,a27,
  a32,a33,a34,a35,a36,a37,
  a42,a43,a44,a45,a46,a47,
  a62,a63,a64,a65,a66,a67,
  a72,a73,a74,a75,a76,a77);
  T D6 = D6x6(
  a12,a13,a14,a15,a16,a17,
  a22,a23,a24,a25,a26,a27,
  a32,a33,a34,a35,a36,a37,
  a42,a43,a44,a45,a46,a47,
  a52,a53,a54,a55,a56,a57,
  a72,a73,a74,a75,a76,a77);
  T D7 = D6x6(
  a12,a13,a14,a15,a16,a17,
  a22,a23,a24,a25,a26,a27,
  a32,a33,a34,a35,a36,a37,
  a42,a43,a44,a45,a46,a47,
  a52,a53,a54,a55,a56,a57,
  a62,a63,a64,a65,a66,a67);
  return a11*D1 - a21*D2 + a31*D3 - a41*D4 + a51*D5 - a61*D6 + a71*D7;
}
*/
}
#endif

namespace kernels
{
// ov[idet][iwalk] = T[iwalk][iexcit[idet][p+nex]][iexcit[idet][q]]
__global__ void kernel_det_singles(int nwalk,
                                   int ndet,
                                   int const* iexcit,
                                   thrust::complex<double> const* T,
                                   int ldT,
                                   long Tstride,
                                   thrust::complex<double>* ov,
                                   int ldo) 
{
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y; 
  if ( iwalk < nwalk and idet < ndet ) 
    ov[idet*ldo + iwalk] = T[iwalk*Tstride + iexcit[idet*2 + 1]*ldT + iexcit[idet*2]];
}

__global__ void kernel_det_doubles(int nwalk,
                                   int ndet,
                                   int const* iexcit,
                                   thrust::complex<double> const* T,
                                   int ldT,
                                   long Tstride,
                                   thrust::complex<double>* ov,
                                   int ldo)
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
    ov[idet*ldo + iwalk] = Tw[ie2*ldT + ie0] * Tw[ie3*ldT + ie1] 
                         - Tw[ie2*ldT + ie1] * Tw[ie3*ldT + ie0]; 
  }
}

__global__ void kernel_det_triples(int nwalk,
                                   int ndet,
                                   int const* iexcit,
                                   thrust::complex<double> const* T,
                                   int ldT,
                                   long Tstride,
                                   thrust::complex<double>* ov,
                                   int ldo)
{
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  {
    auto Tw(T + iwalk*Tstride);
    int ie0(iexcit[6*idet]);
    int ie1(iexcit[6*idet+1]);
    int ie2(iexcit[6*idet+2]);
    int ie3(iexcit[6*idet+3]);
    int ie4(iexcit[6*idet+4]);
    int ie5(iexcit[6*idet+5]);
    /*          30   31   32 
     * D = det| 40   41   42 |
     *          50   51   52 
     */ 
    auto D1 (Tw[ie4*ldT + ie1] * Tw[ie5*ldT + ie2] - Tw[ie5*ldT + ie1] * Tw[ie4*ldT + ie2]);
    auto D2 (Tw[ie3*ldT + ie1] * Tw[ie5*ldT + ie2] - Tw[ie5*ldT + ie1] * Tw[ie3*ldT + ie2]);
    auto D3 (Tw[ie3*ldT + ie1] * Tw[ie4*ldT + ie2] - Tw[ie3*ldT + ie2] * Tw[ie4*ldT + ie1]);
    ov[idet*ldo + iwalk] = Tw[ie3*ldT + ie0] * D1 
                          -Tw[ie4*ldT + ie0] * D2
                          +Tw[ie5*ldT + ie0] * D3;   
  }
}

// only compile in non debug mode, nvcc takes too long to compile in debug
#ifdef NDEBUG
__global__ void kernel_det_quads(int nwalk,
                                 int ndet,
                                 int const* iexcit,
                                 thrust::complex<double> const* T,
                                 int ldT,
                                 long Tstride,
                                 thrust::complex<double>* ov,
                                 int ldo)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<double>, 16*DOT_BLOCK_SIZE> cache;
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  { 
    auto Tw(T + iwalk*Tstride);
    int ie0(iexcit[8*idet]);
    int ie1(iexcit[8*idet+1]);
    int ie2(iexcit[8*idet+2]);
    int ie3(iexcit[8*idet+3]);
    int ie4(iexcit[8*idet+4]);
    int ie5(iexcit[8*idet+5]);
    int ie6(iexcit[8*idet+6]);
    int ie7(iexcit[8*idet+7]);
    auto A(cache + threadIdx.x*16);
    A[0] = Tw[ie4*ldT + ie0];
    A[1] = Tw[ie4*ldT + ie1];
    A[2] = Tw[ie4*ldT + ie2];
    A[3] = Tw[ie4*ldT + ie3];
    A[4] = Tw[ie5*ldT + ie0];
    A[5] = Tw[ie5*ldT + ie1];
    A[6] = Tw[ie5*ldT + ie2];
    A[7] = Tw[ie5*ldT + ie3];
    A[8] = Tw[ie6*ldT + ie0];
    A[9] = Tw[ie6*ldT + ie1];
    A[10] = Tw[ie6*ldT + ie2];
    A[11] = Tw[ie6*ldT + ie3];
    A[12] = Tw[ie7*ldT + ie0];
    A[13] = Tw[ie7*ldT + ie1];
    A[14] = Tw[ie7*ldT + ie2];
    A[15] = Tw[ie7*ldT + ie3];
    thrust::complex<double> D1 = A[5] *(A[10]*A[15] - A[11]*A[14]) 
                              -A[9] *(A[6] *A[15] - A[7] *A[14]) 
                              +A[13]*(A[6] *A[11] - A[7] *A[10]);
    thrust::complex<double> D2 = A[1] *(A[10]*A[15] - A[11]*A[14]) 
                              -A[9] *(A[2] *A[15] - A[3] *A[14]) 
                              +A[13]*(A[2] *A[11] - A[3] *A[10]);
    thrust::complex<double> D3 = A[1] *(A[6] *A[15] - A[7] *A[14])
                              -A[5] *(A[2] *A[15] - A[3] *A[14])
                              +A[13]*(A[2] *A[7]  - A[3] *A[6]);
    thrust::complex<double> D4 = A[1] *(A[6] *A[11] - A[7] *A[10])
                              -A[5] *(A[2] *A[11] - A[3] *A[10])
                              +A[9] *(A[2] *A[7]  - A[3] *A[6]);
    ov[idet*ldo + iwalk] = A[0] * D1 - A[4] * D2 + A[8] * D3 - A[12] * D4;
  }
}

__global__ void kernel_det_nex5(int nwalk,
                                 int ndet,
                                 int const* iexcit,
                                 thrust::complex<double> const* T,
                                 int ldT,
                                 long Tstride,
                                 thrust::complex<double>* ov,
                                 int ldo)
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
    thrust::complex<double> D1 = cuda_device_functions::D4x4(
            A[6],  A[7],  A[8],  A[9],
            A[11], A[12], A[13], A[14],
            A[16], A[17], A[18], A[19],
            A[21], A[22], A[23], A[24]); 
    thrust::complex<double> D2 = cuda_device_functions::D4x4(
            A[1], A[2], A[3], A[4],
            A[11], A[12], A[13], A[14],
            A[16], A[17], A[18], A[19],
            A[21], A[22], A[23], A[24]);
    thrust::complex<double> D3 = cuda_device_functions::D4x4(
            A[1], A[2], A[3], A[4],
            A[6],  A[7],  A[8],  A[9],
            A[16], A[17], A[18], A[19],
            A[21], A[22], A[23], A[24]);
    thrust::complex<double> D4 = cuda_device_functions::D4x4(
            A[1], A[2], A[3], A[4],
            A[6],  A[7],  A[8],  A[9],
            A[11], A[12], A[13], A[14],
            A[21], A[22], A[23], A[24]);
    thrust::complex<double> D5 = cuda_device_functions::D4x4(
            A[1], A[2], A[3], A[4],
            A[6],  A[7],  A[8],  A[9],
            A[11], A[12], A[13], A[14],
            A[16], A[17], A[18], A[19]);
    ov[idet*ldo + iwalk] = A[0] * D1 - A[5] * D2 + A[10] * D3 - A[15] * D4 + A[20] * D5;
  }
}

__global__ void kernel_det_nex6(int nwalk,
                                 int ndet,
                                 int const* iexcit,
                                 thrust::complex<double> const* T,
                                 int ldT,
                                 long Tstride,
                                 thrust::complex<double>* ov,
                                 int ldo)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<double>, 36*DOT_BLOCK_SIZE> cache;
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  { 
    auto Tw(T + iwalk*Tstride);
    int e0[6], e1[6];
    for(int i=0; i<6; i++) { 
      e0[i] = iexcit[12*idet+i];
      e1[i] = iexcit[12*idet+i+6];
    }
    auto A(cache + threadIdx.x*36);
    for(int i=0, ij=0; i<6; i++)
      for(int j=0; j<6; j++, ij++)
        A[ij] = Tw[e1[i]*ldT + e0[j]];
    thrust::complex<double> D1 = cuda_device_functions::D5x5(
            A[7],  A[8],  A[9],  A[10], A[11],
            A[13], A[14], A[15], A[16], A[17],
            A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29],
            A[31], A[32], A[33], A[34], A[35]);
    thrust::complex<double> D2 = cuda_device_functions::D5x5(
            A[1],  A[2],  A[3],  A[4],  A[5],
            A[13], A[14], A[15], A[16], A[17],
            A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29],
            A[31], A[32], A[33], A[34], A[35]);
    thrust::complex<double> D3 = cuda_device_functions::D5x5(
            A[1],  A[2],  A[3],  A[4],  A[5],
            A[7],  A[8],  A[9],  A[10], A[11],
            A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29],
            A[31], A[32], A[33], A[34], A[35]);
    thrust::complex<double> D4 = cuda_device_functions::D5x5(
            A[1],  A[2],  A[3],  A[4],  A[5],
            A[7],  A[8],  A[9],  A[10], A[11],
            A[13], A[14], A[15], A[16], A[17],
            A[25], A[26], A[27], A[28], A[29],
            A[31], A[32], A[33], A[34], A[35]);
    thrust::complex<double> D5 = cuda_device_functions::D5x5(
            A[1],  A[2],  A[3],  A[4],  A[5],
            A[7],  A[8],  A[9],  A[10], A[11],
            A[13], A[14], A[15], A[16], A[17],
            A[19], A[20], A[21], A[22], A[23],
            A[31], A[32], A[33], A[34], A[35]);
    thrust::complex<double> D6 = cuda_device_functions::D5x5(
            A[1],  A[2],  A[3],  A[4],  A[5],
            A[7],  A[8],  A[9],  A[10], A[11],
            A[13], A[14], A[15], A[16], A[17],
            A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29]);
    ov[idet*ldo + iwalk] = A[0] * D1 - A[6] * D2 + A[12] * D3 - A[18] * D4 + A[24] * D5 - A[30] * D6;
  }
}

__global__ void kernel_det_nex7(int nwalk,
                                 int ndet,
                                 int const* iexcit,
                                 thrust::complex<double> const* T,
                                 int ldT,
                                 long Tstride,
                                 thrust::complex<double>* ov,
                                 int ldo)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<double>, 49*DOT_BLOCK_SIZE> cache;
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  { 
    auto Tw(T + iwalk*Tstride);
    int e0[7], e1[7];
    for(int i=0; i<7; i++) { 
      e0[i] = iexcit[14*idet+i];
      e1[i] = iexcit[14*idet+i+7];
    }
    auto A(cache + threadIdx.x*49);
    for(int i=0, ij=0; i<7; i++)
      for(int j=0; j<7; j++, ij++)
        A[ij] = Tw[e1[i]*ldT + e0[j]];
    thrust::complex<double> D1 = cuda_device_functions::D6x6(
            A[8],  A[9],  A[10], A[11], A[12], A[13],
            A[15], A[16], A[17], A[18], A[19], A[20],
            A[22], A[23], A[24], A[25], A[26], A[27],
            A[29], A[30], A[31], A[32], A[33], A[34],
            A[36], A[37], A[38], A[39], A[40], A[41],
            A[43], A[44], A[45], A[46], A[47], A[48]);
    thrust::complex<double> D2 = cuda_device_functions::D6x6(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6],
            A[15], A[16], A[17], A[18], A[19], A[20],
            A[22], A[23], A[24], A[25], A[26], A[27],
            A[29], A[30], A[31], A[32], A[33], A[34],
            A[36], A[37], A[38], A[39], A[40], A[41],
            A[43], A[44], A[45], A[46], A[47], A[48]);
    thrust::complex<double> D3 = cuda_device_functions::D6x6(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6],
            A[8],  A[9],  A[10], A[11], A[12], A[13],
            A[22], A[23], A[24], A[25], A[26], A[27],
            A[29], A[30], A[31], A[32], A[33], A[34],
            A[36], A[37], A[38], A[39], A[40], A[41],
            A[43], A[44], A[45], A[46], A[47], A[48]);
    thrust::complex<double> D4 = cuda_device_functions::D6x6(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6],
            A[8],  A[9],  A[10], A[11], A[12], A[13],
            A[15], A[16], A[17], A[18], A[19], A[20],
            A[29], A[30], A[31], A[32], A[33], A[34],
            A[36], A[37], A[38], A[39], A[40], A[41],
            A[43], A[44], A[45], A[46], A[47], A[48]);
    thrust::complex<double> D5 = cuda_device_functions::D6x6(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6],
            A[8],  A[9],  A[10], A[11], A[12], A[13],
            A[15], A[16], A[17], A[18], A[19], A[20],
            A[22], A[23], A[24], A[25], A[26], A[27],
            A[36], A[37], A[38], A[39], A[40], A[41],
            A[43], A[44], A[45], A[46], A[47], A[48]);
    thrust::complex<double> D6 = cuda_device_functions::D6x6(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6],
            A[8],  A[9],  A[10], A[11], A[12], A[13],
            A[15], A[16], A[17], A[18], A[19], A[20],
            A[22], A[23], A[24], A[25], A[26], A[27],
            A[29], A[30], A[31], A[32], A[33], A[34],
            A[43], A[44], A[45], A[46], A[47], A[48]);
    thrust::complex<double> D7 = cuda_device_functions::D6x6(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6],
            A[8],  A[9],  A[10], A[11], A[12], A[13],
            A[15], A[16], A[17], A[18], A[19], A[20],
            A[22], A[23], A[24], A[25], A[26], A[27],
            A[29], A[30], A[31], A[32], A[33], A[34],
            A[36], A[37], A[38], A[39], A[40], A[41]);
    ov[idet*ldo + iwalk] = A[0] * D1 - A[7] * D2 + A[14] * D3 - A[21] * D4 + A[28] * D5 
                         - A[35] * D6 + A[42] * D7;
  }
}
#endif

/*
 * Right now nex8 is slower than the generic version using getrf.
 * This can probably be improved using more threads per determinant, 
 * but I'm not doing it right now. It takes a long time to compile D7x7 anyway.
 */
/*
__global__ void kernel_det_nex8(int nwalk,
                                 int ndet,
                                 int const* iexcit,
                                 thrust::complex<double> const* T,
                                 int ldT,
                                 long Tstride,
                                 thrust::complex<double>* ov,
                                 int ldo)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<double>, 64*DOT_BLOCK_SIZE> cache;
  int idet = blockIdx.x*blockDim.x + threadIdx.x;
  int iwalk = blockIdx.y;
  if ( iwalk < nwalk and idet < ndet )
  { 
    auto Tw(T + iwalk*Tstride);
    int e0[8], e1[8];
    for(int i=0; i<8; i++) { 
      e0[i] = iexcit[16*idet+i];
      e1[i] = iexcit[16*idet+i+8];
    }
    auto A(cache + threadIdx.x*64);
    for(int i=0, ij=0; i<8; i++)
      for(int j=0; j<8; j++, ij++)
        A[ij] = Tw[e1[i]*ldT + e0[j]];
    thrust::complex<double> D1 = cuda_device_functions::D7x7(
            A[9],  A[10], A[11], A[12], A[13], A[14], A[15],
            A[17], A[18], A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29], A[30], A[31],
            A[33], A[34], A[35], A[36], A[37], A[38], A[39],
            A[41], A[42], A[43], A[44], A[45], A[46], A[47],
            A[49], A[50], A[51], A[52], A[53], A[54], A[55],
            A[57], A[58], A[59], A[60], A[61], A[62], A[63]);
    thrust::complex<double> D2 = cuda_device_functions::D7x7(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6], A[7],
            A[17], A[18], A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29], A[30], A[31],
            A[33], A[34], A[35], A[36], A[37], A[38], A[39],
            A[41], A[42], A[43], A[44], A[45], A[46], A[47],
            A[49], A[50], A[51], A[52], A[53], A[54], A[55],
            A[57], A[58], A[59], A[60], A[61], A[62], A[63]);
    thrust::complex<double> D3 = cuda_device_functions::D7x7(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6], A[7],
            A[9],  A[10], A[11], A[12], A[13], A[14], A[15],
            A[25], A[26], A[27], A[28], A[29], A[30], A[31],
            A[33], A[34], A[35], A[36], A[37], A[38], A[39],
            A[41], A[42], A[43], A[44], A[45], A[46], A[47],
            A[49], A[50], A[51], A[52], A[53], A[54], A[55],
            A[57], A[58], A[59], A[60], A[61], A[62], A[63]);
    thrust::complex<double> D4 = cuda_device_functions::D7x7(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6], A[7],
            A[9],  A[10], A[11], A[12], A[13], A[14], A[15],
            A[17], A[18], A[19], A[20], A[21], A[22], A[23],
            A[33], A[34], A[35], A[36], A[37], A[38], A[39],
            A[41], A[42], A[43], A[44], A[45], A[46], A[47],
            A[49], A[50], A[51], A[52], A[53], A[54], A[55],
            A[57], A[58], A[59], A[60], A[61], A[62], A[63]);
    thrust::complex<double> D5 = cuda_device_functions::D7x7(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6], A[7],
            A[9],  A[10], A[11], A[12], A[13], A[14], A[15],
            A[17], A[18], A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29], A[30], A[31],
            A[41], A[42], A[43], A[44], A[45], A[46], A[47],
            A[49], A[50], A[51], A[52], A[53], A[54], A[55],
            A[57], A[58], A[59], A[60], A[61], A[62], A[63]);
    thrust::complex<double> D6 = cuda_device_functions::D7x7(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6], A[7],
            A[9],  A[10], A[11], A[12], A[13], A[14], A[15],
            A[17], A[18], A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29], A[30], A[31],
            A[33], A[34], A[35], A[36], A[37], A[38], A[39],
            A[49], A[50], A[51], A[52], A[53], A[54], A[55],
            A[57], A[58], A[59], A[60], A[61], A[62], A[63]);
    thrust::complex<double> D7 = cuda_device_functions::D7x7(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6], A[7],
            A[9],  A[10], A[11], A[12], A[13], A[14], A[15],
            A[17], A[18], A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29], A[30], A[31],
            A[33], A[34], A[35], A[36], A[37], A[38], A[39],
            A[41], A[42], A[43], A[44], A[45], A[46], A[47],
            A[57], A[58], A[59], A[60], A[61], A[62], A[63]);
    thrust::complex<double> D8 = cuda_device_functions::D7x7(
            A[1],  A[2],  A[3],  A[4],  A[5], A[6], A[7],
            A[9],  A[10], A[11], A[12], A[13], A[14], A[15],
            A[17], A[18], A[19], A[20], A[21], A[22], A[23],
            A[25], A[26], A[27], A[28], A[29], A[30], A[31],
            A[33], A[34], A[35], A[36], A[37], A[38], A[39],
            A[41], A[42], A[43], A[44], A[45], A[46], A[47],
            A[49], A[50], A[51], A[52], A[53], A[54], A[55]);
    ov[idet*ldo + iwalk] = A[0] * D1 - A[8] * D2 + A[16] * D3 - A[24] * D4 + A[32] * D5 
                         - A[40] * D6 + A[48] * D7 - A[56] * D8;
  }
}
*/

void phmsd_det_singles(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE; 
  dim3 grid_dim(grid_dim_x, nwalk, 1); 
  kernel_det_singles<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void phmsd_det_doubles(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;      
  dim3 grid_dim(grid_dim_x, nwalk, 1);
  kernel_det_doubles<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void phmsd_det_triples(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
  dim3 grid_dim(grid_dim_x, nwalk, 1);
  kernel_det_triples<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

#ifdef NDEBUG
void phmsd_det_quads(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE; 
  dim3 grid_dim(grid_dim_x, nwalk, 1);
  kernel_det_quads<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void phmsd_det_nex5(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
  dim3 grid_dim(grid_dim_x, nwalk, 1);
  kernel_det_nex5<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void phmsd_det_nex6(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
  dim3 grid_dim(grid_dim_x, nwalk, 1);
  kernel_det_nex6<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void phmsd_det_nex7(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
  dim3 grid_dim(grid_dim_x, nwalk, 1);
  kernel_det_nex7<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());      
  qmc_cuda::cuda_check(cudaDeviceSynchronize()); 
}
#endif

/*
void phmsd_det_nex8(int nwalk, int ndet, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  int grid_dim_x = (ndet + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
  dim3 grid_dim(grid_dim_x, nwalk, 1);
  kernel_det_nex8<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(nwalk, ndet, iexcit,
                                                 reinterpret_cast<thrust::complex<double> const*>(T),
                                                 ldT, Tstride,
                                                 reinterpret_cast<thrust::complex<double>*>(ov),
                                                 ldo);
  qmc_cuda::cuda_check(cudaGetLastError());      
  qmc_cuda::cuda_check(cudaDeviceSynchronize()); 
}
*/

void phmsd_det(int nwalk, int ndet, int nex, int const* iexcit, std::complex<double> const* T,
                       int ldT, long Tstride, std::complex<double>* ov, int ldo)
{
  switch(nex) 
  {
    case 1:
    {
      phmsd_det_singles(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
    case 2:
    {
      phmsd_det_doubles(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
    case 3:
    {
      phmsd_det_triples(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
#ifdef NDEBUG
    case 4:
    {
      phmsd_det_quads(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
    case 5:
    {
      phmsd_det_nex5(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
    case 6:
    {
      phmsd_det_nex6(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
    case 7:
    {
      phmsd_det_nex7(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
#endif
/*
    case 8:
    {
      phmsd_det_nex8(nwalk,ndet,iexcit,T,ldT,Tstride,ov,ldo);
      break;   
    }
*/
    default:
    {
      throw std::runtime_error("out of bounds"); 
    }
  }
}

} // kernels
