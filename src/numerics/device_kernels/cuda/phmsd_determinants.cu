#include <complex>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "arch/arch.h"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"
#include "numerics/operations/determinants.hpp"

namespace kernels::device::detail
{
/*
template<class T>
__device__ T D3x3( T const a11, T const a12, T const a13, 
                   T const a21, T const a22, T const a23, 
                   T const a31, T const a32, T const a33)
{
  return   a11 * (a22 * a33 - a32 * a23)
         - a21 * (a12 * a33 - a32 * a13)
         + a31 * (a12 * a23 - a22 * a13);
}

//#ifdef NDEBUG
template<class T>
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

template<class T>
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
*/
//#endif

template<typename T_t, typename Ov_t>
void phmsd_det_impl(int nex, int const* iex, T_t const& T, Ov_t& ov)
{
  auto T_d = to_cuda_std_mdspan(T);
  auto ov_d = to_cuda_std_mdspan(ov);
  long ndet = ov.extent(0);
  long nwalk = ov.extent(1);
  switch(nex) 
  {
    case 1:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet; 
        long idet = n - iw*ndet; 
        ov_d(idet,iw) = T_d(iw,iex[2*idet+1],iex[2*idet]);
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;   
    }
    case 2:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 4*idet; 
        /*        | 20   21| 
         * D = det| 30   21|
         */
        ov_d(idet,iw) = T_d(iw,x[2],x[0])*T_d(iw,x[3],x[1]) - T_d(iw,x[2],x[1])*T_d(iw,x[3],x[0]);
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
/*
    case 3:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 6*idet; 
        ov_d(idet,iw) = D3x3(T_d(iw,x[3],x[0]),T_d(iw,x[3],x[1]),T_d(iw,x[3],x[2]),
                             T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]));
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
// MAM: not clear when the default implementation is faster, test and make a heuristic choice!
    case 4:
    {
      // at this point this is too much work for a single thread, consider evaluating sub-determinants on separate threads and then collecting results in shared memory
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 8*idet; 
        ov_d(idet,iw) = D4x4(T_d(iw,x[4],x[0]),T_d(iw,x[4],x[1]),T_d(iw,x[4],x[2]),T_d(iw,x[4],x[3]),
                             T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),
                             T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),         
                             T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3])); 
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 5:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 10*idet;
        ov_d(idet,iw) = D5x5(T_d(iw,x[5],x[0]),T_d(iw,x[5],x[1]),T_d(iw,x[5],x[2]),T_d(iw,x[5],x[3]),T_d(iw,x[5],x[4]),
                             T_d(iw,x[6],x[0]),T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),
                             T_d(iw,x[7],x[0]),T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),
                             T_d(iw,x[8],x[0]),T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),
                             T_d(iw,x[9],x[0]),T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]));
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
    case 6:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 12*idet;
        auto A0 = D5x5(T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),T_d(iw,x[7],x[5]),
                       T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),T_d(iw,x[8],x[5]),
                       T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]),T_d(iw,x[9],x[5]),
                       T_d(iw,x[10],x[1]),T_d(iw,x[10],x[2]),T_d(iw,x[10],x[3]),T_d(iw,x[10],x[4]),T_d(iw,x[10],x[5]),
                       T_d(iw,x[11],x[1]),T_d(iw,x[11],x[2]),T_d(iw,x[11],x[3]),T_d(iw,x[11],x[4]),T_d(iw,x[11],x[5]));
        auto A1 = D5x5(T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),T_d(iw,x[6],x[5]),
                       T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),T_d(iw,x[8],x[5]),
                       T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]),T_d(iw,x[9],x[5]),
                       T_d(iw,x[10],x[1]),T_d(iw,x[10],x[2]),T_d(iw,x[10],x[3]),T_d(iw,x[10],x[4]),T_d(iw,x[10],x[5]),
                       T_d(iw,x[11],x[1]),T_d(iw,x[11],x[2]),T_d(iw,x[11],x[3]),T_d(iw,x[11],x[4]),T_d(iw,x[11],x[5]));
        auto A2 = D5x5(T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),T_d(iw,x[6],x[5]),
                       T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),T_d(iw,x[7],x[5]),
                       T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]),T_d(iw,x[9],x[5]),
                       T_d(iw,x[10],x[1]),T_d(iw,x[10],x[2]),T_d(iw,x[10],x[3]),T_d(iw,x[10],x[4]),T_d(iw,x[10],x[5]),
                       T_d(iw,x[11],x[1]),T_d(iw,x[11],x[2]),T_d(iw,x[11],x[3]),T_d(iw,x[11],x[4]),T_d(iw,x[11],x[5]));
        auto A3 = D5x5(T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),T_d(iw,x[6],x[5]),
                       T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),T_d(iw,x[7],x[5]),
                       T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),T_d(iw,x[8],x[5]),
                       T_d(iw,x[10],x[1]),T_d(iw,x[10],x[2]),T_d(iw,x[10],x[3]),T_d(iw,x[10],x[4]),T_d(iw,x[10],x[5]),
                       T_d(iw,x[11],x[1]),T_d(iw,x[11],x[2]),T_d(iw,x[11],x[3]),T_d(iw,x[11],x[4]),T_d(iw,x[11],x[5]));
        auto A4 = D5x5(T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),T_d(iw,x[6],x[5]),
                       T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),T_d(iw,x[7],x[5]),
                       T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),T_d(iw,x[8],x[5]),
                       T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]),T_d(iw,x[9],x[5]),
                       T_d(iw,x[11],x[1]),T_d(iw,x[11],x[2]),T_d(iw,x[11],x[3]),T_d(iw,x[11],x[4]),T_d(iw,x[11],x[5]));
        auto A5 = D5x5(T_d(iw,x[6],x[1]),T_d(iw,x[6],x[2]),T_d(iw,x[6],x[3]),T_d(iw,x[6],x[4]),T_d(iw,x[6],x[5]),
                       T_d(iw,x[7],x[1]),T_d(iw,x[7],x[2]),T_d(iw,x[7],x[3]),T_d(iw,x[7],x[4]),T_d(iw,x[7],x[5]),
                       T_d(iw,x[8],x[1]),T_d(iw,x[8],x[2]),T_d(iw,x[8],x[3]),T_d(iw,x[8],x[4]),T_d(iw,x[8],x[5]),
                       T_d(iw,x[9],x[1]),T_d(iw,x[9],x[2]),T_d(iw,x[9],x[3]),T_d(iw,x[9],x[4]),T_d(iw,x[9],x[5]),
                       T_d(iw,x[10],x[1]),T_d(iw,x[10],x[2]),T_d(iw,x[10],x[3]),T_d(iw,x[10],x[4]),T_d(iw,x[10],x[5]));
        ov_d(idet,iw) =  T_d(iw,x[6],x[0])  * A0 
                       - T_d(iw,x[7],x[0])  * A1 
                       + T_d(iw,x[8],x[0])  * A2 
                       - T_d(iw,x[9],x[0])  * A3 
                       + T_d(iw,x[10],x[0]) * A4 
                       - T_d(iw,x[11],x[0]) * A5; 
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
*/
    default:
    {
      // M[iwalk][idet][ip][iq]
      memory::buffered_array<DEVICE_MEMORY,nda::get_value_t<T_t>,4> M(ndet,nwalk,nex,nex);
      auto M_d = to_cuda_std_mdspan(M);
      long n1 = nwalk*nex*nex;
      long n2 = nex*nex;
      auto extract = [=] __device__(long n) {
        long idet = n/n1;
        n -= idet*n1;
        long iw   = n/n2;
        n -= iw*n2;
        long i   = n/nex;
        long j   = n - i*nex; 
        M_d(idet,iw,i,j) = T_d(iw,iex[2*nex*idet + nex + i],iex[2*nex*idet + j]);    
      };
      cub::DeviceFor::Bulk(ndet*nwalk*nex*nex,extract);
      sfqmc::arch::synchronize();
      memory::buffered_array<DEVICE_MEMORY,int,2> ipiv(ndet*nwalk,nex);
      memory::buffered_array<DEVICE_MEMORY,nda::get_value_t<T_t>,1> work;
      auto M3d = nda::reshape(M,std::array<long,3>{ndet*nwalk,nex,nex});
      nda::lapack::getrf(M3d,ipiv,work);
      math::log_determinant_from_getrf(M3d,ipiv,nda::flatten(ov));
    }
  };
  sfqmc::arch::synchronize_if_set();
}

template<typename T_t, typename R_t> 
void phmsd_compact_R_impl(int nex, int const* refc, int const* iex, T_t const& T, R_t &R) 
{
  using nda::range;
  auto all = range::all;
  // T(nw,nact,nel)
  auto T_d = to_cuda_std_mdspan(T);
  // R(nwalk,ndet,nex,nact) 
  auto R_d = to_cuda_std_mdspan(R);
  using value_t = std::decay_t<decltype(T_d(0,0,0))>;
  long nwalk = R.extent(0);
  long ndet = R.extent(1);
  long nel = T.extent(2);
  long nact = R.extent(3);
  memory::buffered_array<DEVICE_MEMORY,ComplexType,2> ov(nwalk,ndet);
  ov() = ComplexType(0);
  auto ov_d = to_cuda_std_mdspan(ov);
  memory::buffered_array<DEVICE_MEMORY,ComplexType,4> M(nwalk,ndet,nex,nex);
  auto M_d = to_cuda_std_mdspan(M);
  // Compute inverses 
  switch(nex)
  {
    case 1:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        ov_d(iw,idet) = T_d(iw,iex[2*idet+1],iex[2*idet]);
        if(abs(ov_d(iw,idet)) != 0) 
          M_d(iw,idet,0,0) = value_t(1.0)/ov_d(iw,idet);
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
/*
    case 2:
    {
      auto f = [=] __device__(long n) {
        long iw   = n/ndet;
        long idet = n - iw*ndet;
        int const* x = iex + 4*idet;
        ov_d(idet,iw) = T_d(iw,x[2],x[0])*T_d(iw,x[3],x[1]) - T_d(iw,x[2],x[1])*T_d(iw,x[3],x[0]);
      };
      cub::DeviceFor::Bulk(nwalk*ndet,f);
      break;
    }
*/
    default:
    {
      // M[iwalk][idet][ip][iq]
      long n1 = ndet*nex*nex;
      long n2 = nex*nex;
      auto extract = [=] __device__(long n) {
        long iw   = n/n1;
        n -= iw*n1;
        long idet = n/n2;
        n -= idet*n2;
        long i   = n/nex;
        long j   = n - i*nex;
        M_d(iw,idet,i,j) = T_d(iw,iex[2*nex*idet + nex + i],iex[2*nex*idet + j]);
      };
      cub::DeviceFor::Bulk(nwalk*n1,extract);
      sfqmc::arch::synchronize();
      auto M3d = nda::reshape(M,std::array<long,3>{ndet*nwalk,nex,nex});
      memory::buffered_array<DEVICE_MEMORY,int,2> ipiv(ndet*nwalk,nex);
      memory::buffered_array<DEVICE_MEMORY,nda::get_value_t<T_t>,1> work;
      nda::lapack::getrf(M3d,ipiv,work);
      math::log_determinant_from_getrf(M3d,ipiv,nda::flatten(ov));
      sfqmc::arch::synchronize();
      nda::lapack::getri(M3d,ipiv,work);
    }
  }
  sfqmc::arch::synchronize();
  // construct compact R
  {
    long n1 = ndet*nex*nel;
    long n2 = nex*nel;
    auto f = [=] __device__(long n) {
      long iw = n/n1;
      n -= iw*n1;
      long idet = n/n2;
      n -= idet*n2;
      if(abs(ov_d(iw,idet)) == 0) return;
      long p = n/nel;
      int i = int(n-p*nel); 
      int a = refc[i];   
      auto iex_ = iex + idet*2*nex;
      for (int q = 0; q < nex; ++q)
        if(i == iex_[q]) 
          a = iex_[q+nex];
      for (int q = 0; q < nex; ++q) {
        R_d(iw,idet,p,a) -= M_d(iw,idet,p,q) * T_d(iw,iex_[q+nex],i);
        if(i == iex_[q]) 
          R_d(iw,idet,p,a) += M_d(iw,idet,p,q); 
      }
    };
    cub::DeviceFor::Bulk(nwalk*ndet*nex*nel,f);
  }
  sfqmc::arch::synchronize();
}

template<typename W_t, typename Rb_t, typename R_t>
void phmsd_reduce_R_impl(int nex, int const* refc, int const* iex, W_t const& wgt, Rb_t const& Rbuff, R_t &R)
{
  using nda::range;
  auto all = range::all;
  // Rbuff(nwalk,ndet,nex,nact) 
  auto Rb_d = to_cuda_std_mdspan(Rbuff);
  // R(nwalk,nel,nact) 
  auto R_d = to_cuda_std_mdspan(R);
  auto w_d = to_cuda_std_mdspan(wgt);
  long ndet = Rbuff.extent(1);
  long nwalk = R.extent(0);
  long nel = R.extent(1);
  long nact = R.extent(2);
  using value_t = std::decay_t<decltype(R_d(0,0,0))>;

  long n1 = ndet*nel*nact;
  long n2 = nel*nact;
  auto f = [=] __device__(long n) {
    long iw = n/n1;
    n -= iw*n1; 
    long idet = n/n2;
    n -= idet*n2;
    int i = n/nact;
    int a = n-i*nact;
    int orb_i = refc[i];   
    auto iex_ = iex + idet*2*nex;
    value_t y(0);
// MAM: loop over ~16 determinants, time and test
    //for()
    {
      for (int q = 0; q < nex; ++q) 
        if(i == iex_[q]) { 
          orb_i = iex_[q+nex];
          y += w_d(idet,iw) * Rb_d(iw,idet,q,a); 
        }
      if(a==orb_i) y += w_d(idet,iw);
    }
    double re   = y.real();
    double im   = y.imag();
    double* re_ = reinterpret_cast<double*>(&R_d(iw,i,a));
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);    
  };
  cub::DeviceFor::Bulk(nwalk*ndet*nel*nact,f);
  sfqmc::arch::synchronize();
}

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _inst_(T,V) \
template void phmsd_det_impl(int,int const*,V<const T,3,basic_layout_t<3>> const&,V<T,2,basic_layout_t<2>>&); \
template void phmsd_compact_R_impl(int,int const*,int const*,V<const T,3,basic_layout_t<3>> const&,V<T,4,basic_layout_t<4>>&); \
template void phmsd_reduce_R_impl(int,int const*,int const*,V<const T,2,basic_layout_t<2>>const&,V<const T,4,basic_layout_t<4>> const&,V<T,3,basic_layout_t<3>>&); \
template void phmsd_reduce_R_impl(int,int const*,int const*,V<const T,2,basic_layout_t<2>>const&,V<T,4,basic_layout_t<4>> const&,V<T,3,basic_layout_t<3>>&); 

_inst_(std::complex<double>,device_array_view)

} // kernels
