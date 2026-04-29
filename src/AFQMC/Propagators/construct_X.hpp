
#pragma once

#if defined(__CUDACC__)
#include <cuda/std/mdspan>
#include <cuda/std/complex>
#else
#include <complex>
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#endif
#include "AFQMC/Utilities/probit.h"

namespace sfqmc::afqmc::detail
{

template<typename V1, typename V2, typename V3, typename V4, typename V5>
struct construct_X_impl
{
  bool zero;
  bool free_projection;
  double sqrtdt;
  double vbias_bound;
  /*
   * 0: ContinuousChargePropagator,
   * 1: ContinuousSpinPropagator,
   * 2: DiscreteChargePropagator,
   * 3: DiscreteSpinPropagator,
   */     
  V1 FieldTypes;
  V2 vMF;
  V3 MF;
  V3 HWs;
  V4 RNs;
  V5 X;  

  #if defined(__CUDACC__)
    __device__
  #endif
  void operator()(long nn)
  {
#if defined(__CUDACC__)
    using ::cuda::std::complex;
#else
    using std::abs;
    using std::log;
    using std::exp;
    using std::complex;
    using std::real;
    using std::imag;
#endif
    // nn = nCV*nwalk
    complex<double> im(0.0,1.0);
    long m = nn/X.extent(0);
    long iw = nn%X.extent(0);
    auto vmf_0 = vMF(m);
    auto vmf_t = (abs(vmf_0) > vbias_bound * sqrtdt ?
                  vmf_0 / (abs(vmf_0) / vbias_bound * sqrtdt) : vmf_0 );
    // X[iw,m] = rand[iw,m] + im * ( vbias[iw,m] - vMF[m]  )
    // HW[iw] = sum_m [ im * ( vMF[m] - vbias[iw,m] ) *
    //                     ( rand[iw,m] + halfim * ( vbias[iw,m] - vMF[m] ) ) ]
    //           = sum_m [ im * ( vMF[m] - vbias[iw,m] ) *
    //                     ( X[iw,m] - halfim * ( vbias[iw,m] - vMF[m] ) ) ]
    // MF[iw] = sum_m ( im * X[iw,m] * vMF[m] )
    {
      auto vb_t = (abs(X(iw,m)) > vbias_bound * sqrtdt ?
                    X(iw,m) / (abs(X(iw,m)) / vbias_bound * sqrtdt) : X(iw,m) );

      if (zero) {
        if (FieldTypes(m) == 1) {
          vb_t = complex<double>(0.0,imag(vb_t));
        } else if (FieldTypes(m) == 0) {
          vb_t = complex<double>(real(vb_t),0.0);
        }
      }

      if( FieldTypes(m) == 0 or FieldTypes(m) == 1 ) { 
        complex<double> vdiff =
            free_projection ? complex<double>(0.0, 0.0) : (im * (vb_t - vmf_t));
        X(iw,m) = probit(RNs(iw,m)) + vdiff;
        HWs(iw) -= vdiff * X(iw,m) - complex<double>(0.5) * vdiff;
        MF(iw) += im * X(iw,m) * vmf_0;
      } else if( FieldTypes(m) == 2 or FieldTypes(m) == 3 ) { 
        complex<double> vdiff =
            free_projection ? complex<double>(0.0, 0.0) : (im * (vb_t - vmf_t));
        auto wp = abs(exp(vdiff));
        auto wm = abs(exp(-vdiff));
        auto P = wm/(wp+wm);
        if( RNs(iw,m) < P ) {
          X(iw,m) = complex<double>(-1.0);
          HWs(iw) += log(1.0/P);
        } else {
          X(iw,m) = complex<double>(1.0);
          HWs(iw) += log(1.0/(1.0-P));
        }
        // W_MSsub = exp(-MF), careful with sign convention
        MF(iw) += im * complex<double>(X(iw,m)) * vmf_0;
      }
    } // iw
  };
};

} // sfqmc::afqmc::utils
