////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef DOT_WABN_H
#define DOT_WABN_H

#include <cassert>
#include <complex>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"

namespace kernels
{
void dot_wabn(int nwalk,
              int nocc,
              int nchol,
              std::complex<double> const alpha,
              std::complex<double> const* Tab,
              std::complex<double>* y,
              int incy);
void dot_wabn(int nwalk,
              int nocc,
              int nchol,
              std::complex<float> const alpha,
              std::complex<float> const* Tab,
              std::complex<float>* y,
              int incy);
void dot_wabn(int nwalk,
              int nocc,
              int nchol,
              std::complex<float> const alpha,
              std::complex<float> const* Tab,
              std::complex<double>* y,
              int incy);


void dot_wanb(int nwalk,
              int nocc,
              int nact,
              int nchol,
              std::complex<double> const alpha,
              std::complex<double> const* Tab,
              std::complex<double>* y,
              int incy);
void dot_wanb(int nwalk,
              int nocc,
              int nact,
              int nchol,
              std::complex<float> const alpha,
              std::complex<float> const* Tab,
              std::complex<float>* y,
              int incy);
void dot_wanb(int nwalk,
              int nocc,
              int nact,
              int nchol,
              std::complex<float> const alpha,
              std::complex<float> const* Tab,
              std::complex<double>* y,
              int incy);


void dot_wpan_waqn_Fwpq(int nwalk,
                        int nmo,
                        int nchol,
                        std::complex<double> const alpha,
                        std::complex<double> const* Tab,
                        std::complex<double>* F);
void dot_wpan_waqn_Fwpq(int nwalk,
                        int nmo,
                        int nchol,
                        std::complex<float> const alpha,
                        std::complex<float> const* Tab,
                        std::complex<double>* F);
void dot_wpan_waqn_Fwpq(int nwalk,
                        int nmo,
                        int nchol,
                        std::complex<float> const alpha,
                        std::complex<float> const* Tab,
                        std::complex<float>* F);

} // namespace kernels

#endif
