////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef BATCHED_DOT_WABN_WBAN_H
#define BATCHED_DOT_WABN_WBAN_H

#include <cassert>
#include <complex>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"

namespace kernels
{
template<typename T1, typename T2>
void batched_dot_wabn_wban(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<T1> const* alpha,
                           std::complex<T1> const* Tab,
                           std::complex<T2>* y,
                           int incy);

template<typename T1, typename T2>
void batched_dot_wanb_wbna(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<T1> const* alpha,
                           std::complex<T1> const* Tab,
                           std::complex<T2>* y,
                           int incy);

} // namespace kernels

#endif
