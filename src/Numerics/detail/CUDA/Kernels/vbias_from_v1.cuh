////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef VBIAS_FROM_V1_H
#define VBIAS_FROM_V1_H

#include <cassert>
#include <complex>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"

namespace kernels
{
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
                   std::complex<T3>* vb);
} // namespace kernels

#endif
